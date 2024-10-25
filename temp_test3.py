import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from ncps.torch.ltc import LTC
import torch.multiprocessing as mp
import torch.distributed as dist

class SequentialMNISTDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(SequentialMNISTDataLoader, self).__init__(*args, **kwargs)

    def __iter__(self):
        for batch in super(SequentialMNISTDataLoader, self).__iter__():
            yield batch[0].view(batch[0].size(0), -1, 1), batch[1]

# Initialize the process group
def setup(rank, world_size):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, init_method="file:///scratch-shared/bhrd_sharedfile", world_size=world_size)
    print("Process group initialized successfully.")

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # Hyperparameters
    batch_size = 16
    learning_rate = 0.01
    hidden_units = 32
    num_epochs = 1

    # Create data loaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_loader = SequentialMNISTDataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler)
    test_loader = SequentialMNISTDataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler)

    class MNISTModel(nn.Module):
        def __init__(self, input_size, units):
            super(MNISTModel, self).__init__()
            self.ltc = LTC(input_size, units)
            self.fc = nn.Linear(units, 10)  # Add a fully connected layer to map to 10 classes

        def forward(self, x):
            output, _ = self.ltc(x)
            output = self.fc(output[:, -1, :])  # Apply the fully connected layer to the last output of the LTC
            return output

    # Initialize the LTC model
    model = MNISTModel(1, hidden_units)  # Assuming input size is 1 (MNIST images are unrolled into 1D sequence)
    model.to(rank)
    model = DDP(model, device_ids=[rank])

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)

    # Training loop
    for epoch in range(num_epochs):  # Number of epochs
        train_sampler.set_epoch(epoch)  # Shuffle data at every epoch
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(rank), target.to(rank)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0 and rank == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    # Testing loop
    if rank == 0:
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(rank), target.to(rank)
                output = model(data)
                test_loss += criterion(output, target).item()  # Sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    cleanup()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    # Remove the shared file used for process group initialization
    if rank == 0:
        import os
        try:
            os.remove("file:///scratch-shared/bhrd_sharedfile")
            print("Shared file for process group initialization removed successfully.")
        except OSError as e:
            print(f"Error removing shared file: {e}")

if __name__ == '__main__':
    main()
