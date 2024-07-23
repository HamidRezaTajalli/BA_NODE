import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ncps.torch.ltc import LTC

class SequentialMNISTDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(SequentialMNISTDataLoader, self).__init__(*args, **kwargs)

    def __iter__(self):
        for batch in super(SequentialMNISTDataLoader, self).__iter__():
            yield batch[0].view(batch[0].size(0), -1, 1), batch[1]

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 16
learning_rate = 0.01
hidden_units = 32
num_epochs = 1

# Create data loaders
train_loader = SequentialMNISTDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = SequentialMNISTDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)


if not torch.cuda.is_available():
    raise Exception("CUDA is not available. Please check your GPU settings.")
    sys.exit(1)

# Training loop
for epoch in range(num_epochs):  # Number of epochs
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# Testing loop
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()  # Sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))