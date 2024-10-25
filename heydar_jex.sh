#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --time=4-23:00:00
#SBATCH --output=script_logging/slurm_%A.out
#SBATCH --mail-type=END,FAIL                     # send email when job ends or fails
#SBATCH --mail-user=hamidreza.tajalli@ru.nl      # email address



# Loading modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

srun torchrun --standalone --nproc_per_node=4 /home/htajalli/prjs0962/repos/BA_NODE/temp_test_torchrun.py

# # pip install git+https://github.com/huggingface/transformers
# srun $HOME/TAConvDR/component3_retriever/bm25/evaluation.py \
#     --index_dir_path "corpus/indexes" \
#     --result_qrel_path "component3_retriever/results" \
#     --gold_qrel_path "component3_retriever/data/topiocqa/dev/qrel_gold.trec" \
#     --dataset_name "topiocqa" \
#     --query_format "human_rewritten" \
#     --seed 42


# # dataset_name = ["topiocqa", "inscit", "qrecc"]
# # query_format = ['original', 'human_rewritten', 'all_history', 'same_topic']