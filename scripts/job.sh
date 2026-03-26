#!/bin/bash
#SBATCH --job-name=extract_features
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=48:00:00
#SBATCH --output=/home/<USER>/logs/slurm-%j.log
#SBATCH --error=/home/<USER>/logs/slurm-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=<USER>@andrew.cmu.edu

# --- 1. Environment Setup ---

export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.6"
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800


echo "Running extract_features..."

uv run python extract_features.py --shard_id <SHARD_ID> --config configs/train_config.yaml

echo "Job finished."
