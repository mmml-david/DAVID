#!/bin/bash
#SBATCH --job-name=extract_features
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=48:00:00                
#SBATCH --output=/home/hsuanhal/logs/slurm-%j.log
#SBATCH --error=/home/hsuanhal/logs/slurm-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hsuanhal@andrew.cmu.edu

# --- 1. Environment Setup (CRITICAL) ---
source ~/.bashrc
cd ~/11777/DAVID
source .venv/bin/activate

# Recommended environment tuning to reduce CUDA OOMs and allocator fragmentation
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.6"
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
# Activate the Virtual Environment (This ensures 'python' finds your installed packages)
# NOTE: Replace the path with the actual location of your venv if using one.
# python3 -m venv venv
# source venv/bin/activate
# pip install -r scripts/requirements.txt

# --- 2. Main Execution ---

echo "Venv activated. Running extract_features..."
uv run python extract_features.py --shard_id 0 --config configs/train_config.yaml
echo "Job finished."
