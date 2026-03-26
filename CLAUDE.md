# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DAVID** (Dynamic Adaptive Video Inference and Decoding) is a research project implementing a hierarchical video VAE. It encodes videos via a frozen Qwen3-VL backbone into a fixed set of 64 latent tokens with stochastic prefix truncation, enabling adaptive inference at variable computational budgets.

## Setup

```bash
uv sync
source .venv/bin/activate
```

## Common Commands

```bash
# Smoke test (validates full pipeline end-to-end, 2 steps, no logging)
python train.py --config configs/train_config.yaml --smoke_test

# Feature extraction (one-time; supports sharding for distributed runs)
python extract_features.py --config configs/train_config.yaml
python extract_features.py --config configs/train_config.yaml --shard_id 0 --num_shards 8

# Training on cached features (recommended)
python train.py --config configs/train_config.yaml

# Resume training from checkpoint
python train.py --config configs/train_config.yaml --resume

# Training in online mode (slower; runs backbone on-the-fly)
python train.py --config configs/train_config.yaml --online
```

CLI flags override YAML config values (e.g. `--device cuda:1 --max_samples 100`).

## Architecture

The pipeline has two stages:

**Stage 1 — Feature Extraction** (`extract_features.py`):
- Loads PerceptionTest videos from HuggingFace, deduplicates (~30k rows → ~11.6k unique videos)
- Runs frozen `Qwen3VLBackbone` (Qwen3-VL-8B vision encoder, outputs `[N, 4096]` pooler tokens at 1 FPS)
- Caches features as `.pt` files (float16) for fast training

**Stage 2 — VAE Training** (`train.py`):
- `DAVIDEncoder`: 64 learned query vectors cross-attend to N input tokens → `[B, 64, 2048]` (mu, logvar)
- Reparameterization → z, then **stochastic prefix truncation**: sample m ~ Uniform(1, 64), keep only `z[:, :m, :]`
- `DAVIDDecoder`: 256 learned positional queries cross-attend to truncated prefix → `[B, 256, 4096]` reconstructed features
- Loss: MSE(recon, interpolated targets) + beta * KL, with linear beta warm-up

**Key modules:**
- `david/backbone.py` — `Qwen3VLBackbone`: wraps frozen vision encoder, returns padded features + masks
- `david/vae.py` — `DAVIDVAE`, `DAVIDEncoder`, `DAVIDDecoder`, `CrossAttentionBlock`, `DAVIDConfig`
- `david/dataset.py` — `PerceptionTestVideoDataset` (cached and online modes, custom collate for variable-length padding)
- `david/loss.py` — `david_loss()`, `BetaScheduler`, `reconstruction_loss()`, `kl_loss()`
- `david/utils.py` — padding/masking utilities, `interpolate_features()`

## Configuration

All settings live in `configs/train_config.yaml` with five sections: `model`, `training`, `data`, `extraction`, `logging`.

Key defaults to be aware of:
- `data.feature_cache_dir` and checkpoint paths default to `/data/user_data/hsuanhal/11777/DAVID/` (research cluster paths — override for local runs)
- `model.L = 64` (number of latent tokens), `model.N_queries = 256` (decoder output tokens), `model.input_dim = 4096`
- `training.batch_size = 8`, `gradient_accumulation_steps = 4`, `beta_target = 1e-4`, `beta_warmup_steps = 10000`
- SLURM job template: `scripts/job.sh` (targets L40S GPU, uses `<SHARD_ID>` placeholder)
