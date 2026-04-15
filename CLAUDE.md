# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DAVID** (Dynamic Adaptive Video Inference and Decoding) is a research project implementing a hierarchical video VAE. It encodes videos via a frozen Qwen3-VL-8B backbone into a variable-length sequence of latent tokens with stochastic prefix truncation, enabling adaptive inference at variable computational budgets.

## Setup

```bash
uv sync
source .venv/bin/activate
```

## Common Commands

```bash
# Smoke test (validates full pipeline end-to-end using synthetic data, 2 steps, no logging)
python train.py --config configs/train_config.yaml --smoke_test

# Feature extraction (one-time; supports sharding for distributed runs)
python extract_features.py --config configs/train_config.yaml --shard_id 0
# num_shards is set in config (currently 8); shard_id must be provided explicitly when num_shards > 1

# Single-GPU training on cached features
python train.py --config configs/train_config.yaml

# Multi-GPU training via torchrun (e.g. 4 GPUs)
torchrun --nproc_per_node=4 train.py --config configs/train_config.yaml

# Resume training from latest checkpoint
python train.py --config configs/train_config.yaml --resume

# Training in online mode (slower; runs backbone on-the-fly)
python train.py --config configs/train_config.yaml --online

# Override WandB run name
python train.py --config configs/train_config.yaml --run_name my-experiment

# VQA evaluation (Qwen3-VL baseline or DAVID+VAE)
python evaluate_vqa.py --questions_json perception_test.json --method qwen --max_samples 0
python evaluate_vqa.py --questions_json perception_test.json --method both --vae_checkpoint ./checkpoints/best.pt
```

CLI flags override YAML config values (e.g. `--device cuda:1 --max_samples 100`).

## Architecture

The pipeline has two stages:

**Stage 1 — Feature Extraction** (`extract_features.py`):
- Loads PerceptionTest videos from HuggingFace, deduplicates (~30k rows → ~11.6k unique videos by video filename stem)
- Runs frozen `Qwen3VLBackbone` (Qwen3-VL-8B vision encoder only — LLM weights are loaded on CPU then discarded to save ~16 GB VRAM)
- Videos sampled at 1 fps (cap: 64 frames); output shape `[N, 4096]` where N varies with video duration
- Caches features as `{video_name}.pt` files (float16, no mask saved — reconstructed as all-True at load time)

**Stage 2 — VAE Training** (`train.py`):
- `DAVIDEncoder`: self-attention blocks over the full `[B, N, D]` input (with optional progressive masking); outputs `mu, logvar` at same shape `[B, N, D]`
- Reparameterization → `z [B, N, D]`, then **stochastic prefix truncation**: sample `m ~ Uniform(1, N)`, zero-pad positions `m..N` in z
- `DAVIDDecoder`: self-attention blocks over zero-padded z; outputs reconstructed features `[B, N, D]`
- Loss: adaptive `recon_loss(m, N) + beta * KL`, where recon_loss exponent and scale both = `2*sigmoid(m/N)` (L1 for small m, L2 for large m)

**Training** (`train.py`):
- Single-GPU or multi-GPU via `torchrun` (DDP with `DistributedSampler`; `m` broadcast from rank 0)
- `bfloat16` autocast wraps forward + loss for memory savings
- EMA (`EMAModel`, decay=0.999) updated after each optimizer step; checkpointed alongside model
- Validation runs every `eval_every` steps on the validation split (capped at `max_val_samples`); saves `best.pt` on improvement
- WandB logging (entity `mmml-david`, project `DAVID`); `--run_name` CLI flag overrides `logging.run_name` in config
- `python-dotenv` loaded at startup (place API keys in `.env`)

**Key modules:**
- `david/backbone.py` — `Qwen3VLBackbone`: loads only the vision encoder onto GPU (LLM freed via `del model; gc.collect()`); forward unpacks tuple `(hidden_states, _deepstack)`
- `david/vae.py` — `DAVIDVAE`, `DAVIDEncoder`, `DAVIDDecoder`, `SelfAttentionBlock`, `DAVIDConfig`, `progressive_attn_mask()`
- `david/dataset.py` — `PerceptionTestVideoDataset` (cached and online modes; `min_pixels`/`max_pixels` for online resolution control)
- `david/loss.py` — `david_loss()`, `BetaScheduler`, `reconstruction_loss()`, `kl_loss()`
- `david/utils.py` — `pad_sequence_to_max()`, `interpolate_features()`, `EMAModel`
- `evaluate_vqa.py` — VQA accuracy evaluation on PerceptionTest; supports `qwen`-only or `both` (DAVID+VAE) modes; reads questions from `perception_test.json`

## Configuration

All settings live in `configs/train_config.yaml` with five sections: `model`, `training`, `data`, `extraction`, `logging`.

Key defaults to be aware of:
- `data.feature_cache_dir` and checkpoint paths default to `/data/user_data/hsuanhal/11777/DAVID/` (research cluster paths — override for local runs)
- `model.input_dim = 4096` (Qwen3-VL-8B hidden dim; 2048 for 2B model — use `configs/train_config_2b_online.yaml`)
- `model.progressive_ratio = 0.0` (0.0 = no progressive masking; increase toward 1.0 to enforce stronger ordering)
- `model.grad_checkpoint = true` — gradient checkpointing enabled by default
- `training.batch_size = 4`, `gradient_accumulation_steps = 4`, `beta_target = 1e-4`, `beta_warmup_steps = 10000`, `ema_decay = 0.999`
- `logging.eval_every = 1000`, `logging.max_val_samples` — cap validation batches (set to 30 in 2B config)
- `data.min_pixels` / `data.max_pixels` — pixel budget for online mode frame resizing (128×128 to 320×640 in 2B config)
- `extraction.num_shards = 8` — must set `--shard_id` explicitly on each machine
