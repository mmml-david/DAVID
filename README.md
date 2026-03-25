# DAVID — Dynamic Adaptive Video Inference and Decoding

A hierarchical video representation that encodes a video as a prefix-ordered sequence of semantic tokens, built on top of a frozen Qwen3-VL vision backbone.

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Workflow

### 1. Extract features (one-time)

Run all videos through the frozen Qwen3-VL backbone and cache the features as `.pt` files. This only needs to be done once before training.

```bash
python extract_features.py \
    --cache_dir ./features_cache \
    --split train \
    --max_frames 8 \
    --model_name Qwen/Qwen3-VL-2B-Instruct \
    --device cuda
```

Use `--max_samples N` to limit to N videos (useful for testing).

### 2. Smoke test

Verify the full pipeline (shapes, loss computation) with a single batch and no WandB logging:

```bash
python train.py --config configs/train_config.yaml --smoke_test
```

### 3. Train

```bash
python train.py --config configs/train_config.yaml
```

Resume from the latest checkpoint:

```bash
python train.py --config configs/train_config.yaml --resume
```

Train in online mode (extracts features during training — slower, no cache needed):

```bash
python train.py --config configs/train_config.yaml --online
```

## Project Structure

```
DAVID/
├── requirements.txt
├── configs/
│   └── train_config.yaml       # All hyperparameters
├── david/
│   ├── backbone.py             # Frozen Qwen3-VL feature extractor
│   ├── vae.py                  # DAVIDEncoder + DAVIDDecoder + DAVIDVAE
│   ├── dataset.py              # Dataset loader (cached and online modes)
│   ├── loss.py                 # MSE + KL loss + beta warm-up scheduler
│   └── utils.py                # Padding, masking, interpolation helpers
├── extract_features.py         # One-time offline feature extraction
└── train.py                    # Main training entry point
```

## Architecture

The DAVID VAE compresses variable-length Qwen3-VL visual features (`[B, N, 2048]`) into `L=64` fixed latent tokens, trained with **Stochastic Prefix Truncation** to enforce coarse-to-fine semantic ordering.

- **Encoder**: `L` learned queries cross-attend to `N` visual tokens → `mu`, `logvar` each `[B, L, D]`
- **Truncation**: sample `m ~ Uniform(1, L)`, keep only `z[:, :m, :]`
- **Decoder**: `N_queries=256` learned queries cross-attend to `z_prefix` → reconstructed features `[B, 256, D]`
- **Loss**: `MSE(recon, target) + beta * KL` with beta linearly annealed from 0

At inference, tokens can be truncated at any prefix length for adaptive reasoning under variable token budgets.
