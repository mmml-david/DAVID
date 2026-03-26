# DAVID — Dynamic Adaptive Video Inference and Decoding

A hierarchical video representation that encodes a video as a prefix-ordered sequence of semantic tokens, built on top of a frozen Qwen3-VL vision backbone.

## Setup

```bash
uv sync
source .venv/bin/activate
```

## Workflow

### 1. Extract features (one-time)

Run all videos through the frozen Qwen3-VL backbone and cache features as `.pt` files.
All options are read from `configs/train_config.yaml` (`extraction:` section); CLI flags override.

```bash
# Single machine — all settings from config
python extract_features.py --config configs/train_config.yaml

# Override specific settings
python extract_features.py --config configs/train_config.yaml \
    --device cuda:1 --max_samples 100

# Distributed — run on each machine with a unique --shard_id
# (num_shards is set in the config; shard_id must be provided explicitly)
python extract_features.py --config configs/train_config.yaml --shard_id 0
python extract_features.py --config configs/train_config.yaml --shard_id 1
# ...
```

The script deduplicates by video filename stem (derived from the `video` column path).
Output files are named `{video_name}.pt`. Re-runs skip already-extracted files, so it
is safe to resume or re-run on any shard.

**Disk usage:** ~35 GB for the full PerceptionTest set (float16, ~11.6k videos).

**HuggingFace cache setup:** The dataset stores videos as individual files in the hub
snapshot. If the snapshot symlink tree is incomplete (e.g. interrupted download),
the extraction script will automatically recreate missing symlinks from already-present
local blobs with no network I/O. Videos whose blobs are absent will be skipped with a
warning; re-run after completing the download to fill gaps.

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

Train in online mode (extracts features on-the-fly — slower, no cache needed):

```bash
python train.py --config configs/train_config.yaml --online
```

## Project Structure

```
DAVID/
├── pyproject.toml
├── configs/
│   └── train_config.yaml       # All hyperparameters (model, training, data, extraction, logging)
├── david/
│   ├── backbone.py             # Frozen Qwen3-VL feature extractor
│   ├── vae.py                  # DAVIDEncoder + DAVIDDecoder + DAVIDVAE
│   ├── dataset.py              # Dataset loader (cached and online modes)
│   ├── loss.py                 # MSE + KL loss + beta warm-up scheduler
│   └── utils.py                # Padding, masking, interpolation helpers
├── extract_features.py         # One-time offline feature extraction
└── train.py                    # Main training entry point
```

## Configuration

`configs/train_config.yaml` has five sections:

| Section | Purpose |
|---|---|
| `model` | Backbone name, VAE dimensions, number of latent tokens `L`, decoder output tokens `N_queries` |
| `training` | Batch size, learning rate, beta schedule, gradient clipping |
| `data` | Dataset name, feature cache path, `sample_fps`, `max_frames` |
| `extraction` | All `extract_features.py` defaults (`num_shards`, `cache_dir`, `device`, etc.) |
| `logging` | WandB project, checkpoint directory, log/save intervals |

## Architecture

The DAVID VAE compresses variable-length Qwen3-VL visual features into `L=64` fixed latent tokens, trained with **Stochastic Prefix Truncation** to enforce coarse-to-fine semantic ordering.

- **Backbone**: Frozen Qwen3-VL-8B vision encoder; outputs `[N, 4096]` post-PatchMerger tokens for a whole video (all frames concatenated). `N` varies by video duration.
- **Encoder**: `L` learned queries cross-attend to the `N` video tokens → `mu`, `logvar` each `[B, L, D]`
- **Truncation**: sample `m ~ Uniform(1, L)` during training, keep only `z[:, :m, :]`
- **Decoder**: `N_queries=256` learned queries cross-attend to `z_prefix` → reconstructed features `[B, 256, D]`
- **Loss**: `MSE(recon, target) + beta * KL` with beta linearly annealed from 0

Videos are sampled at 1 fps (configurable) so token count reflects actual duration. Batches pad shorter sequences; the encoder ignores padding via attention masking.

At inference, tokens can be truncated at any prefix length for adaptive reasoning under variable token budgets.
