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

Verify the full pipeline using synthetic data (no backbone or cache needed):

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

### 4. Evaluate VQA sanity prompts (Qwen3-VL and DAVID+VAE)

Run caption-style prompting on PerceptionTest videos (defaults to `split=validation`):

```bash
# Baseline Qwen3-VL only
python evaluate_vqa.py \
    --method qwen \
    --split validation \
    --max_samples 100 \
    --prompt "What is the video caption?"

# Side-by-side comparison (Qwen3-VL + DAVID reconstruction)
python evaluate_vqa.py \
    --method both \
    --split validation \
    --max_samples 100 \
    --prompt "What is the video caption?" \
    --vae_checkpoint ./checkpoints/step_0050000.pt
```

The script writes JSONL predictions and a summary JSON under `./eval_outputs/`.
Note: the `chancharikm/QualityCheck` PerceptionTest subset currently exposes `train` and `validation` splits.

## Project Structure

```
DAVID/
├── pyproject.toml
├── configs/
│   └── train_config.yaml       # All hyperparameters (model, training, data, extraction, logging)
├── david/
│   ├── backbone.py             # Frozen Qwen3-VL vision encoder (LLM weights discarded)
│   ├── vae.py                  # DAVIDEncoder + DAVIDDecoder + DAVIDVAE (self-attention)
│   ├── dataset.py              # Dataset loader (cached and online modes)
│   ├── loss.py                 # Adaptive recon loss + KL + beta warm-up scheduler
│   └── utils.py                # Padding, masking, interpolation helpers
├── extract_features.py         # One-time offline feature extraction
└── train.py                    # Main training entry point
```

## Configuration

`configs/train_config.yaml` has five sections:

| Section | Purpose |
|---|---|
| `model` | Backbone name, `input_dim`, encoder/decoder layers, `progressive_ratio` |
| `training` | Batch size, learning rate, beta schedule, gradient clipping |
| `data` | Dataset name, feature cache path, `sample_fps`, `max_frames` |
| `extraction` | All `extract_features.py` defaults (`num_shards`, `cache_dir`, `device`, etc.) |
| `logging` | WandB project, checkpoint directory, log/save intervals |

## Architecture

**Backbone** (`backbone.py`): Loads only the Qwen3-VL-8B vision encoder onto GPU and discards the LLM weights (~16 GB VRAM saving). Videos are sampled at 1 fps and processed as a whole, producing a flat `[N, 4096]` token sequence where N varies with video duration.

**DAVID VAE** (`vae.py`): All three stages — input, latent, output — share the same `[B, N, D]` shape.

- **Encoder**: self-attention blocks over the full video token sequence (padding masked). Optional `progressive_ratio` causes later tokens to attend to progressively fewer random peers, strengthening the coarse-to-fine ordering incentive. Outputs `mu, logvar` at `[B, N, D]`.
- **Truncation**: sample `m ~ Uniform(1, N)` during training; zero-pad z beyond position m before decoding. This forces early tokens to carry coarse information and later tokens to carry refinements.
- **Decoder**: self-attention blocks over the zero-padded z; outputs reconstructed features `[B, N, D]`.

**Loss** (`loss.py`): `recon_loss + beta * KL`
- `recon_loss` is adaptive: exponent `p` and scale `s` both equal `2 * sigmoid(m/N)`, smoothly transitioning from L1 (small prefix → coarse reconstruction) to L2 (full prefix → precise reconstruction).
- `beta` is linearly annealed from 0 to `beta_target` over `beta_warmup_steps` to prevent posterior collapse.

At inference, tokens can be truncated at any prefix length for adaptive reasoning under variable token budgets.
