"""Visualization of attention column-entropy regularization using real model checkpoints.

Compares:
  (a) step_0000500  — early checkpoint, lambda_entropy=0 (no reg applied yet)
  (b) best.pt       — step 5000, lambda_entropy at full target

Attention weights are captured from the first decoder block (compute_attn_entropy=True)
via a forward hook. Features are loaded from the cached perception_test dataset.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CKPT_BEFORE = "/data/user_data/hsuanhal/11777/DAVID/checkpoints/ethan-2b-online-attn-entropy/step_0000500.pt"
CKPT_AFTER  = "/data/user_data/hsuanhal/11777/DAVID/checkpoints/ethan-2b-online-attn-entropy/best.pt"
CACHE_DIR   = "/data/user_data/hsuanhal/11777/DAVID/features_cache/perception_test/train"
N_SAMPLES   = 4   # average attention over this many videos
N_VIZ       = 64  # truncate sequences to this length for visualization

C_EARLY  = "#4C9BE8"
C_LATE   = "#E8734C"
C_MID    = "#9B59B6"
C_WEIGHT = "#2ECC71"
C_BG     = "#F8FAFD"
C_GRID   = "#DDE3ED"

# ── Model config (must match training config) ─────────────────────────────────
from david.vae import DAVIDVAE, DAVIDConfig

MODEL_CFG = DAVIDConfig(
    input_dim=2048,
    n_encoder_layers=4,
    n_decoder_layers=4,
    n_heads=16,
    dropout=0.1,
    entropy_decoder_layers=2,
    entropy_layer_decay=0.5,
    grad_checkpoint=False,  # disable for inference
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(ckpt_path: str) -> DAVIDVAE:
    model = DAVIDVAE(MODEL_CFG).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    ema = ckpt.get("ema_state_dict", {})
    state = ema.get("shadow") or ckpt["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def get_attn_weights(model: DAVIDVAE, features: torch.Tensor, mask: torch.Tensor):
    """Run features through the VAE decoder and return attention weights [H, N, N]
    from the first decoder block (the one with compute_attn_entropy=True)."""
    captured = {}

    def hook(module, inputs, output):
        # output is (x, entropy); we want to re-run attention to capture weights
        # Instead, hook into the block's forward to intercept attn_weights
        pass

    # Patch the first entropy block to capture attn_weights during forward
    first_block = model.decoder.blocks[0]
    attn_store = {}

    original_forward = first_block.forward

    def patched_forward(x, key_padding_mask=None):
        B, N, D = x.shape
        h = first_block.norm1(x)
        q, k, v = first_block.qkv(h).chunk(3, dim=-1)
        H, d = first_block.n_heads, first_block.head_dim
        q = q.view(B, N, H, d).transpose(1, 2)
        k = k.view(B, N, H, d).transpose(1, 2)
        v = v.view(B, N, H, d).transpose(1, 2)
        scale = d ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1)  # [B, H, N, N]
        attn_store["weights"] = weights.detach().float().cpu()
        return original_forward(x, key_padding_mask)

    first_block.forward = patched_forward

    with torch.no_grad():
        # Encode → z → pad → decode (replicates VAE forward without loss)
        feat = features.to(DEVICE)
        msk  = mask.to(DEVICE)
        z, mu, logvar, _ = model.encoder(feat, msk)
        # use full prefix (no truncation) so we see the full attention pattern
        m = feat.shape[1]
        z_padded = z  # no truncation
        model.decoder(z_padded)

    first_block.forward = original_forward
    return attn_store.get("weights")  # [B, H, N, N]


def load_samples(n: int):
    """Extract real features from PerceptionTest videos using the 2B backbone."""
    from david.backbone import Qwen3VLBackbone
    from david.dataset import PerceptionTestVideoDataset

    print("  Loading Qwen3-VL-2B backbone for feature extraction...")
    backbone = Qwen3VLBackbone("Qwen/Qwen3-VL-2B-Instruct", dtype=torch.bfloat16, device=DEVICE)

    ds = PerceptionTestVideoDataset(
        feature_cache_dir=None,
        split="train",
        mode="online",
        hf_dataset_name="chancharikm/QualityCheck",
        subset="PerceptionTest",
        backbone=backbone,
        sample_fps=1.0,
        max_frames=64,
        shortest_edge=16384,
        longest_edge=204800,
    )

    feats, masks = [], []
    for i in range(n):
        item = ds[i]
        feat = item["features"].float()   # [L, 2048]
        L = feat.shape[0]
        N = N_VIZ
        if L >= N:
            feats.append(feat[:N])
        else:
            pad = torch.zeros(N - L, feat.shape[1])
            feats.append(torch.cat([feat, pad], dim=0))
        masks.append(torch.ones(N, dtype=torch.bool))
        print(f"  Processed video {i+1}/{n} (L={L})")

    del backbone
    import gc; gc.collect(); torch.cuda.empty_cache()

    return torch.stack(feats), torch.stack(masks), N_VIZ


def avg_attn(model, features, mask):
    """Average attention matrix over batch and heads → [N, N]."""
    weights = get_attn_weights(model, features, mask)  # [B, H, N, N]
    return weights.mean(dim=(0, 1)).numpy()  # [N, N]


# ── Load data & models ────────────────────────────────────────────────────────
print("Loading cached features...")
features, mask, N = load_samples(N_SAMPLES)
print(f"  {N_SAMPLES} videos, sequence length N={N}")

print("Loading checkpoints...")
model_before = load_model(CKPT_BEFORE)
model_after  = load_model(CKPT_AFTER)

print("Extracting attention weights...")
A_before = avg_attn(model_before, features, mask)
A_after  = avg_attn(model_after,  features, mask)


# ── Derived quantities ────────────────────────────────────────────────────────
def column_entropy(col):
    col = np.clip(col, 1e-10, None)
    col = col / col.sum()
    return -(col * np.log(col)).sum()

H_before = np.array([column_entropy(A_before[:, j]) for j in range(N)])
H_after  = np.array([column_entropy(A_after[:,  j]) for j in range(N)])
js = np.arange(N)
w  = js / N


# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 6.5), facecolor="white")

gs = gridspec.GridSpec(
    2, 4,
    figure=fig,
    left=0.05, right=0.97,
    top=0.78, bottom=0.11,
    wspace=0.50, hspace=0.60,
)

ax_before = fig.add_subplot(gs[:, 0])
ax_after  = fig.add_subplot(gs[:, 1])
ax_Hj     = fig.add_subplot(gs[0, 2])
ax_wHj    = fig.add_subplot(gs[1, 2])
ax_col    = fig.add_subplot(gs[:, 3])


def style_ax(ax):
    ax.set_facecolor(C_BG)
    ax.grid(True, color=C_GRID, linewidth=0.6, zorder=0)
    for sp in ax.spines.values():
        sp.set_color("#CCCCCC"); sp.set_linewidth(0.8)
    ax.tick_params(labelsize=8, colors="#444444", length=3)


# ── Attention matrices ────────────────────────────────────────────────────────
for ax, A, title, tag in [
    (ax_before, A_before, f"Without Entropy Reg  (step 500)", "(a)"),
    (ax_after,  A_after,  f"With Entropy Reg  (step 5000)",   "(b)"),
]:
    # Per-matrix scaling: clip top 2% so structure is visible even in diffuse matrices
    vmax = float(np.percentile(A, 98))
    ax.imshow(A, aspect="auto", cmap="Blues", vmin=0, vmax=vmax,
              interpolation="nearest")
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5, color="#222222")
    ax.set_xlabel("Key position  j  (DAVID token)", fontsize=8, labelpad=3)
    ax.set_ylabel("Query position  i", fontsize=8, labelpad=3)
    ticks = [0, N // 4, N // 2, 3 * N // 4, N - 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=7)
    ax.set_yticks([0, N - 1])
    ax.set_yticklabels(["0", str(N - 1)], fontsize=7)
    ax.text(0.04, 0.97, tag, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", color="#555555")

# Annotate early/late columns on "with reg" panel
for j, lbl, col in [(1, "early\n(coarse)", C_EARLY),
                    (N - 2, "late\n(fine)", C_LATE)]:
    ax_after.annotate(
        lbl, xy=(j, N - 0.5), xytext=(j, N + 1.8),
        xycoords="data", textcoords="data",
        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4),
        fontsize=7, ha="center", color=col, fontweight="bold",
        annotation_clip=False,
    )


# ── Panel (c): column entropy H_j ────────────────────────────────────────────
style_ax(ax_Hj)
bw = 0.38
ax_Hj.bar(js - bw/2, H_before, width=bw, color=C_EARLY, alpha=0.75,
          label="step 500 (no reg)", zorder=2)
ax_Hj.bar(js + bw/2, H_after,  width=bw, color=C_LATE,  alpha=0.80,
          label="step 5000 (reg)",  zorder=2)
ax_Hj.set_xlabel("Key position  j", fontsize=8)
ax_Hj.set_ylabel("H(p·ⱼ)  [nats]", fontsize=8)
ax_Hj.set_title("(c)  Column Entropy  H(p·ⱼ)", fontsize=9,
                fontweight="bold", color="#222222")
ax_Hj.legend(fontsize=7.5, framealpha=0.9, loc="upper right")
xticks = [0, N // 4, N // 2, 3 * N // 4, N - 1]
ax_Hj.set_xticks(xticks)
ax_Hj.set_xticklabels([str(t) for t in xticks], fontsize=7)


# ── Panel (d): weighted penalty w_j * H_j ────────────────────────────────────
style_ax(ax_wHj)
ax_wHj.bar(js - bw/2, w * H_before, width=bw, color=C_EARLY, alpha=0.75,
           label="step 500 (no reg)", zorder=2)
ax_wHj.bar(js + bw/2, w * H_after,  width=bw, color=C_LATE,  alpha=0.80,
           label="step 5000 (reg)",  zorder=2)
ax_wHj.plot(js, w * np.log(N), "--", color=C_WEIGHT, linewidth=1.5,
            label="max  (j/N)·ln N", alpha=0.85, zorder=3)
ax_wHj.set_xlabel("Key position  j", fontsize=8)
ax_wHj.set_ylabel("(j/N)·H(p·ⱼ)", fontsize=8)
ax_wHj.set_title("(d)  Weighted Penalty  (j/N)·H(p·ⱼ)", fontsize=9,
                fontweight="bold", color="#222222")
ax_wHj.legend(fontsize=7, framealpha=0.9, loc="upper left")
ax_wHj.set_xticks(xticks)
ax_wHj.set_xticklabels([str(t) for t in xticks], fontsize=7)


# ── Panel (e): column distribution examples ───────────────────────────────────
style_ax(ax_col)
qs = np.arange(N)
examples = [
    (1,          "early  (j=1,  coarse)", C_EARLY, "o"),
    (N // 2,     f"mid    (j={N//2})",    C_MID,   "s"),
    (N - 2,      f"late   (j={N-2},  fine)", C_LATE, "^"),
]
for j, lbl, colour, marker in examples:
    ax_col.plot(qs, A_after[:, j], f"-{marker}", color=colour, label=lbl,
                linewidth=1.8, markersize=5, alpha=0.9, zorder=3)
ax_col.set_xlabel("Query position  i", fontsize=8)
ax_col.set_ylabel("p_{i,j}  (column distribution)", fontsize=8)
ax_col.set_title("(e)  Column Distributions\n(step 5000, with reg)", fontsize=9,
                fontweight="bold", color="#222222")
ax_col.legend(fontsize=7.5, framealpha=0.9, loc="upper left")
ax_col.grid(True, color=C_GRID, linewidth=0.6, zorder=0)
ax_col.tick_params(labelsize=8, colors="#444444", length=3)
ax_col.set_xticks([0, N // 4, N // 2, 3 * N // 4, N - 1])
ax_col.set_xticklabels([str(t) for t in [0, N//4, N//2, 3*N//4, N-1]], fontsize=7)


# ── Title ─────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.985, "Attention Column-Entropy Regularization  (Real Checkpoint)",
         ha="center", va="top", fontsize=14, fontweight="bold", color="#1A1A2E")
fig.text(0.5, 0.935,
         r"$\mathcal{L}_{\mathrm{entropy}}\;=\;\lambda\!\sum_{j=0}^{N-1}\frac{j}{N}\cdot H(p_{\cdot j})$",
         ha="center", va="top", fontsize=13, color="#2C2C6C")
fig.text(0.5, 0.870,
         r"Decoder block 0 · averaged over 4 PerceptionTest videos and all attention heads",
         ha="center", va="top", fontsize=9, color="#555555", style="italic")


# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = Path(__file__).parent
out_pdf = out_dir / "entropy_viz_real.pdf"
out_png = out_dir / "entropy_viz_real.png"
fig.savefig(out_pdf, bbox_inches="tight", dpi=200)
fig.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"Saved:\n  {out_pdf}\n  {out_png}")
plt.close()
