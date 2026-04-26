"""DAVID model architecture block diagram for the poster."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── Palette ──────────────────────────────────────────────────────────────────
C_FROZEN  = "#2C5F8A"   # dark blue  – frozen components
C_TRAINABLE = "#C0392B" # deep red   – trainable DAVID components
C_TRUNC   = "#8E44AD"   # purple     – prefix truncation
C_VQA     = "#1A7A4A"   # green      – output / evaluation
C_ARROW   = "#444444"
C_FEAT    = "#555555"
C_ENTROPY = "#E67E22"   # orange     – entropy loss annotation
C_BG      = "#FAFBFD"

FROZEN_ALPHA    = 0.18
TRAINABLE_ALPHA = 0.18

fig, ax = plt.subplots(figsize=(18, 7), facecolor="white")
ax.set_xlim(0, 18)
ax.set_ylim(0, 7)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Helper functions ──────────────────────────────────────────────────────────

def frozen_badge(ax, x, y, colour):
    """Small 'FROZEN' badge at (x, y)."""
    ax.text(x, y, "FROZEN", ha="center", va="center",
            fontsize=6.5, fontweight="bold", color="white", zorder=5,
            bbox=dict(facecolor=colour, edgecolor="none",
                      boxstyle="round,pad=0.25", alpha=0.85))

def rbox(ax, x, y, w, h, colour, label_lines, sublabel=None,
         frozen=False, alpha=0.18, fontsize=10.5, lw=2.2, style="round,pad=0.1"):
    """Draw a rounded box with multi-line label."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=style,
                         linewidth=lw, edgecolor=colour,
                         facecolor=colour, alpha=alpha,
                         zorder=2)
    ax.add_patch(box)
    cx, cy = x + w / 2, y + h / 2
    full = "\n".join(label_lines)
    ax.text(cx, cy, full, ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color=colour, zorder=3,
            multialignment="center", linespacing=1.4)
    if sublabel:
        ax.text(cx, y + 0.22, sublabel, ha="center", va="bottom",
                fontsize=8, color=colour, style="italic", zorder=3, alpha=0.85)
    if frozen:
        frozen_badge(ax, x + w - 0.42, y + h - 0.28, colour)

def arrow(ax, x0, y0, x1, y1, label=None, label_y_off=0.22, colour=C_ARROW, lw=2.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>",
                                color=colour, lw=lw,
                                mutation_scale=18),
                zorder=3)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2 + label_y_off
        ax.text(mx, my, label, ha="center", va="bottom",
                fontsize=8, color=C_FEAT, style="italic",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
                zorder=4)

def feat_label(ax, x, y, text, colour=C_FEAT):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=7.8, color=colour, style="italic",
            bbox=dict(facecolor="white", edgecolor=colour, linewidth=0.8,
                      boxstyle="round,pad=0.25", alpha=0.85),
            zorder=5)

# ── Layout constants ──────────────────────────────────────────────────────────
MID_Y  = 3.1   # vertical center of all main blocks
BOX_H  = 1.9   # standard block height
NARROW = 1.65  # narrow box width
WIDE   = 2.0   # wider box width

# ── 1. Video frames ───────────────────────────────────────────────────────────
for off in [0.20, 0.10, 0.0]:
    ax.add_patch(FancyBboxPatch(
        (0.3 + off, MID_Y - 0.5 + off), 1.15, 1.5,
        boxstyle="round,pad=0.08", linewidth=1.4,
        edgecolor="#888888", facecolor="#D6E8F5", alpha=0.6, zorder=2 - int(off*10)
    ))
ax.text(0.3 + 0.575, MID_Y - 0.5 + 0.75, "▶", ha="center", va="center",
        fontsize=26, color="#4A90C4", zorder=5)
ax.text(0.3 + 0.575, MID_Y - 0.5 - 0.28, "Video\nFrames",
        ha="center", va="top", fontsize=9, color="#444444",
        fontweight="bold", multialignment="center")

# Arrow: video → Qwen Vision
arrow(ax, 1.72, MID_Y + 0.25, 2.25, MID_Y + 0.25)

# ── 2. Qwen Vision Encoder (frozen) ──────────────────────────────────────────
rbox(ax, 2.25, MID_Y - BOX_H/2, WIDE, BOX_H, C_FROZEN,
     ["Qwen3-VL", "Vision", "Encoder"],
     frozen=True, fontsize=10, alpha=0.16)
ax.text(2.25 + WIDE/2, MID_Y - BOX_H/2 - 0.30, "frozen",
        ha="center", va="top", fontsize=8.5, color=C_FROZEN,
        style="italic")

# Arrow: Qwen Vision → DAVID Encoder  (feature label between)
arrow(ax, 2.25 + WIDE, MID_Y + 0.25, 5.00, MID_Y + 0.25,
      label="[N × 4096]")

# ── 3. DAVID Encoder (trainable) ──────────────────────────────────────────────
rbox(ax, 5.00, MID_Y - BOX_H/2, WIDE, BOX_H, C_TRAINABLE,
     ["DAVID", "Encoder"],
     fontsize=10.5, alpha=0.16)
ax.text(5.00 + WIDE/2, MID_Y - BOX_H/2 - 0.30, "trainable",
        ha="center", va="top", fontsize=8.5, color=C_TRAINABLE, style="italic")

# KL annotation (below encoder)
ax.text(5.00 + WIDE/2, MID_Y - BOX_H/2 - 0.68,
        "μ, σ  →  z ~ N(μ, σ²)\n+ KL loss",
        ha="center", va="top", fontsize=7.8, color=C_TRAINABLE,
        multialignment="center",
        bbox=dict(facecolor="#FDF0EE", edgecolor=C_TRAINABLE,
                  linewidth=0.9, boxstyle="round,pad=0.3", alpha=0.9))

# ── 4. Prefix Truncation ──────────────────────────────────────────────────────
arrow(ax, 5.00 + WIDE, MID_Y + 0.25, 7.55, MID_Y + 0.25,
      label="z  [N × 4096]")

# Draw token squares
TRUNC_X = 7.55
TOKEN_W = 0.22
TOKEN_H = 0.38
TOKEN_GAP = 0.05
N_TOK = 10
M_TOK = 5   # kept tokens

ax.text(TRUNC_X + (N_TOK * (TOKEN_W + TOKEN_GAP))/2, MID_Y + 0.72,
        "Prefix Truncation  (m ~ Uniform(1, N))",
        ha="center", va="bottom", fontsize=9.5, fontweight="bold",
        color=C_TRUNC)

for i in range(N_TOK):
    tx = TRUNC_X + i * (TOKEN_W + TOKEN_GAP)
    ty = MID_Y + 0.18
    kept = i < M_TOK
    fc = C_TRUNC if kept else "#CCCCCC"
    ec = C_TRUNC if kept else "#AAAAAA"
    alpha = 0.80 if kept else 0.30
    ax.add_patch(FancyBboxPatch((tx, ty), TOKEN_W, TOKEN_H,
                                boxstyle="round,pad=0.02",
                                facecolor=fc, edgecolor=ec,
                                linewidth=1.2, alpha=alpha, zorder=3))
    if not kept:
        ax.text(tx + TOKEN_W/2, ty + TOKEN_H/2, "0",
                ha="center", va="center", fontsize=7, color="#999999", zorder=4)

# Brace / label
ax.annotate("", xy=(TRUNC_X, MID_Y + 0.10),
            xytext=(TRUNC_X + M_TOK * (TOKEN_W + TOKEN_GAP) - TOKEN_GAP, MID_Y + 0.10),
            arrowprops=dict(arrowstyle="<->", color=C_TRUNC, lw=1.3))
ax.text(TRUNC_X + M_TOK * (TOKEN_W + TOKEN_GAP) / 2, MID_Y + 0.0,
        "m tokens", ha="center", va="top", fontsize=7.5, color=C_TRUNC)

ax.annotate("", xy=(TRUNC_X + M_TOK * (TOKEN_W + TOKEN_GAP), MID_Y + 0.10),
            xytext=(TRUNC_X + N_TOK * (TOKEN_W + TOKEN_GAP) - TOKEN_GAP, MID_Y + 0.10),
            arrowprops=dict(arrowstyle="<->", color="#999999", lw=1.3))
ax.text(TRUNC_X + (M_TOK + N_TOK) / 2 * (TOKEN_W + TOKEN_GAP) - 0.05, MID_Y + 0.0,
        "zero-pad", ha="center", va="top", fontsize=7.5, color="#888888")

# ── 5. Arrow: Truncation → DAVID Decoder ─────────────────────────────────────
TRUNC_END_X = TRUNC_X + N_TOK * (TOKEN_W + TOKEN_GAP) + 0.05
arrow(ax, TRUNC_END_X, MID_Y + 0.25, TRUNC_END_X + 0.60, MID_Y + 0.25,
      label="z_padded")

# ── 6. DAVID Decoder (trainable) ──────────────────────────────────────────────
DEC_X = TRUNC_END_X + 0.60
rbox(ax, DEC_X, MID_Y - BOX_H/2, WIDE, BOX_H, C_TRAINABLE,
     ["DAVID", "Decoder"],
     fontsize=10.5, alpha=0.16)
ax.text(DEC_X + WIDE/2, MID_Y - BOX_H/2 - 0.30, "trainable",
        ha="center", va="top", fontsize=8.5, color=C_TRAINABLE, style="italic")

# Entropy loss callout (below decoder)
ent_cx = DEC_X + WIDE / 2
ent_by = MID_Y - BOX_H/2 - 1.02
ax.text(ent_cx, ent_by,
        r"$\mathcal{L}_{\mathrm{ent}} = \lambda\!\sum_j \frac{j}{N} H(p_{\cdot j})$"
        "\n(first K decoder layers only)",
        ha="center", va="top", fontsize=8.2, color=C_ENTROPY,
        multialignment="center",
        bbox=dict(facecolor="#FEF5E7", edgecolor=C_ENTROPY,
                  linewidth=1.0, boxstyle="round,pad=0.35", alpha=0.95))
# Dashed connector from callout to decoder
ax.annotate("", xy=(ent_cx, MID_Y - BOX_H/2),
            xytext=(ent_cx, ent_by),
            arrowprops=dict(arrowstyle="-|>", color=C_ENTROPY,
                            lw=1.3, linestyle="dashed",
                            connectionstyle="arc3,rad=0.0"))

# ── 7. Arrow: Decoder → Qwen3 LM ─────────────────────────────────────────────
arrow(ax, DEC_X + WIDE, MID_Y + 0.25, DEC_X + WIDE + 0.80, MID_Y + 0.25,
      label="[N × 4096]")

# ── 8. Qwen3 LM Decoder (frozen) ─────────────────────────────────────────────
LM_X = DEC_X + WIDE + 0.80
rbox(ax, LM_X, MID_Y - BOX_H/2, WIDE, BOX_H, C_FROZEN,
     ["Qwen3-VL", "Language", "Model"],
     frozen=True, fontsize=10, alpha=0.16)
ax.text(LM_X + WIDE/2, MID_Y - BOX_H/2 - 0.30, "frozen · eval only",
        ha="center", va="top", fontsize=8.5, color=C_FROZEN, style="italic")

# ── 9. Arrow: LM → VQA answer ────────────────────────────────────────────────
arrow(ax, LM_X + WIDE, MID_Y + 0.25, LM_X + WIDE + 0.65, MID_Y + 0.25,
      colour=C_VQA)

# VQA answer box
VQA_X = LM_X + WIDE + 0.65
rbox(ax, VQA_X, MID_Y - 0.55, 1.30, 1.10, C_VQA,
     ["VQA", "Answer"],
     fontsize=10, alpha=0.18, lw=2.0)

# ── Legend: frozen vs trainable ───────────────────────────────────────────────
legend_x, legend_y = 0.30, 6.40
ax.add_patch(FancyBboxPatch((legend_x, legend_y - 0.30), 5.6, 0.85,
                            boxstyle="round,pad=0.1",
                            facecolor="#F0F4F8", edgecolor="#CCCCCC",
                            linewidth=0.8, alpha=0.9, zorder=1))
for i, (col, lbl) in enumerate([(C_FROZEN, "Frozen (not trained)"),
                                  (C_TRAINABLE, "Trainable (DAVID VAE)"),
                                  (C_ENTROPY, "Entropy reg. loss term")]):
    lx = legend_x + 0.20 + i * 1.90
    ax.add_patch(mpatches.Rectangle((lx, legend_y + 0.14), 0.32, 0.24,
                                    facecolor=col, alpha=0.6,
                                    edgecolor=col, linewidth=1.2, zorder=3))
    ax.text(lx + 0.42, legend_y + 0.26, lbl,
            va="center", fontsize=8.2, color="#333333", zorder=3)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(9.0, 6.72, "DAVID — Model Architecture",
        ha="center", va="center", fontsize=15, fontweight="bold",
        color="#1A1A2E")

ax.text(9.0, 6.30,
        "Dynamic Adaptive Video Inference and Decoding  ·  "
        "Variable-budget visual feature compression via a prefix-ordered VAE",
        ha="center", va="center", fontsize=8.8, color="#555555", style="italic")

# ── Save ──────────────────────────────────────────────────────────────────────
out_pdf = "/Users/ethan/ethanfolder/cmu/mmml/project/DAVID/scripts/architecture_diagram.pdf"
out_png = out_pdf.replace(".pdf", ".png")
fig.savefig(out_pdf, bbox_inches="tight", dpi=220)
fig.savefig(out_png, bbox_inches="tight", dpi=220)
print(f"Saved:\n  {out_pdf}\n  {out_png}")
