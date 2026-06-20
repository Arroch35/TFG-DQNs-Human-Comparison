"""
Cross-seed RSA analysis
=======================
For each game and each layer, computes the pairwise Spearman RSA
between all 5 DQN seeds, then reports mean ± std of the off-diagonal
correlations as a reliability/consistency metric.

Output per game:
  - {game}_{layer}_cross_seed_RSA.npy   : 5x5 RSA matrix
  - {game}_{layer}_cross_seed_RSA.png   : heatmap
  - cross_seed_summary.csv              : mean ± std per game/layer
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

# =========================================================
# CONFIG
# =========================================================
SEEDS      = ["seed_0", "seed_1", "seed_2", "seed_3", "seed_42"]
GAMES      = ["pong", "pacman", "spaceinvaders"]
LAYERS     = ["conv1", "conv2", "conv3", "fc"]
RDM_METHOD = "correlation"

RDMS_ROOT   = "../data/test_16_rdms/big_rdm_equal_size" #"../../data/test_16_rdms/buenos_25" #"../../data/DQN_rdms"
SAVE_FOLDER = "../data/multi_seed/cross_seed_rsa/big_rdm_equal_size"

os.makedirs(SAVE_FOLDER, exist_ok=True)

# =========================================================
# HELPERS
# =========================================================
def upper_tri_vector(rdm):
    idx = np.triu_indices_from(rdm, k=1)
    return rdm[idx]


def spearman_rsa(rdm_a, rdm_b):
    """Spearman correlation between upper triangles of two RDMs."""
    v_a = upper_tri_vector(rdm_a)
    v_b = upper_tri_vector(rdm_b)
    if np.std(v_a) == 0 or np.std(v_b) == 0:
        return np.nan
    # Spearman = Pearson on ranks
    rank_a = v_a.argsort().argsort().astype(float)
    rank_b = v_b.argsort().argsort().astype(float)
    return np.corrcoef(rank_a, rank_b)[0, 1]


def load_rdm(seed, game, layer):
    path = os.path.join(
        RDMS_ROOT, seed, game,
        f"{game}_{layer}_{RDM_METHOD}_RDM.npy"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing RDM: {path}")
    return np.load(path)


# =========================================================
# MAIN
# =========================================================
summary_rows = []

for game in GAMES:
    print(f"\n{'='*60}")
    print(f"GAME: {game}")
    print(f"{'='*60}")

    game_save_folder = os.path.join(SAVE_FOLDER, game)
    os.makedirs(game_save_folder, exist_ok=True)

    for layer in LAYERS:
        print(f"\n  Layer: {layer}")

        # ── Load all 5 RDMs for this game/layer ──────────────────
        rdms = {}
        missing = False
        for seed in SEEDS:
            try:
                rdms[seed] = load_rdm(seed, game, layer)
            except FileNotFoundError as e:
                print(f"  WARNING: {e}")
                missing = True

        if missing or len(rdms) < 2:
            print(f"  Skipping {game}/{layer} — insufficient RDMs.")
            continue

        seed_list = list(rdms.keys())
        n         = len(seed_list)

        # ── Pairwise RSA matrix (n x n) ──────────────────────────
        rsa_matrix = np.eye(n)   # diagonal = 1 by definition

        for i, seed_i in enumerate(seed_list):
            for j, seed_j in enumerate(seed_list):
                if i == j:
                    continue
                rho = spearman_rsa(rdms[seed_i], rdms[seed_j])
                rsa_matrix[i, j] = rho

        # ── Mean ± std of off-diagonal entries ───────────────────
        off_diag = rsa_matrix[np.triu_indices(n, k=1)]   # upper tri only
        mean_rsa = np.nanmean(off_diag)
        std_rsa  = np.nanstd(off_diag)

        print(f"  Cross-seed RSA — mean: {mean_rsa:.3f}  std: {std_rsa:.3f}")

        summary_rows.append({
            "game":     game,
            "layer":    layer,
            "mean_rsa": mean_rsa,
            "std_rsa":  std_rsa,
        })

        # ── Save RSA matrix ──────────────────────────────────────
        npy_path = os.path.join(
            game_save_folder,
            f"{game}_{layer}_cross_seed_RSA.npy"
        )
        np.save(npy_path, rsa_matrix)

        # ── Plot heatmap ─────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(5, 4))

        im = ax.imshow(rsa_matrix, cmap="viridis", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="RSA (Spearman ρ)")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(seed_list, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(seed_list, fontsize=8)

        for i in range(n):
            for j in range(n):
                val = rsa_matrix[i, j]
                ax.text(j, i, f"{val:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if val > 0.5 else "black")

        ax.set_title(
            f"{game} | {layer}\n"
            f"Cross-seed RSA  (mean={mean_rsa:.3f} ± {std_rsa:.3f})",
            fontsize=9
        )
        plt.tight_layout()

        png_path = npy_path.replace(".npy", ".png")
        plt.savefig(png_path, dpi=200)
        plt.close()

        print(f"  Saved → {png_path}")

# =========================================================
# SUMMARY TABLE
# =========================================================
df = pd.DataFrame(summary_rows).round(4)

# Pivot for a clean layer × game view
pivot_mean = df.pivot(index="layer", columns="game", values="mean_rsa")
pivot_std  = df.pivot(index="layer", columns="game", values="std_rsa")

print(f"\n{'='*60}")
print("CROSS-SEED RSA SUMMARY (mean ± std)")
print(f"{'='*60}")
for layer in LAYERS:
    if layer not in pivot_mean.index:
        continue
    row = "  ".join(
        f"{game}: {pivot_mean.loc[layer, game]:.3f} ± {pivot_std.loc[layer, game]:.3f}"
        for game in GAMES if game in pivot_mean.columns
    )
    print(f"  {layer:6s}  {row}")

# Save CSV
csv_path = os.path.join(SAVE_FOLDER, "cross_seed_summary.csv")
df.to_csv(csv_path, index=False)
print(f"\nSummary saved → {csv_path}")

print("\nCross-seed RSA analysis complete.")
