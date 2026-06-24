"""
exp11_cross_seed_rsa.py
Cross-seed RSA analysis
=======================
For each game and each layer, computes the pairwise Spearman RSA
between all 5 DQN seeds, then reports mean ± std of the off-diagonal
correlations as a reliability/consistency metric.

Output per game:
  - {game}_{layer}_cross_seed_RSA.npy  : 5×5 RSA matrix
  - {game}_{layer}_cross_seed_RSA.png  : heatmap
  - cross_seed_summary.csv             : mean ± std per game/layer
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import SEEDS, GAMES, REPR, get_path, ensure
from src.utils import upper_tri

# =========================================================
# CONFIG
# =========================================================
LAYERS     = ["conv1", "conv2", "conv3", "fc"]
RDM_METHOD = REPR["rdm_method"]   # "correlation"

SAVE_FOLDER = ensure("results_cross_seed_rsa") 

# =========================================================
# HELPERS
# =========================================================
def spearman_rsa(rdm_a, rdm_b):
    v_a, v_b = upper_tri(rdm_a), upper_tri(rdm_b)
    if np.std(v_a) == 0 or np.std(v_b) == 0:
        return np.nan
    rank_a = v_a.argsort().argsort().astype(float)
    rank_b = v_b.argsort().argsort().astype(float)
    return np.corrcoef(rank_a, rank_b)[0, 1]


def load_rdm(seed, game, layer):
    path = get_path("rdms_bigset", seed=seed, game=game) / f"{game}_{layer}_{RDM_METHOD}_RDM.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing RDM: {path}")
    return np.load(path)


# =========================================================
# MAIN
# =========================================================
summary_rows = []

for game in GAMES:
    print(f"\n{'='*60}\nGAME: {game}\n{'='*60}")

    game_save_folder = SAVE_FOLDER / game
    game_save_folder.mkdir(parents=True, exist_ok=True)

    for layer in LAYERS:
        print(f"\n  Layer: {layer}")

        rdms, missing = {}, False
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

        rsa_matrix = np.eye(n)
        for i, si in enumerate(seed_list):
            for j, sj in enumerate(seed_list):
                if i != j:
                    rsa_matrix[i, j] = spearman_rsa(rdms[si], rdms[sj])

        off_diag = rsa_matrix[np.triu_indices(n, k=1)]
        mean_rsa = np.nanmean(off_diag)
        std_rsa  = np.nanstd(off_diag)
        print(f"  Cross-seed RSA — mean: {mean_rsa:.3f}  std: {std_rsa:.3f}")

        summary_rows.append({"game": game, "layer": layer, "mean_rsa": mean_rsa, "std_rsa": std_rsa})

        # ── Save matrix ───────────────────────────────────
        npy_path = game_save_folder / f"{game}_{layer}_cross_seed_RSA.npy"
        np.save(npy_path, rsa_matrix)

        # ── Heatmap ───────────────────────────────────────
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(rsa_matrix, cmap="viridis", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="RSA (Spearman ρ)")
        ax.set_xticks(range(n)); ax.set_xticklabels(seed_list, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(seed_list, fontsize=8)
        for i in range(n):
            for j in range(n):
                val = rsa_matrix[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if val > 0.5 else "black")
        ax.set_title(f"{game} | {layer}\nCross-seed RSA  (mean={mean_rsa:.3f} ± {std_rsa:.3f})", fontsize=9)
        plt.tight_layout()
        png_path = npy_path.with_suffix(".png")
        plt.savefig(png_path, dpi=200)
        plt.close()
        print(f"  Saved → {png_path}")

# =========================================================
# SUMMARY TABLE
# =========================================================
df = pd.DataFrame(summary_rows).round(4)

pivot_mean = df.pivot(index="layer", columns="game", values="mean_rsa")
pivot_std  = df.pivot(index="layer", columns="game", values="std_rsa")

print(f"\n{'='*60}\nCROSS-SEED RSA SUMMARY (mean ± std)\n{'='*60}")
for layer in LAYERS:
    if layer not in pivot_mean.index:
        continue
    row = "  ".join(
        f"{g}: {pivot_mean.loc[layer, g]:.3f} ± {pivot_std.loc[layer, g]:.3f}"
        for g in GAMES if g in pivot_mean.columns
    )
    print(f"  {layer:6s}  {row}")

csv_path = SAVE_FOLDER / "cross_seed_summary.csv"
df.to_csv(csv_path, index=False)
print(f"\nSummary saved → {csv_path}")
print("\nCross-seed RSA analysis complete.")
