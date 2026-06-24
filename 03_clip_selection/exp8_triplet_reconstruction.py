import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import spearmanr
from tqdm import tqdm
import cy_tste

from src.config import GAMES, REFERENCE_SEED, REPR, TSTE, get_path, ensure
from src.utils import embedding_to_rdm, build_triplets_from_rdm

# =========================================================
# CONFIG
# =========================================================
SEED   = REFERENCE_SEED             # "seed_42"
METHOD = REPR["rdm_method"]         # "correlation"

LAYERS = ["fc"]                     # extend as needed: ["conv1", "conv2", "conv3", "fc"]

DIM      = TSTE["dim"]              # 2
N_REPEAT = TSTE["n_repeats"]        # 100
MAX_ITER = TSTE["max_iter"]         # 1000

SUBSET_PERCENTS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

# Paths
SAVE_FOLDER = ensure("rdms_selected15", seed=SEED, game="triplet_experiment_results")
# NOTE: "rdms_selected15" resolves to data/test_16_rdms/selected_subset_15/{seed}/{game}.
# The original output path was data/triplet_experiment_results/selected_subset_15/{seed}.
# If you want a dedicated key, consider adding to config.py:
#   "triplet_exp_results": DATA / "triplet_experiment_results" / "selected_subset_15" / "{seed}",
# Until then, SAVE_FOLDER is derived manually:
from src.config import DATA
SAVE_FOLDER = DATA / "triplet_experiment_results" / "selected_subset_15" / SEED
SAVE_FOLDER.mkdir(parents=True, exist_ok=True)

# =========================================================
# HELPERS
# =========================================================
def compare_rdms(rdm1, rdm2):
    idx = np.triu_indices_from(rdm1, k=1)
    return spearmanr(rdm1[idx], rdm2[idx]).correlation


# =========================================================
# MAIN LOOP
# =========================================================
all_results = []

for game in GAMES:
    for layer in LAYERS:
        print("\n" + "=" * 60)
        print(f"GAME: {game} | LAYER: {layer}")
        print("=" * 60)

        rdm_path = get_path("rdms_selected15", seed=SEED, game=game) / f"{game}_{layer}_{METHOD}_RDM.npy"

        if not rdm_path.exists():
            print(f"Missing file: {rdm_path}")
            continue

        rdm = np.load(rdm_path)
        N   = rdm.shape[0]
        print(f"Clips: {N}")

        all_triplets = build_triplets_from_rdm(rdm)
        print(f"Total triplets: {len(all_triplets)}")

        for p in SUBSET_PERCENTS:
            n_triplets = int(len(all_triplets) * p)
            print(f"\n--- {int(p*100)}% ({n_triplets} triplets) ---")

            for repeat in tqdm(range(N_REPEAT)):
                idxs   = np.random.choice(len(all_triplets), size=n_triplets, replace=False)
                subset = all_triplets[idxs]

                X = cy_tste.tste(
                    subset, no_dims=DIM, max_iter=MAX_ITER,
                    verbose=False, use_log=True,
                )

                rdm_rec = embedding_to_rdm(X)
                score   = compare_rdms(rdm, rdm_rec)

                all_results.append({
                    "game": game, "layer": layer,
                    "percent": p, "n_triplets": n_triplets,
                    "repeat": repeat, "score": score,
                })

# =========================================================
# SAVE RESULTS
# =========================================================
df      = pd.DataFrame(all_results)
raw_csv = SAVE_FOLDER / "triplet_results_raw.csv"
df.to_csv(raw_csv, float_format="%.4f", index=False)
print(f"\nSaved raw results → {raw_csv}")

summary = (
    df.groupby(["game", "layer", "percent"])
    .agg(
        mean_score=("score", "mean"),
        std_score= ("score", "std"),
        min_score= ("score", "min"),
        max_score= ("score", "max"),
    )
    .reset_index()
)
summary_csv = SAVE_FOLDER / "triplet_results_summary.csv"
summary.to_csv(summary_csv, float_format="%.4f", index=False)
print(f"Saved summary → {summary_csv}")

# =========================================================
# PLOTS
# =========================================================
PLOT_FOLDER = SAVE_FOLDER / "plots"
PLOT_FOLDER.mkdir(parents=True, exist_ok=True)

summary_df = pd.read_csv(summary_csv)

for layer in LAYERS:
    plt.figure(figsize=(8, 5))

    for game in GAMES:
        subset = summary_df[
            (summary_df["layer"] == layer) &
            (summary_df["game"]  == game)
        ].sort_values("percent")

        if len(subset) == 0:
            continue

        x    = subset["percent"].values * 100
        y    = subset["mean_score"].values
        yerr = subset["std_score"].values

        plt.plot(x, y, marker="o", label=game)
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.2)

    plt.xlabel("Percentage of triplets used")
    plt.ylabel("Spearman correlation")
    plt.title(f"t-STE reconstruction quality - {layer}")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()

    save_path = PLOT_FOLDER / f"{layer}_reconstruction_plot.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot → {save_path}")
