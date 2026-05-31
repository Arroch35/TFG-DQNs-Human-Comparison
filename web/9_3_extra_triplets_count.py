import numpy as np
import os
import pandas as pd
from itertools import combinations
from scipy.stats import spearmanr
from tqdm import tqdm
import cy_tste

# =========================================================
# CONFIG
# =========================================================
GAMES = ["pacman", "pong", "spaceinvaders"]
LAYERS = [ "fc"] #"conv1", "conv2", "conv3",
METHOD = "correlation"
SEED = "seed_42"
BASE_RDM_FOLDER = f"../data/test_16_rdms/selected_subset_15/{SEED}"
SAVE_FOLDER = f"../data/triplet_experiment_results/selected_subset_15/{SEED}"
os.makedirs(SAVE_FOLDER, exist_ok=True)

DIM = 2
N_REPEATS = 100
MAX_ITER = 1000

SUBSET_PERCENTS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

# =========================================================
# BUILD TRIPLETS
# =========================================================
def build_triplets_from_rdm(rdm):
    triplets = []

    for i, j, k in combinations(range(len(rdm)), 3):
        dij = rdm[i, j]
        dik = rdm[i, k]
        djk = rdm[j, k]

        if dij <= dik and dij <= djk:
            triplets.append((i, j, k))
            triplets.append((j, i, k))
        elif dik <= dij and dik <= djk:
            triplets.append((i, k, j))
            triplets.append((k, i, j))
        else:
            triplets.append((j, k, i))
            triplets.append((k, j, i))

    return np.array(triplets, dtype=np.int32)

# =========================================================
# EMBEDDING → RDM
# =========================================================
def embedding_to_rdm(X):
    diff = X[:, None, :] - X[None, :, :]
    return np.linalg.norm(diff, axis=-1)

# =========================================================
# COMPARE RDMS
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

        print("\n" + "="*60)
        print(f"GAME: {game} | LAYER: {layer}")
        print("="*60)

        rdm_path = os.path.join(
            BASE_RDM_FOLDER,
            game,
            f"{game}_{layer}_{METHOD}_RDM.npy"
        )

        if not os.path.exists(rdm_path):
            print(f"Missing file: {rdm_path}")
            continue

        rdm = np.load(rdm_path)
        N = rdm.shape[0]

        print(f"Clips: {N}")

        # -------------------------------------------------
        # Build triplets ONCE per layer
        # -------------------------------------------------
        all_triplets = build_triplets_from_rdm(rdm)
        print(f"Total triplets: {len(all_triplets)}")

        # -------------------------------------------------
        # Subsampling experiment
        # -------------------------------------------------
        for p in SUBSET_PERCENTS:

            n_triplets = int(len(all_triplets) * p)
            print(f"\n--- {int(p*100)}% ({n_triplets} triplets) ---")

            for repeat in tqdm(range(N_REPEATS)):

                # Sample subset
                idxs = np.random.choice(len(all_triplets), size=n_triplets, replace=False)
                subset = all_triplets[idxs]

                # Fit t-STE
                X = cy_tste.tste(
                    subset,
                    no_dims=DIM,
                    max_iter=MAX_ITER,
                    verbose=False,
                    use_log=True
                )

                # Reconstruct RDM
                rdm_rec = embedding_to_rdm(X)

                # Compare
                score = compare_rdms(rdm, rdm_rec)

                # Save individual result
                all_results.append({
                    "game": game,
                    "layer": layer,
                    "percent": p,
                    "n_triplets": n_triplets,
                    "repeat": repeat,
                    "score": score
                })

# =========================================================
# SAVE RAW RESULTS
# =========================================================
df = pd.DataFrame(all_results)
raw_csv = os.path.join(SAVE_FOLDER, "triplet_results_raw.csv")
df.to_csv(raw_csv, float_format='%.4f', index=False)
print(f"\nSaved raw results → {raw_csv}")

# =========================================================
# SAVE AGGREGATED RESULTS
# =========================================================
summary = (
    df.groupby(["game", "layer", "percent"])
    .agg(
        mean_score=("score", "mean"),
        std_score=("score", "std"),
        min_score=("score", "min"),
        max_score=("score", "max"),
    )
    .reset_index()
)

summary_csv = os.path.join(SAVE_FOLDER, "triplet_results_summary.csv")
summary.to_csv(summary_csv, float_format='%.4f', index=False)

print(f"Saved summary → {summary_csv}")

# =========================================================
# PLOTS PER LAYER
# =========================================================
import matplotlib.pyplot as plt

PLOT_FOLDER = os.path.join(SAVE_FOLDER, "plots")
os.makedirs(PLOT_FOLDER, exist_ok=True)

# Reload summary CSV (optional but cleaner)
summary_df = pd.read_csv(summary_csv)

for layer in LAYERS:

    plt.figure(figsize=(8, 5))

    for game in GAMES:

        # ---------------------------------------------
        # Select data
        # ---------------------------------------------
        subset = summary_df[
            (summary_df["layer"] == layer) &
            (summary_df["game"] == game)
        ].sort_values("percent")

        if len(subset) == 0:
            continue

        x = subset["percent"].values * 100
        y = subset["mean_score"].values
        yerr = subset["std_score"].values

        # ---------------------------------------------
        # Plot mean curve
        # ---------------------------------------------
        plt.plot(
            x,
            y,
            marker="o",
            label=game
        )

        # ---------------------------------------------
        # Plot std band
        # ---------------------------------------------
        plt.fill_between(
            x,
            y - yerr,
            y + yerr,
            alpha=0.2
        )

    # -------------------------------------------------
    # Cosmetics
    # -------------------------------------------------
    plt.xlabel("Percentage of triplets used")
    plt.ylabel("Spearman correlation")
    plt.title(f"t-STE reconstruction quality - {layer}")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()

    # -------------------------------------------------
    # Save
    # -------------------------------------------------
    save_path = os.path.join(
        PLOT_FOLDER,
        f"{layer}_reconstruction_plot.png"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved plot → {save_path}")