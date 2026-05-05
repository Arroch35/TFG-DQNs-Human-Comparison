import numpy as np
import os
import pandas as pd
from itertools import combinations
from scipy.stats import spearmanr
from tqdm import tqdm
import cy_tste
import random
import re
import matplotlib.pyplot as plt

from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm

# =========================================================
# CONFIG
# =========================================================
GAMES = ["pacman", "pong", "spaceinvaders"]
LAYERS = ["conv1", "conv2", "conv3", "fc0"]
METHOD = "euclidean"

ACTIVATIONS_FOLDER = "../data/test_16_PRUEBAS/big_rdm_equal_size"
SAVE_FOLDER = "../data/triplet_experiment_results"
os.makedirs(SAVE_FOLDER, exist_ok=True)

DIM = 2
N_REPEATS = 100
MAX_ITER = 1000

SUBSET_PERCENTS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

N_CLIPS = 20

# =========================================================
# HELPERS
# =========================================================
def extract_layer_name(key):
    match = re.search(r"(conv\d+|fc\d+)$", key)
    return match.group(1)


def compute_rdm_from_files(files, layer, method):

    activations = []

    for file in files:
        path = os.path.join(ACTIVATIONS_FOLDER, file)
        data = np.load(path)

        for key in data.files:
            if extract_layer_name(key) == layer:
                activations.append(data[key])

    activations = np.concatenate(activations, axis=0)

    dataset = Dataset(
        activations,
        obs_descriptors={"clips": np.arange(len(files))},
        channel_descriptors={"units": np.arange(activations.shape[1])}
    )

    rdm_obj = calc_rdm(dataset, method=method)
    return rdm_obj.get_matrices()[0]


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


def embedding_to_rdm(X):
    diff = X[:, None, :] - X[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def compare_rdms(rdm1, rdm2):
    idx = np.triu_indices_from(rdm1, k=1)
    return spearmanr(rdm1[idx], rdm2[idx]).correlation


# =========================================================
# MAIN LOOP
# =========================================================
all_results = []

for game in GAMES:
    print(f"\n===== GAME: {game} =====")

    all_files = [
        f for f in os.listdir(ACTIVATIONS_FOLDER)
        if f.endswith("_activations.npz") and game in f.lower()
    ]

    for layer in LAYERS:

        print(f"\n--- LAYER: {layer} ---")

        for repeat in range(N_REPEATS):

            subset_files = random.sample(all_files, N_CLIPS)
            rdm = compute_rdm_from_files(subset_files, layer, METHOD)
            all_triplets = build_triplets_from_rdm(rdm)

            for p in SUBSET_PERCENTS:

                n_triplets = int(len(all_triplets) * p)

                idxs = np.random.choice(len(all_triplets), size=n_triplets, replace=False)
                subset = all_triplets[idxs]

                X = cy_tste.tste(
                    subset,
                    no_dims=DIM,
                    max_iter=MAX_ITER,
                    verbose=False,
                    use_log=True
                )

                rdm_rec = embedding_to_rdm(X)
                score = compare_rdms(rdm, rdm_rec)

                all_results.append({
                    "game": game,
                    "layer": layer,
                    "n_clips": N_CLIPS,
                    "percent": p,
                    "n_triplets": n_triplets,
                    "repeat": repeat,
                    "score": score
                })


# =========================================================
# SAVE RESULTS
# =========================================================
df = pd.DataFrame(all_results)

df.to_csv(os.path.join(SAVE_FOLDER, "triplet_results_raw_20clips.csv"),
          float_format='%.4f', index=False)

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

summary.to_csv(os.path.join(SAVE_FOLDER, "triplet_results_summary_{N_CLIPS}clips.csv"),
               float_format='%.4f', index=False)

print("\nSaved results.")


# =========================================================
# ORGANIZE RESULTS FOR PLOTTING
# =========================================================
results_struct = {
    layer: {
        game: {p: [] for p in SUBSET_PERCENTS}
        for game in GAMES
    }
    for layer in LAYERS
}

for _, row in df.iterrows():
    results_struct[row["layer"]][row["game"]][row["percent"]].append(row["score"])


# =========================================================
# PLOTTING
# =========================================================
def plot_layer(layer):

    layer_folder = os.path.join(SAVE_FOLDER, layer)
    os.makedirs(layer_folder, exist_ok=True)

    # ---------- PER GAME ----------
    for game in GAMES:

        means = []
        data = []

        for p in SUBSET_PERCENTS:
            vals = np.array(results_struct[layer][game][p])
            means.append(vals.mean())
            data.append(vals)

        x_vals = [int(p * 100) for p in SUBSET_PERCENTS]

        # LINE
        plt.figure()
        plt.plot(x_vals, means, marker='o')
        plt.xlabel("Percentage of triplets")
        plt.ylabel("Mean RSA recovery")
        plt.title(f"{layer} - {game}")
        plt.grid(True)
        plt.savefig(os.path.join(layer_folder, f"{game}_mean.png"))
        plt.close()

        # BOX
        plt.figure()
        plt.boxplot(data, labels=x_vals)
        plt.xlabel("Percentage of triplets")
        plt.ylabel("RSA recovery")
        plt.title(f"{layer} - {game}")
        plt.grid(True)
        plt.savefig(os.path.join(layer_folder, f"{game}_box.png"))
        plt.close()

    # ---------- GLOBAL ----------
    means = []
    data = []

    for p in SUBSET_PERCENTS:
        pooled = []
        for game in GAMES:
            pooled.extend(results_struct[layer][game][p])

        pooled = np.array(pooled)
        means.append(pooled.mean())
        data.append(pooled)

    x_vals = [int(p * 100) for p in SUBSET_PERCENTS]

    # LINE
    plt.figure()
    plt.plot(x_vals, means, marker='o')
    plt.xlabel("Percentage of triplets")
    plt.ylabel("Mean RSA recovery")
    plt.title(f"{layer} - GLOBAL")
    plt.grid(True)
    plt.savefig(os.path.join(layer_folder, "global_mean.png"))
    plt.close()

    # BOX
    plt.figure()
    plt.boxplot(data, labels=x_vals)
    plt.xlabel("Percentage of triplets")
    plt.ylabel("RSA recovery")
    plt.title(f"{layer} - GLOBAL")
    plt.grid(True)
    plt.savefig(os.path.join(layer_folder, "global_box.png"))
    plt.close()


# =========================================================
# RUN PLOTS
# =========================================================
for layer in LAYERS:
    plot_layer(layer)

print("All plots saved.")