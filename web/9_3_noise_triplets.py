import numpy as np
import os
import pandas as pd
from itertools import combinations
from scipy.stats import spearmanr
from tqdm import tqdm
import cy_tste
import random
import re

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
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4]

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


# =========================================================
# FIXED NOISE MODEL
# =========================================================
def add_noise_to_triplets(triplets, noise_level):
    noisy = []

    for (a, b, c) in triplets:
        if np.random.rand() < noise_level:
            noisy.append((a, c, b))  # flip constraint
        else:
            noisy.append((a, b, c))

    return np.array(noisy, dtype=np.int32)


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

        for noise in NOISE_LEVELS:

            print(f"\n=== NOISE {noise} ===")

            for repeat in tqdm(range(N_REPEATS)):

                # -------------------------------
                # 1. SAMPLE CLIPS
                # -------------------------------
                subset_files = random.sample(all_files, N_CLIPS)

                # -------------------------------
                # 2. BUILD RDM
                # -------------------------------
                rdm = compute_rdm_from_files(subset_files, layer, METHOD)

                # -------------------------------
                # 3. BUILD TRIPLETS
                # -------------------------------
                all_triplets = build_triplets_from_rdm(rdm)

                for p in SUBSET_PERCENTS:

                    n_triplets = int(len(all_triplets) * p)

                    idxs = np.random.choice(
                        len(all_triplets),
                        size=n_triplets,
                        replace=False
                    )

                    subset = all_triplets[idxs]

                    # -------------------------------
                    # 4. ADD NOISE
                    # -------------------------------
                    subset_noisy = add_noise_to_triplets(subset, noise)

                    # -------------------------------
                    # 5. t-STE
                    # -------------------------------
                    X = cy_tste.tste(
                        subset_noisy,
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
                        "noise": noise,
                        "n_triplets": n_triplets,
                        "repeat": repeat,
                        "score": score
                    })


# =========================================================
# SAVE
# =========================================================
df = pd.DataFrame(all_results)

df.to_csv(os.path.join(SAVE_FOLDER, "noise_triplet_results_raw_20clips.csv"),
          float_format="%.4f", index=False)

summary = (
    df.groupby(["game", "layer", "percent", "noise"])
    .agg(
        mean_score=("score", "mean"),
        std_score=("score", "std"),
        min_score=("score", "min"),
        max_score=("score", "max"),
    )
    .reset_index()
)

summary.to_csv(os.path.join(SAVE_FOLDER, "noise_triplet_results_summary_20clips.csv"),
               float_format="%.4f", index=False)

print("\nSaved results.")