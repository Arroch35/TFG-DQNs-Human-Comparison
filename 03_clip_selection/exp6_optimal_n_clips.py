import os
import random
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import spearmanr
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm, compare, concat
from src.utils import extract_layer_name

# =========================================================
# CONFIG
# =========================================================
GAMES = ["pacman", "pong", "spaceinvaders"]
METHODS = ["correlation"] #, "euclidean"
SEED = "seed_42"
PCA_FOLDER = "../models/pca_models"

ACTIVATIONS_FOLDER = f"../data/test_16_PRUEBAS/big_rdm_equal_size/{SEED}"
FULL_RSA_FOLDER = f"../data/test_16_rdms/big_rdm_equal_size/{SEED}"
OUTPUT_FOLDER = "../data/extra"

SUBSET_SIZES = [10, 12, 15, 20, 25, 30, 50]
N_SUBSETS = 1000

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =========================================================
# COMPUTE RSA
# =========================================================
def compute_layer_rsa_from_files(activation_files, method):

    layer_activations = {}
    clip_names = []

    for file in activation_files:
        path = os.path.join(ACTIVATIONS_FOLDER, file)
        data = np.load(path)

        clip_name = file.replace("_activations.npz", "")
        clip_names.append(clip_name)

        for key in data.files:
            layer_name = extract_layer_name(key)
            act = data[key]

            if layer_name not in layer_activations:
                layer_activations[layer_name] = []

            layer_activations[layer_name].append(act)

    rdm_objects = []
    layer_names = sorted(layer_activations.keys())

    for layer_name in layer_names:
        activations = np.concatenate(
            layer_activations[layer_name],
            axis=0
        ).astype(np.float32)

        # -------------------------------------------------
        # Apply PCA only for correlation distance
        # -------------------------------------------------
        if method == "correlation":

            pca_path = os.path.join(
                PCA_FOLDER,
                game,
                SEED,
                f"{game}_{layer_name}_pca.pkl"
            )

            if not os.path.exists(pca_path):
                raise FileNotFoundError(
                    f"Missing PCA model: {pca_path}"
                )

            pca_data = joblib.load(pca_path)

            pca = pca_data["pca"]
            scaler = pca_data["scaler"]

            # Normalize with training scaler
            if scaler is not None:
                activations = scaler.transform(activations)

            # PCA projection
            activations = pca.transform(activations)

        dataset = Dataset(
            activations,
            obs_descriptors={"clips": np.array(clip_names)},
            channel_descriptors={"units": np.arange(activations.shape[1])}
        )

        rdm_obj = calc_rdm(dataset, method=method)
        rdm_objects.append(rdm_obj)

    combined_rdms = concat(rdm_objects)
    rsa_matrix = compare(combined_rdms, combined_rdms, method="spearman")

    return rsa_matrix


# =========================================================
# STORAGE
# =========================================================
raw_rows = []
summary_rows = []

results = {
    game: {
        method: {size: [] for size in SUBSET_SIZES}
        for method in METHODS
    }
    for game in GAMES
}


# =========================================================
# MAIN EXPERIMENT
# =========================================================
for subset_size in SUBSET_SIZES:
    print(f"\n!!!===== {subset_size} =====!!!")

    for game in GAMES:
        print(f"\n===== {game} =====")

        all_files = [
            f for f in os.listdir(ACTIVATIONS_FOLDER)
            if f.endswith("_activations.npz") and game in f.lower()
        ]

        for method in METHODS:
            print(f"\n--- {method} ---")

            full_path = os.path.join(
                FULL_RSA_FOLDER,
                game,
                f"{game}_DQN_layer_RSA_{method}_matrix.npy"
            )
            full_rsa = np.load(full_path)

            correlations = []

            for i in range(N_SUBSETS):

                subset_files = random.sample(all_files, subset_size)

                subset_rsa = compute_layer_rsa_from_files(
                    subset_files,
                    method
                )

                idx = np.triu_indices_from(full_rsa, k=1)

                corr, _ = spearmanr(
                    full_rsa[idx],
                    subset_rsa[idx]
                )

                correlations.append(corr)

                # Save raw result
                raw_rows.append({
                    "game": game,
                    "method": method,
                    "subset_size": subset_size,
                    "iteration": i,
                    "correlation": corr
                })

            correlations = np.array(correlations)

            results[game][method][subset_size].extend(correlations)

            # Save summary
            summary_rows.append({
                "game": game,
                "method": method,
                "subset_size": subset_size,
                "mean": round(correlations.mean(), 4),
                "std": round(correlations.std(), 4),
                "min": round(correlations.min(), 4),
                "max": round(correlations.max(), 4)
            })

            print(f"Mean: {correlations.mean():.4f}")
            print(f"Std:  {correlations.std():.4f}")


# =========================================================
# SAVE CSVs
# =========================================================
raw_df = pd.DataFrame(raw_rows)
summary_df = pd.DataFrame(summary_rows)

raw_df.to_csv(os.path.join(OUTPUT_FOLDER, "raw_results.csv"), index=False)
summary_df.to_csv(os.path.join(OUTPUT_FOLDER, "summary_results.csv"), index=False)


# =========================================================
# PLOTTING
# =========================================================
def plot_per_game():

    for game in GAMES:
        for method in METHODS:

            means = []
            data = []

            for size in SUBSET_SIZES:
                vals = np.array(results[game][method][size])
                means.append(vals.mean())
                data.append(vals)

            # ---- LINE PLOT ----
            plt.figure()
            plt.plot(SUBSET_SIZES, means, marker='o')
            plt.xlabel("Number of clips")
            plt.ylabel("Mean RSA correlation")
            plt.title(f"{game} - Mean RSA vs Clips ({method})")
            plt.grid(True)

            plt.savefig(os.path.join(
                OUTPUT_FOLDER,
                f"{game}_mean_{method}.png"
            ))
            plt.close()

            # ---- BOX PLOT ----
            plt.figure()
            plt.boxplot(data, labels=SUBSET_SIZES)
            plt.xlabel("Number of clips")
            plt.ylabel("RSA correlation")
            plt.title(f"{game} - Distribution ({method})")
            plt.grid(True)

            plt.savefig(os.path.join(
                OUTPUT_FOLDER,
                f"{game}_box_{method}.png"
            ))
            plt.close()


def plot_global():

    for method in METHODS:

        means = []
        data = []

        for size in SUBSET_SIZES:

            # pool across games
            pooled = []
            for game in GAMES:
                pooled.extend(results[game][method][size])

            pooled = np.array(pooled)

            means.append(pooled.mean())
            data.append(pooled)

        # ---- LINE PLOT ----
        plt.figure()
        plt.plot(SUBSET_SIZES, means, marker='o')
        plt.xlabel("Number of clips")
        plt.ylabel("Mean RSA correlation")
        plt.title(f"GLOBAL Mean RSA vs Clips ({method})")
        plt.grid(True)

        plt.savefig(os.path.join(
            OUTPUT_FOLDER,
            f"global_mean_{method}.png"
        ))
        plt.close()

        # ---- BOX PLOT ----
        plt.figure()
        plt.boxplot(data, labels=SUBSET_SIZES)
        plt.xlabel("Number of clips")
        plt.ylabel("RSA correlation")
        plt.title(f"GLOBAL Distribution ({method})")
        plt.grid(True)

        plt.savefig(os.path.join(
            OUTPUT_FOLDER,
            f"global_box_{method}.png"
        ))
        plt.close()


# Run plots
plot_per_game()
plot_global()