import os
import random
import re
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm, compare, concat
from src.utils import extract_layer_name, upper_tri

#? ENSEGUIDA CONSIGE UN BUEN SUBSET, PUEDE QUE LO TENGA QUE ADAPTAR PARA QUE, A PARTE DE ESTO, COMPUTE LO UQE TENIA ANTES, LO DE LA MEJOR SEPARACIÓN ENTE LOS CLIPS

# =========================================================
# CONFIG
# =========================================================
GAMES = ["pong", "pacman", "spaceinvaders"]

SEED = "seed_42"
METHOD = "correlation"

# Pool of 25 candidate clips
ACTIVATIONS_FOLDER = f"../data/test_16_PRUEBAS/buenos_25/{SEED}"

# Big-RDM reference
BIG_RSA_FOLDER = f"../data/test_16_rdms/big_rdm_equal_size/{SEED}"

# PCA models
PCA_FOLDER = "../models/pca_models"

# Clip maps
MAP_FOLDER = "../data/maps/buenos_25"

# Output
SAVE_FOLDER = f"../data/subset_selection/{SEED}"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# SA parameters
N_SELECT = 15
N_ITER = 300

# =========================================================
# SAVE RSA HEATMAP
# =========================================================
def save_rsa_heatmap(rsa_matrix, layer_names, save_path, title):

    plt.figure(figsize=(6, 6))

    im = plt.imshow(rsa_matrix, cmap="viridis")

    plt.colorbar(im)

    plt.xticks(
        range(len(layer_names)),
        layer_names,
        rotation=45
    )

    plt.yticks(
        range(len(layer_names)),
        layer_names
    )

    # values inside cells
    for i in range(rsa_matrix.shape[0]):
        for j in range(rsa_matrix.shape[1]):

            plt.text(
                j,
                i,
                f"{rsa_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=8
            )

    plt.title(title)

    plt.tight_layout()

    plt.savefig(save_path, dpi=300)

    plt.close()

# =========================================================
# LOAD BIG RSA (TARGET)
# =========================================================
def load_big_rsa(game):

    path = os.path.join(
        BIG_RSA_FOLDER,
        game,
        f"{game}_DQN_layer_RSA_{METHOD}_matrix.npy"
    )

    return np.load(path)


# =========================================================
# LOAD ACTIVATIONS FOR ONE CLIP
# =========================================================
def load_clip_activations(file_path):

    data = np.load(file_path)

    activations = {}

    for key in data.files:

        layer_name = extract_layer_name(key)

        act = data[key].astype(np.float32)

        activations[layer_name] = act

    return activations


# =========================================================
# LOAD PCA FOR GAME + LAYER
# =========================================================
def load_pca(game, layer):

    pca_path = os.path.join(
        PCA_FOLDER,
        game,
        SEED,
        f"{game}_{layer}_pca.pkl"
    )

    pca_data = joblib.load(pca_path)

    return pca_data["scaler"], pca_data["pca"]


# =========================================================
# COMPUTE RSA MATRIX FROM A SUBSET OF CLIPS
# =========================================================
def compute_subset_rsa(game, subset_files):

    # ---------------------------------------------
    # Collect activations per layer
    # ---------------------------------------------
    layer_acts = {}
    clip_names = []

    for file in subset_files:

        clip_name = file.replace("_activations.npz", "")
        clip_names.append(clip_name)

        path = os.path.join(ACTIVATIONS_FOLDER, file)

        clip_acts = load_clip_activations(path)

        for layer, act in clip_acts.items():

            if layer not in layer_acts:
                layer_acts[layer] = []

            layer_acts[layer].append(act)

    # ---------------------------------------------
    # Compute RDMs
    # ---------------------------------------------
    rdm_objects = []

    layer_names = sorted(layer_acts.keys())

    for layer in layer_names:

        activations = np.concatenate(
            layer_acts[layer],
            axis=0
        ).astype(np.float32)

        # -----------------------------------------
        # PCA ONLY FOR CORRELATION
        # -----------------------------------------
        if METHOD == "correlation":

            scaler, pca = load_pca(game, layer)

            activations = scaler.transform(activations)

            activations = pca.transform(activations)

        dataset = Dataset(
            activations,
            obs_descriptors={
                "clips": np.array(clip_names)
            },
            channel_descriptors={
                "units": np.arange(activations.shape[1])
            }
        )

        rdm_obj = calc_rdm(dataset, method=METHOD)

        rdm_objects.append(rdm_obj)

    # ---------------------------------------------
    # Layer RSA matrix
    # ---------------------------------------------
    combined_rdms = concat(rdm_objects)

    rsa_matrix = compare(
        combined_rdms,
        combined_rdms,
        method="spearman"
    )

    return rsa_matrix, layer_names


# =========================================================
# OBJECTIVE:
# similarity between subset RSA and big RSA
# =========================================================
def subset_score(game, subset_files, target_rsa):

    subset_rsa, _ = compute_subset_rsa(
        game,
        subset_files
    )

    v1 = upper_tri(target_rsa)
    v2 = upper_tri(subset_rsa)

    #pearson = np.corrcoef(v1, v2)[0,1]
    spearman, _ = spearmanr(v1, v2)

    #score = 0.5 * pearson + 0.5 * spearman

    return spearman


# =========================================================
# SIMULATED ANNEALING
# =========================================================
def run_sa(game):

    print("\n" + "=" * 60)
    print(f"GAME: {game}")
    print("=" * 60)

    # ---------------------------------------------
    # Load target big RSA
    # ---------------------------------------------
    target_rsa = load_big_rsa(game)

    # ---------------------------------------------
    # Candidate files
    # ---------------------------------------------
    all_files = [
        f for f in os.listdir(ACTIVATIONS_FOLDER)
        if f.endswith("_activations.npz")
        and game in f.lower()
    ]

    print(f"Candidate clips: {len(all_files)}")

    # ---------------------------------------------
    # Initial subset
    # ---------------------------------------------
    current = random.sample(all_files, N_SELECT)

    current_score = subset_score(
        game,
        current,
        target_rsa
    )

    best = list(current)
    best_score = current_score

    print(f"Initial score: {current_score:.4f}")

    # ---------------------------------------------
    # SA loop
    # ---------------------------------------------
    for step in range(N_ITER):

        proposal = list(current)

        # remove one
        out_file = random.choice(proposal)
        proposal.remove(out_file)

        # add another
        remaining = [
            f for f in all_files
            if f not in proposal
        ]

        proposal.append(
            random.choice(remaining)
        )

        # evaluate
        proposal_score = subset_score(
            game,
            proposal,
            target_rsa
        )

        # greedy accept
        if proposal_score > current_score:

            current = proposal
            current_score = proposal_score

            if proposal_score > best_score:

                best = list(proposal)
                best_score = proposal_score

        if step % 100 == 0:

            print(
                f"[{game}] "
                f"step={step} "
                f"current={current_score:.4f} "
                f"best={best_score:.4f}"
            )

    return best, best_score


# =========================================================
# MAIN
# =========================================================
for game in GAMES:

    best_files, best_score = run_sa(game)

    while best_score!=1.0:
        best_files, best_score = run_sa(game)


    print("\nBEST SCORE:", best_score)

    # -----------------------------------------------------
    # Recompute BEST RSA
    # -----------------------------------------------------
    best_rsa, layer_names = compute_subset_rsa(
        game,
        best_files
    )

    # -----------------------------------------------------
    # Save RSA matrix
    # -----------------------------------------------------
    rsa_npy_path = os.path.join(
        SAVE_FOLDER,
        f"{game}_best_subset_RSA.npy"
    )

    np.save(rsa_npy_path, best_rsa)

    # -----------------------------------------------------
    # Save RSA heatmap
    # -----------------------------------------------------
    rsa_png_path = os.path.join(
        SAVE_FOLDER,
        f"{game}_best_subset_RSA.png"
    )

    save_rsa_heatmap(
        best_rsa,
        layer_names,
        rsa_png_path,
        title=f"{game} - Best 15 subset RSA"
    )

    print(f"Saved RSA heatmap → {rsa_png_path}")

    # -----------------------------------------------------
    # Save subset file names
    # -----------------------------------------------------
    subset_df = pd.DataFrame({
        "clip_file": best_files
    })

    subset_csv = os.path.join(
        SAVE_FOLDER,
        f"{game}_best_subset.csv"
    )

    subset_df.to_csv(subset_csv, index=False)

    print(f"Saved subset → {subset_csv}")



    # -----------------------------------------------------
    # Convert filenames to clip indices
    # using your map csv
    # -----------------------------------------------------
    map_path = os.path.join(
        MAP_FOLDER,
        f"{game}_clip_map.csv"
    )

    if os.path.exists(map_path):

        clip_map = pd.read_csv(map_path)

        selected_clip_names = [
            f.replace("_activations.npz", ".mp4")
            for f in best_files
        ]

        selected_rows = clip_map[
            clip_map["clip_name"].isin(selected_clip_names)
]

        indices_csv = os.path.join(
            SAVE_FOLDER,
            f"{game}_best_subset_indices.csv"
        )

        selected_rows.to_csv(
            indices_csv,
            index=False
        )

        print(f"Saved indices → {indices_csv}")

    # -----------------------------------------------------
    # Save score
    # -----------------------------------------------------
    with open(
        os.path.join(
            SAVE_FOLDER,
            f"{game}_score.txt"
        ),
        "w"
    ) as f:

        f.write(f"{best_score:.6f}")

print("\nDONE.")