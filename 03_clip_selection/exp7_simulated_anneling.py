import os
import random
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm, compare, concat

from src.config import GAMES, REFERENCE_SEED, REPR, TSTE, get_path, ensure
from src.utils import extract_layer_name, upper_tri

# =========================================================
# CONFIG
# =========================================================
SEED     = REFERENCE_SEED           # "seed_42"
METHOD   = REPR["rdm_method"]       # "correlation"
N_SELECT = TSTE["n_clips"]          # 15
N_ITER   = 300

# Paths
ACTIVATIONS_FOLDER = get_path("activations_buenos25_seed", seed=SEED)
SAVE_FOLDER        = ensure("subsets_seed",        seed=SEED)

# NOTE: "clip_maps" in config points to "maps/selected_15/{game}_clip_map.csv"
# but this script uses "maps/buenos_25/{game}_clip_map.csv" — a different source.
# Suggested addition to config.py:
#   "clip_maps_buenos25": DATA / "maps" / "buenos_25" / "{game}_clip_map.csv",
# Until then we derive it manually from the DATA path:
from src.config import DATA
MAP_FOLDER = DATA / "maps" / "buenos_25"

# =========================================================
# SAVE RSA HEATMAP
# =========================================================
def save_rsa_heatmap(rsa_matrix, layer_names, save_path, title):
    plt.figure(figsize=(6, 6))
    im = plt.imshow(rsa_matrix, cmap="viridis")
    plt.colorbar(im)
    plt.xticks(range(len(layer_names)), layer_names, rotation=45)
    plt.yticks(range(len(layer_names)), layer_names)

    for i in range(rsa_matrix.shape[0]):
        for j in range(rsa_matrix.shape[1]):
            plt.text(j, i, f"{rsa_matrix[i, j]:.2f}",
                     ha="center", va="center", color="white", fontsize=8)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================================================
# LOAD BIG RSA (TARGET)
# =========================================================
def load_big_rsa(game):
    path = get_path("rdms_big", seed=SEED, game=game) / f"{game}_DQN_layer_RSA_{METHOD}_matrix.npy"
    return np.load(path)


# =========================================================
# LOAD ACTIVATIONS FOR ONE CLIP
# =========================================================
def load_clip_activations(file_path):
    data = np.load(file_path)
    return {extract_layer_name(key): data[key].astype(np.float32) for key in data.files}


# =========================================================
# LOAD PCA FOR GAME + LAYER
# =========================================================
def load_pca(game, layer):
    pca_path = get_path("models_pca_layer", game=game, seed=SEED) / f"{game}_{layer}_pca.pkl"
    pca_data = joblib.load(pca_path)
    return pca_data["scaler"], pca_data["pca"]


# =========================================================
# COMPUTE RSA MATRIX FROM A SUBSET OF CLIPS
# =========================================================
def compute_subset_rsa(game, subset_files):
    layer_acts = {}
    clip_names = []

    for file in subset_files:
        clip_names.append(file.replace("_activations.npz", ""))
        clip_acts = load_clip_activations(ACTIVATIONS_FOLDER / file)
        for layer, act in clip_acts.items():
            layer_acts.setdefault(layer, []).append(act)

    rdm_objects = []
    layer_names = sorted(layer_acts.keys())

    for layer in layer_names:
        activations = np.concatenate(layer_acts[layer], axis=0).astype(np.float32)

        if METHOD == "correlation":
            scaler, pca = load_pca(game, layer)
            activations = scaler.transform(activations)
            activations = pca.transform(activations)

        dataset = Dataset(
            activations,
            obs_descriptors={"clips": np.array(clip_names)},
            channel_descriptors={"units": np.arange(activations.shape[1])},
        )
        rdm_objects.append(calc_rdm(dataset, method=METHOD))

    combined_rdms = concat(rdm_objects)
    rsa_matrix    = compare(combined_rdms, combined_rdms, method="spearman")

    return rsa_matrix, layer_names


# =========================================================
# OBJECTIVE: similarity between subset RSA and big RSA
# =========================================================
def subset_score(game, subset_files, target_rsa):
    subset_rsa, _ = compute_subset_rsa(game, subset_files)
    spearman, _   = spearmanr(upper_tri(target_rsa), upper_tri(subset_rsa))
    return spearman


# =========================================================
# SIMULATED ANNEALING
# =========================================================
def run_sa(game):
    print("\n" + "=" * 60)
    print(f"GAME: {game}")
    print("=" * 60)

    target_rsa = load_big_rsa(game)

    all_files = [
        f for f in os.listdir(ACTIVATIONS_FOLDER)
        if f.endswith("_activations.npz") and game in f.lower()
    ]
    print(f"Candidate clips: {len(all_files)}")

    current       = random.sample(all_files, N_SELECT)
    current_score = subset_score(game, current, target_rsa)
    best          = list(current)
    best_score    = current_score

    print(f"Initial score: {current_score:.4f}")

    for step in range(N_ITER):
        proposal  = list(current)
        out_file  = random.choice(proposal)
        proposal.remove(out_file)
        remaining = [f for f in all_files if f not in proposal]
        proposal.append(random.choice(remaining))

        proposal_score = subset_score(game, proposal, target_rsa)

        if proposal_score > current_score:
            current       = proposal
            current_score = proposal_score
            if proposal_score > best_score:
                best       = list(proposal)
                best_score = proposal_score

        if step % 100 == 0:
            print(f"[{game}] step={step} current={current_score:.4f} best={best_score:.4f}")

    return best, best_score


# =========================================================
# MAIN
# =========================================================
for game in GAMES:
    best_files, best_score = run_sa(game)

    while best_score != 1.0:
        best_files, best_score = run_sa(game)

    print("\nBEST SCORE:", best_score)

    best_rsa, layer_names = compute_subset_rsa(game, best_files)

    # Save RSA matrix
    np.save(SAVE_FOLDER / f"{game}_best_subset_RSA.npy", best_rsa)

    # Save RSA heatmap
    save_rsa_heatmap(
        best_rsa, layer_names,
        SAVE_FOLDER / f"{game}_best_subset_RSA.png",
        title=f"{game} - Best {N_SELECT} subset RSA",
    )
    print(f"Saved RSA heatmap → {SAVE_FOLDER / f'{game}_best_subset_RSA.png'}")

    # Save subset filenames
    subset_csv = SAVE_FOLDER / f"{game}_best_subset.csv"
    pd.DataFrame({"clip_file": best_files}).to_csv(subset_csv, index=False)
    print(f"Saved subset → {subset_csv}")

    # Convert filenames → clip indices via map CSV
    # Uses get_path("subsets_csv", seed=SEED, game=game) for the output path (already in config)
    map_path = MAP_FOLDER / f"{game}_clip_map.csv"

    if map_path.exists():
        clip_map = pd.read_csv(map_path)
        selected_clip_names = [f.replace("_activations.npz", ".mp4") for f in best_files]
        selected_rows = clip_map[clip_map["clip_name"].isin(selected_clip_names)]

        # get_path("subsets_csv", ...) resolves to the canonical indices file path
        indices_csv = get_path("subsets_csv", seed=SEED, game=game)
        indices_csv.parent.mkdir(parents=True, exist_ok=True)
        selected_rows.to_csv(indices_csv, index=False)
        print(f"Saved indices → {indices_csv}")

    # Save score
    (SAVE_FOLDER / f"{game}_score.txt").write_text(f"{best_score:.6f}")

print("\nDONE.")
