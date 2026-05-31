import os
import re
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================
GAMES = ["pacman", "pong", "spaceinvaders"]
SEED="seed_42"
ACTIVATIONS_FOLDER = f"../data/test_16_PRUEBAS/pca_training/{SEED}"

SAVE_FOLDER = "../models/pca_models/"

N_COMPONENTS = 100
NORMALIZE = True

os.makedirs(SAVE_FOLDER,  exist_ok=True)

# =========================================================
# HELPER
# =========================================================
def extract_layer_name(key):
    """
    Example:
    clipname_conv1 -> conv1
    clipname_fc -> fc
    """
    match = re.search(r"(conv\d+|fc)$", key)

    if match:
        return match.group(1)

    raise ValueError(f"Could not extract layer name from key: {key}")

# =========================================================
# PROCESS EACH GAME
# =========================================================
for game in GAMES:

    print("\n" + "=" * 60)
    print(f"TRAINING PCA FOR GAME: {game}")
    print("=" * 60)

    # -----------------------------------------------------
    # Find activation files
    # -----------------------------------------------------
    activation_files = [
        f for f in os.listdir(ACTIVATIONS_FOLDER)
        if f.endswith("_activations.npz") and game in f.lower()
    ]

    if len(activation_files) == 0:
        print(f"No activation files found for {game}")
        continue

    print(f"Found {len(activation_files)} activation files")

    # -----------------------------------------------------
    # Collect activations per layer
    # -----------------------------------------------------
    layer_activations = {}

    for file in tqdm(activation_files):

        file_path = os.path.join(ACTIVATIONS_FOLDER, file)
        data = np.load(file_path)

        for key in data.files:

            layer_name = extract_layer_name(key)

            act = data[key]  # shape: (1, units)

            if layer_name not in layer_activations:
                layer_activations[layer_name] = []

            layer_activations[layer_name].append(act)

    # -----------------------------------------------------
    # Train PCA PER LAYER
    # -----------------------------------------------------
    for layer_name in sorted(layer_activations.keys()):

        print("\n" + "-" * 50)
        print(f"Layer: {layer_name}")
        print("-" * 50)

        # ---------------------------------------------
        # Build matrix
        # ---------------------------------------------
        X = np.concatenate(layer_activations[layer_name], axis=0)

        print("Original shape:", X.shape)

        # ---------------------------------------------
        # Normalize features
        # IMPORTANT for correlation distance
        # ---------------------------------------------
        scaler = None

        if NORMALIZE:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            print("Features normalized")

        # ---------------------------------------------
        # PCA
        # ---------------------------------------------
        n_components = min(
            N_COMPONENTS,
            X.shape[0],   # n_samples
            X.shape[1]    # n_features
        )

        print(f"Training PCA with {n_components} components")

        pca = PCA(
            n_components=n_components,
            svd_solver="full"
        )

        X_pca = pca.fit_transform(X)

        print("PCA output shape:", X_pca.shape)

        explained = np.sum(pca.explained_variance_ratio_)

        print(f"Explained variance: {explained:.4f}")

        # ---------------------------------------------
        # Save PCA
        # ---------------------------------------------
        save_dict = {
            "pca": pca,
            "scaler": scaler
        }

        os.makedirs(os.path.join(SAVE_FOLDER, game, SEED), exist_ok=True)

        save_path = os.path.join(
            SAVE_FOLDER, game, SEED,
            f"{game}_{layer_name}_pca.pkl"
        )

        joblib.dump(save_dict, save_path)

        print(f"Saved PCA model: {save_path}")

print("\nAll PCA models trained successfully.")