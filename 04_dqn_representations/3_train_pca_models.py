import os
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.config import GAMES, REPR, get_path, ensure
from src.utils import extract_layer_name

# =========================================================
# CONFIG
# =========================================================
# Single seed: PCA is trained on one fixed recording session,
# then applied to all seeds at inference time.
# Change this to the seed whose pca_training recordings you want to use.
SEED = "seed_42"

N_COMPONENTS = REPR["n_pca_components"]     # 100
NORMALIZE    = True                         # required for correlation distance

ACTIVATIONS_FOLDER = get_path("activations_pca_seed", seed=SEED)

# =========================================================
# PROCESS EACH GAME
# =========================================================
for game in GAMES:

    print("\n" + "=" * 60)
    print(f"TRAINING PCA FOR GAME: {game}")
    print("=" * 60)

    activation_files = [
        f for f in os.listdir(ACTIVATIONS_FOLDER)
        if f.endswith("_activations.npz") and game in f.lower()
    ]

    if not activation_files:
        print(f"No activation files found for {game}")
        continue

    print(f"Found {len(activation_files)} activation files")

    # Collect activations per layer
    layer_activations = {}
    for file in tqdm(activation_files):
        data = np.load(ACTIVATIONS_FOLDER / file)
        for key in data.files:
            layer_name = extract_layer_name(key)
            layer_activations.setdefault(layer_name, []).append(data[key])

    # Train PCA per layer
    for layer_name in sorted(layer_activations.keys()):
        print(f"\n{'-'*50}\nLayer: {layer_name}\n{'-'*50}")

        X = np.concatenate(layer_activations[layer_name], axis=0)
        print(f"Original shape: {X.shape}")

        scaler = None
        if NORMALIZE:
            scaler = StandardScaler()
            X      = scaler.fit_transform(X)
            print("Features normalized")

        n_components = min(N_COMPONENTS, X.shape[0], X.shape[1])
        print(f"Training PCA with {n_components} components")

        pca     = PCA(n_components=n_components, svd_solver="full")
        X_pca   = pca.fit_transform(X)
        explained = np.sum(pca.explained_variance_ratio_)

        print(f"PCA output shape:   {X_pca.shape}")
        print(f"Explained variance: {explained:.4f}")

        # Save — path from config: models/pca_models/multi_seed/{game}/{seed}/
        save_dir  = ensure("models_pca_layer", game=game, seed=SEED)
        save_path = save_dir / f"{game}_{layer_name}_pca.pkl"

        joblib.dump({"pca": pca, "scaler": scaler}, save_path)
        print(f"Saved PCA model: {save_path}")

print("\nAll PCA models trained successfully.")
