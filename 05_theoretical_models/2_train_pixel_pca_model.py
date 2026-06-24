"""
2_train_pixel_pca_model.py
Train per-game, per-seed PCA models on flattened pixel states
produced by 1_build_pca_training_set.py.
"""
import os
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.config import GAMES, SEEDS, REPR, get_path, ensure

# =========================================================
# CONFIG
# =========================================================
N_COMPONENTS = REPR["n_pca_components"]   # 100
NORMALIZE    = True

# =========================================================
# MAIN
# =========================================================
for seed in SEEDS:
    print(f"\n{'='*60}\nSeed: {seed}\n{'='*60}")

    for game in GAMES:
        print(f"\n{'='*60}\nGame: {game}\n{'='*60}")

        data_folder = get_path("states_pca_game", seed=seed, game=game)

        files = [f for f in os.listdir(data_folder) if f.endswith(".npz") and game in f.lower()]

        if not files:
            print(f"No files found for {game}")
            continue

        print(f"Found {len(files)} files")

        # ── Load states ───────────────────────────────────
        X = np.array([np.load(data_folder / f)["state"] for f in tqdm(files)])
        print("Original shape:", X.shape)

        # ── Normalize ─────────────────────────────────────
        scaler = None
        if NORMALIZE:
            scaler = StandardScaler()
            X      = scaler.fit_transform(X)
            print("Features normalized")

        # ── PCA ───────────────────────────────────────────
        n_components = min(N_COMPONENTS, X.shape[0], X.shape[1])
        print(f"Training PCA with {n_components} components")

        pca   = PCA(n_components=n_components, svd_solver="full")
        X_pca = pca.fit_transform(X)

        print(f"PCA output shape: {X_pca.shape}")
        print(f"Explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")

        # ── Save ──────────────────────────────────────────
        save_path = get_path("models_pca_pixel", game=game, seed=seed) / f"{game}_state_pca.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({"pca": pca, "scaler": scaler}, save_path)
        print(f"Saved PCA model: {save_path}")

print("\nAll PCA models trained successfully.")
