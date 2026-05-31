import os
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================

GAMES = ["pacman", "pong", "spaceinvaders"]

SEED = "seed_42"

DATA_FOLDER = f"../data/dqn_state_action_qvalue/{SEED}/pca_training_set/"

SAVE_FOLDER = "../models/pca_models/pixel_pca_models/"

N_COMPONENTS = 100
NORMALIZE = True

os.makedirs(SAVE_FOLDER, exist_ok=True)

# =========================================================
# PROCESS EACH GAME
# =========================================================

for game in GAMES:

    print("\n" + "=" * 60)
    print(f"TRAINING PCA FOR GAME: {game}")
    print("=" * 60)

    # -----------------------------------------------------
    # Find files
    # -----------------------------------------------------

    game_file_path = os.path.join(DATA_FOLDER, game)

    files = [
        f for f in os.listdir(game_file_path)
        if f.endswith(".npz") and game in f.lower()
    ]

    if len(files) == 0:
        print(f"No files found for {game}")
        continue

    print(f"Found {len(files)} files")

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    X = []

    for file in tqdm(files):

        file_path = os.path.join(game_file_path, file)
        data = np.load(file_path)

        # only one key now: "state"
        state = data["state"]

        X.append(state)

    X = np.array(X)

    print("Original shape:", X.shape)  # (N_samples, 28224)

    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    scaler = None

    if NORMALIZE:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        print("Features normalized")

    # -----------------------------------------------------
    # PCA
    # -----------------------------------------------------

    n_components = min(
        N_COMPONENTS,
        X.shape[0],
        X.shape[1]
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

    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------

    save_dict = {
        "pca": pca,
        "scaler": scaler
    }

    os.makedirs(os.path.join(SAVE_FOLDER, game, SEED), exist_ok=True)

    save_path = os.path.join(
        SAVE_FOLDER,
        game,
        SEED,
        f"{game}_state_pca.pkl"
    )

    joblib.dump(save_dict, save_path)

    print(f"Saved PCA model: {save_path}")

print("\nAll PCA models trained successfully.")