import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm, compare, concat

# =========================================================
# CONFIGURATION
# =========================================================
GAMES = [ "pong", "pacman", "spaceinvaders"] #"pacman", 
METHOD="correlation" #euclidean #!LAS HECHAS CON EUCLIDEAN SON ANTÍGUAS (SOLO la seed 0 es nueva, que he ambiado el código para que no use el pca para esto), NO SE SI TENDRÍA QUE QUITAR EL PCA PARA HACERLAS EUCLIDEAN
PCA_FOLDER = "../models/pca_models"
 # "../data/subset_selection/seed_42\pong_best_subset.csv" #"../data/subset_selection/buenos_25/pong_best_subset.csv"

seeds = ["seed_0", "seed_1", "seed_2", "seed_3"] #, "seed_42" ["seed_0", "seed_1", "seed_2", "seed_3", "seed_4"] --- IGNORE ---
# =========================================================
# HELPER: extract layer name from key
# Example key: sub_pruebas_MsPacmanNoFrameskip-v4_block1_end002550_frames12_conv1
# --> conv1
# =========================================================
def extract_layer_name(key):
    match = re.search(r"(conv\d+|fc)$", key) #\d+
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Could not extract layer name from key: {key}")

# =========================================================
# PROCESS EACH GAME
# =========================================================
for seed in seeds:

    ACTIVATIONS_FOLDER = f"../data/test_16_PRUEBAS/buenos_25/{seed}" # "../data/test_16_PRUEBAS/buenos_25" #"../data/DQN_activations"
    SAVE_BASE_FOLDER = f"../data/test_16_rdms/selected_subset_15/{seed}" #"../data/test_16_rdms/buenos_25" #"../data/DQN_rdms"

    os.makedirs(SAVE_BASE_FOLDER, exist_ok=True)

    for game in GAMES:
        print("\n" + "="*60)
        print(f"PROCESSING GAME: {game}")
        print("="*60)

        game_save_folder = os.path.join(SAVE_BASE_FOLDER, game)
        os.makedirs(game_save_folder, exist_ok=True)

        # -----------------------------------------------------
        # Find all activation files for this game
        # -----------------------------------------------------
        all_files = [
            f for f in os.listdir(ACTIVATIONS_FOLDER)
            if f.endswith("_activations.npz") and game in f.lower()
        ]
        FILTER_CSV = f"../data/subset_selection/seed_42/{game}_best_subset_indices.csv"

        if FILTER_CSV and os.path.exists(FILTER_CSV):
            filter_df = pd.read_csv(FILTER_CSV)
            print(filter_df.columns)
            # Assume your CSV has a column 'clip_name' (e.g., 'sub_pruebas_MsPacman...')
            # We ensure it matches the filename format
            allowed_names = set(filter_df['clip_name'].astype(str).str.replace(".mp4", "", regex=False))
            

            activation_files = [
                f for f in all_files 
                if f.replace("_activations.npz", "") in allowed_names
            ]
            print(all_files[:5])
            print(allowed_names)

            print(f"Filtering active: {len(activation_files)} of {len(all_files)} files kept.")
        else:
            activation_files = all_files
            if FILTER_CSV:
                print(f"Warning: CSV {FILTER_CSV} not found. Loading all files.")

        if len(activation_files) == 0:
            print(f"No activation files found for {game} after filtering, skipping.")
            continue

        if len(activation_files) == 0:
            print(f"No activation files found for {game}, skipping.")
            continue

        print(f"Found {len(activation_files)} activation files")

        # -----------------------------------------------------
        # Collect activations per layer across clips
        # -----------------------------------------------------
        layer_activations = {}   # layer_name -> list of (1, num_units)
        clip_names = []

        for file in activation_files:
            file_path = os.path.join(ACTIVATIONS_FOLDER, file)
            data = np.load(file_path)

            # Save clip name
            clip_name = file.replace("_activations.npz", "")
            clip_names.append(clip_name)

            for key in data.files:
                layer_name = extract_layer_name(key)
                act = data[key]   # shape: (1, num_units)

                if layer_name not in layer_activations:
                    layer_activations[layer_name] = []

                layer_activations[layer_name].append(act)

        print("Layers found:", list(layer_activations.keys()))

        # -----------------------------------------------------
        # Compute RDMs for each layer
        # -----------------------------------------------------
        rdm_objects = []
        layer_names = sorted(layer_activations.keys())

        for layer_name in layer_names:
            print(f"\nProcessing layer: {layer_name}")

            # -------------------------------------------------
            # Stack activations
            # -------------------------------------------------
            activations = np.concatenate(
                layer_activations[layer_name],
                axis=0
            ).astype(np.float32)

            print(f"Original activation shape: {activations.shape}")

            # -------------------------------------------------
            # Apply PCA only for correlation distance
            # -------------------------------------------------
            if METHOD == "correlation":

                # ---------------------------------------------
                # Load corresponding PCA model
                # ---------------------------------------------
                pca_path = os.path.join(
                    PCA_FOLDER,
                    game,
                    seed,
                    f"{game}_{layer_name}_pca.pkl"
                )

                if not os.path.exists(pca_path):
                    raise FileNotFoundError(
                        f"Missing PCA model: {pca_path}"
                    )

                pca_data = joblib.load(pca_path)

                pca = pca_data["pca"]
                scaler = pca_data["scaler"]

                print(f"Loaded PCA: {pca_path}")

                # ---------------------------------------------
                # Normalize using TRAIN scaler
                # ---------------------------------------------
                if scaler is not None:
                    activations = scaler.transform(activations)

                # ---------------------------------------------
                # Apply PCA projection
                # ---------------------------------------------
                activations = pca.transform(activations)

                print(f"PCA activation shape: {activations.shape}")

            else:
                print("Skipping PCA for euclidean distance")



            n_clips = activations.shape[0]

            print(f"Activation matrix shape: {activations.shape}")

            # -------------------------------
            # Create rsatoolbox Dataset
            # -------------------------------
            dataset = Dataset(
                activations,
                obs_descriptors={"clips": np.array(clip_names)},
                channel_descriptors={"units": np.arange(activations.shape[1])}
            )

            # -------------------------------
            # Compute correlation-distance RDM  
            # -------------------------------
            rdm_obj = calc_rdm(dataset, method=METHOD) #METHOD="correlation"
            rdm_objects.append(rdm_obj)

            # Extract square matrix
            rdm_matrix = rdm_obj.get_matrices()[0]

            # -------------------------------
            # Save RDM matrix
            # -------------------------------
            npy_path = os.path.join(game_save_folder, f"{game}_{layer_name}_{METHOD}_RDM.npy")
            np.save(npy_path, rdm_matrix)
            print(f"Saved RDM matrix: {npy_path}")

            # -------------------------------
            # Plot RDM heatmap
            # -------------------------------
            plt.figure(figsize=(8, 8))
            im = plt.imshow(rdm_matrix, cmap="coolwarm", origin="upper")
            plt.colorbar(im)
            plt.title(f"{game} - RDM ({layer_name})")
            plt.xlabel("Clip")
            plt.ylabel("Clip")

            ticks = np.arange(n_clips)
            plt.xticks(ticks, rotation=90, fontsize=6)
            plt.yticks(ticks, fontsize=6)

            plt.tight_layout()
            png_path = os.path.join(game_save_folder, f"{game}_{layer_name}_{METHOD}_RDM.png")
            plt.savefig(png_path, dpi=300)
            plt.close()

            print(f"Saved heatmap: {png_path}")

        # -----------------------------------------------------
        # Compute RSA between layers
        # -----------------------------------------------------
        print("\nComputing RSA between layers...")
        combined_rdms = concat(rdm_objects)
        rsa_matrix = compare(combined_rdms, combined_rdms, method="spearman")

        print("RSA matrix shape:", rsa_matrix.shape)

        # Save RSA matrix
        rsa_save_path = os.path.join(game_save_folder, f"{game}_DQN_layer_RSA_{METHOD}_matrix.npy")
        np.save(rsa_save_path, rsa_matrix)
        print("Saved RSA matrix:", rsa_save_path)

        # Plot RSA heatmap
        plt.figure(figsize=(6, 6))
        im = plt.imshow(rsa_matrix, cmap="viridis")
        plt.colorbar(im)
        plt.xticks(range(len(layer_names)), layer_names, rotation=45)
        plt.yticks(range(len(layer_names)), layer_names)

        # -------------------------------------------------
        # Add correlation values inside each cell
        # -------------------------------------------------
        for i in range(rsa_matrix.shape[0]):
            for j in range(rsa_matrix.shape[1]):
                value = rsa_matrix[i, j]

                plt.text(
                    j, i,
                    f"{value:.2f}",      # 2 decimal places
                    ha="center",
                    va="center",
                    color="white",       # change to black if needed
                    fontsize=8
                )

        plt.title(f"{game} - RSA Between DQN Layers")
        plt.tight_layout()

        png_path = os.path.join(game_save_folder, f"{game}_DQN_layer_{METHOD}_RSA_heatmap.png")
        plt.savefig(png_path, dpi=300)
        plt.close()

        print("Saved RSA heatmap:", png_path)

        print(f"Number of clips used: {len(clip_names)}")

        # print("Clip order used for DQN:")
        # for i, name in enumerate(clip_names):
        #     print(i, name)

    print("\nAll DQN RDMs and RSA matrices computed successfully.")

# sub_big_rdm_MsPacmanNoFrameskip-v4_block1_end046400 ULTIMO FILE