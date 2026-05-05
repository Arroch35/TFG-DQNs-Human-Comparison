import os
import re
import numpy as np
import matplotlib.pyplot as plt

from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm, compare, concat

# =========================================================
# CONFIGURATION
# =========================================================
GAMES = ["pacman", "pong", "spaceinvaders"]
METHOD="euclidean" #euclidean #!LO HE CAMBIADO A euclidean
ACTIVATIONS_FOLDER = "../data/test_16_PRUEBAS/buenos_25" #"../data/test_16_PRUEBAS/big_rdm_equal_size" #"../data/DQN_activations"
SAVE_BASE_FOLDER = "../data/test_16_rdms/buenos_25" #"../data/test_16_rdms/big_rdm_equal_size" #"../data/DQN_rdms"

os.makedirs(SAVE_BASE_FOLDER, exist_ok=True)

# =========================================================
# HELPER: extract layer name from key
# Example key: sub_pruebas_MsPacmanNoFrameskip-v4_block1_end002550_frames12_conv1
# --> conv1
# =========================================================
def extract_layer_name(key):
    match = re.search(r"(conv\d+|fc\d+)$", key)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Could not extract layer name from key: {key}")

# =========================================================
# PROCESS EACH GAME
# =========================================================
for game in GAMES:
    print("\n" + "="*60)
    print(f"PROCESSING GAME: {game}")
    print("="*60)

    game_save_folder = os.path.join(SAVE_BASE_FOLDER, game)
    os.makedirs(game_save_folder, exist_ok=True)

    # -----------------------------------------------------
    # Find all activation files for this game
    # -----------------------------------------------------
    activation_files = [
        f for f in os.listdir(ACTIVATIONS_FOLDER)
        if f.endswith("_activations.npz") and game in f.lower()
    ]

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

        # Stack all clips: (n_clips, num_units)
        activations = np.concatenate(layer_activations[layer_name], axis=0)
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