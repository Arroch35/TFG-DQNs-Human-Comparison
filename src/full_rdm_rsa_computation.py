"""
Compute RDMs and RSA for DQN activations
----------------------------------------

This script:

1. Loads saved activation matrices (num_frames x num_units)
2. Computes correlation-distance RDMs for each layer using rsatoolbox
3. Computes RSA between layers using Spearman correlation
4. Saves RDM heatmaps for each layer
5. Saves RSA matrix and heatmap

No intermediate saving/loading of RDM objects.
"""

# =========================================================
# IMPORTS
# =========================================================
import os
import numpy as np
import matplotlib.pyplot as plt

import rsatoolbox
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm, compare, RDMs, concat

# =========================================================
# CONFIGURATION
# =========================================================
ACTIVATION_FILE = "../data/sub01_Pong_block1_DQN_activations.npz"
SAVE_FOLDER = "../data"

TICK_STEP = 50  # For RDM plots

# =========================================================
# LOAD ACTIVATION MATRICES
# =========================================================
# Each key = layer name, value = activations (num_frames x num_units)
data = np.load(ACTIVATION_FILE)

layer_names = list(data.files)
print("Layers found:", layer_names)

rdm_objects = []  # Store RDM objects for RSA
rdm_matrices_for_plot = {}  # Store square matrices for plotting

# =========================================================
# COMPUTE RDMs FOR EACH LAYER
# =========================================================
for key in layer_names:

    print(f"\nProcessing layer: {key}")

    activations = data[key]
    n_frames = activations.shape[0]

    # -------------------------------
    # Create rsatoolbox Dataset
    # -------------------------------
    # Wrap the activations matrix with descriptors
    dataset = Dataset(
        activations,
        obs_descriptors={"frames": np.arange(n_frames)},
        channel_descriptors={"units": np.arange(activations.shape[1])}
    )

    # -------------------------------
    # Compute correlation-distance RDM
    # -------------------------------
    rdm_obj = calc_rdm(dataset, method="correlation")

    # Store RDM object for later RSA computation
    rdm_objects.append(rdm_obj)

    # Extract square matrix for plotting
    rdm_matrix = rdm_obj.get_matrices()[0]
    rdm_matrices_for_plot[key] = rdm_matrix

    # -------------------------------
    # Plot RDM heatmap
    # -------------------------------
    plt.figure(figsize=(8, 8))
    im = plt.imshow(rdm_matrix, cmap="coolwarm", origin="upper")
    plt.colorbar(im)
    plt.title(f"RDM - {key}")
    plt.xlabel("Frame")
    plt.ylabel("Frame")
    ticks = np.arange(0, n_frames, TICK_STEP)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.tight_layout()

    png_path = os.path.join(SAVE_FOLDER, f"{key}_RDM.png")
    plt.savefig(png_path, dpi=300)
    plt.close()

    print(f"Saved heatmap: {png_path}")

# =========================================================
# COMPUTE RSA BETWEEN ALL LAYERS
# =========================================================
# rdm_objects = list of individual RDM objects (one per layer)
combined_rdms = concat(rdm_objects)

# Now combined_rdms is a proper RDMs object
# You can pass it to compare() like before
rsa_matrix = compare(combined_rdms, combined_rdms, method="spearman")

n_layers = len(layer_names)
print("\nRSA matrix shape:", rsa_matrix.shape)

# =========================================================
# SAVE RSA MATRIX
# =========================================================
rsa_save_path = os.path.join(SAVE_FOLDER, "DQN_layer_RSA_matrix.npy")
np.save(rsa_save_path, rsa_matrix)
print("Saved RSA matrix:", rsa_save_path)

# =========================================================
# PLOT RSA HEATMAP
# =========================================================
plt.figure(figsize=(6, 6))
im = plt.imshow(rsa_matrix, cmap="viridis")
plt.colorbar(im)
plt.xticks(range(n_layers), layer_names, rotation=45)
plt.yticks(range(n_layers), layer_names)
plt.title("RSA Between DQN Layers")
plt.tight_layout()
png_path = os.path.join(SAVE_FOLDER, "DQN_layer_RSA_heatmap.png")
plt.savefig(png_path, dpi=300)
plt.close()
print("Saved RSA heatmap:", png_path)


#? Este script no usa Ranked, pero el paper de referencia si. Preguntar al profe si usarla o no