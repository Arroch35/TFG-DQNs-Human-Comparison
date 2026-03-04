"""
Compute RSA between DQN layers

This script:
1. Loads saved RDMs (.npz)
2. Extracts upper triangle of each RDM
3. Computes Spearman correlation between layers
4. Saves RSA matrix
5. Plots RSA heatmap
"""

# =========================================================
# IMPORTS
# =========================================================
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# =========================================================
# CONFIGURATION
# =========================================================
RDM_FILE = "../data/sub01_Pong_block1_DQN_RDMs.npz"
SAVE_FOLDER = "../data"


# =========================================================
# LOAD RDM FILE
# =========================================================
data = np.load(RDM_FILE)

layer_names = data.files
n_layers = len(layer_names)

print("Layers found:", layer_names)


# =========================================================
# EXTRACT UPPER TRIANGLE FOR EACH LAYER
# =========================================================
# We store flattened upper triangles here
rdm_vectors = {}

for layer in layer_names:

    rdm = data[layer]

    # Get upper triangle indices (exclude diagonal)
    upper_idx = np.triu_indices_from(rdm, k=1)

    # Extract unique pairwise distances
    vector = rdm[upper_idx]

    rdm_vectors[layer] = vector


# =========================================================
# COMPUTE RSA MATRIX (Layer x Layer)
# =========================================================
rsa_matrix = np.zeros((n_layers, n_layers))

for i, layer_i in enumerate(layer_names):
    for j, layer_j in enumerate(layer_names):

        vec_i = rdm_vectors[layer_i]
        vec_j = rdm_vectors[layer_j]

        # Spearman correlation
        corr, _ = spearmanr(vec_i, vec_j)

        rsa_matrix[i, j] = corr


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