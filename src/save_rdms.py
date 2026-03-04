"""
Convert DQN Activations to RDMs + Save Heatmaps

This script:

1. Loads saved activation matrices:
      shape = (num_frames, num_units)

2. Computes pairwise correlation distances between frames:
      RDM[i, j] = 1 - PearsonCorr(frame_i, frame_j)

3. Optionally rank-transforms distances (as in RSA papers).

4. Saves:
      - RDM matrices (.npz)
      - Heatmap images (.png)
"""

# =========================================================
# IMPORT LIBRARIES
# =========================================================
import os                           # File path handling
import numpy as np                  # Numerical arrays
import matplotlib.pyplot as plt     # Plotting
from scipy.spatial.distance import pdist, squareform  # Distance computation
from scipy.stats import rankdata    # Ranking distances


# =========================================================
# CONFIGURATION
# =========================================================
ACTIVATION_FILE = "../data/sub01_Pong_block1_DQN_activations.npz"
SAVE_FOLDER = "../data"
RANK_TRANSFORM = True      # Whether to rank-order RDM (paper style)
TICK_STEP = 50             # Show axis tick every 50 frames


# =========================================================
# LOAD ACTIVATION FILE
# =========================================================
# The file contains multiple arrays:
#   key = layer name
#   value = activation matrix (num_frames x num_units)
data = np.load(ACTIVATION_FILE)

# Dictionary to store computed RDMs
rdms = {}


# =========================================================
# LOOP THROUGH EACH LAYER
# =========================================================
for key in data.files:

    print(f"\nProcessing layer: {key}")

    # -----------------------------------------------------
    # Load activation matrix for this layer
    # shape: (num_frames, num_units)
    # Each row = one frame representation
    # -----------------------------------------------------
    activations = data[key]

    # Number of frames
    n_frames = activations.shape[0]

    # =====================================================
    # STEP 1: COMPUTE PAIRWISE CORRELATION DISTANCES
    # =====================================================
    # pdist computes distance between all pairs of rows
    # metric="correlation" means:
    #   distance = 1 - Pearson correlation
    #
    # Result is a 1D condensed vector containing:
    #   n_frames * (n_frames - 1) / 2 distances
    # -----------------------------------------------------
    distances = pdist(activations, metric="correlation")

    # Convert condensed vector into full square matrix
    # Result shape: (n_frames, n_frames)
    rdm = squareform(distances)

    # Now:
    # rdm[i, j] = dissimilarity between frame i and j
    # Diagonal automatically = 0

    rdm_before_rank=rdm
    # =====================================================
    # STEP 2: OPTIONAL RANK TRANSFORM (RSA STYLE)
    # =====================================================
    if RANK_TRANSFORM:

        # Get indices of upper triangle (excluding diagonal)
        # We only use upper triangle because:
        #   - Matrix is symmetric
        #   - We don't want to double-count distances
        upper_idx = np.triu_indices_from(rdm, k=1)

        # Extract all unique distances
        upper_values = rdm[upper_idx]

        # Rank the distances (smallest = rank 1)
        ranked = rankdata(upper_values)

        # Create empty matrix with same shape
        ranked_rdm = np.zeros_like(rdm)

        # Fill upper triangle with ranked distances
        ranked_rdm[upper_idx] = ranked

        # Mirror upper triangle to lower triangle
        ranked_rdm = ranked_rdm + ranked_rdm.T

        # Replace original RDM with ranked version
        rdm = ranked_rdm


    # Store RDM in dictionary
    rdms[key] = rdm

    print(f"RDM shape: {rdm.shape}")


    # =====================================================
    # STEP 3: SAVE HEATMAP IMAGE
    # =====================================================

    # Create new figure
    plt.figure(figsize=(8, 8))

    # IMPORTANT:
    # origin="lower" means:
    #   row 0 will appear at bottom
    #
    # Remove origin="lower" if you want standard matrix orientation
    #? Doing the image with correlation, NOT RANK
    im = plt.imshow(rdm_before_rank, cmap="coolwarm", origin="upper")

    # Add colorbar scale
    plt.colorbar(im)

    # Title
    plt.title(f"RDM - {key}")

    # Axis labels
    plt.xlabel("Frame")
    plt.ylabel("Frame")

    # Show tick every 50 frames (but data still contains all frames)
    ticks = np.arange(0, n_frames, TICK_STEP)
    plt.xticks(ticks)
    plt.yticks(ticks)

    # Adjust layout to avoid cropping
    plt.tight_layout()

    # Save PNG
    png_path = os.path.join(SAVE_FOLDER, f"{key}_RDM.png")
    plt.savefig(png_path, dpi=300)

    # Close figure (important to avoid memory issues)
    plt.close()

    print(f"Saved heatmap: {png_path}")


# =========================================================
# SAVE ALL RDM MATRICES
# =========================================================
rdm_save_path = os.path.join(
    SAVE_FOLDER,
    "sub01_Pong_block1_DQN_RDMs.npz"
)

np.savez_compressed(rdm_save_path, **rdms)

print(f"\nSaved RDM matrices to:\n{rdm_save_path}")