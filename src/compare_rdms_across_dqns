import os
import numpy as np
import matplotlib.pyplot as plt

from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm, compare, concat

# =========================================================
# CONFIG
# =========================================================
ACTIVATION_FILES = {
    "DQN1": "../data/best_model_deterministic_policy/sub01_Pong_block2_DQN_activations.npz",
    "DQN2": "../data/lasts_RMDs/sub01_Pong_block1_DQN_activations.npz",
    "DQN3": "../data/medium_larger_training/sub01_Pong_block2_DQN_activations.npz",
}

SAVE_FOLDER = "../data"

# =========================================================
# FUNCTION: compute RDMs for one DQN
# =========================================================
def compute_rdms(activation_file):
    data = np.load(activation_file)
    layer_names = list(data.files)

    rdm_objects = []

    for key in layer_names:
        activations = data[key]
        n_frames = activations.shape[0]

        dataset = Dataset(
            activations,
            obs_descriptors={"frames": np.arange(n_frames)},
            channel_descriptors={"units": np.arange(activations.shape[1])}
        )

        rdm = calc_rdm(dataset, method="correlation")
        rdm_objects.append(rdm)

    return concat(rdm_objects), layer_names


# =========================================================
# STEP 1: LOAD ALL DQNs
# =========================================================
all_rdms = {}
layer_names_ref = None

for name, path in ACTIVATION_FILES.items():
    print(f"Processing {name}")
    rdms, layer_names = compute_rdms(path)
    all_rdms[name] = rdms

    # Save layer names once (assumes same architecture)
    if layer_names_ref is None:
        layer_names_ref = layer_names


# =========================================================
# STEP 2: CROSS-DQN RSA
# =========================================================
dqn_names = list(ACTIVATION_FILES.keys())

for i in range(len(dqn_names)):
    for j in range(i + 1, len(dqn_names)):

        dqn_a = dqn_names[i]
        dqn_b = dqn_names[j]

        print(f"\nComparing {dqn_a} vs {dqn_b}")

        rsa_matrix = compare(
            all_rdms[dqn_a],
            all_rdms[dqn_b],
            method="spearman"
        )

        # -------------------------------
        # SAVE MATRIX
        # -------------------------------
        save_path = os.path.join(
            SAVE_FOLDER,
            f"{dqn_a}_vs_{dqn_b}_RSA.npy"
        )
        np.save(save_path, rsa_matrix)
        print("Saved:", save_path)

        # -------------------------------
        # PLOT HEATMAP
        # -------------------------------
        plt.figure(figsize=(6, 6))
        im = plt.imshow(rsa_matrix, cmap="viridis")
        plt.colorbar(im)

        plt.xticks(range(len(layer_names_ref)), layer_names_ref, rotation=45)
        plt.yticks(range(len(layer_names_ref)), layer_names_ref)

        plt.xlabel(dqn_b)
        plt.ylabel(dqn_a)
        plt.title(f"Cross-RSA: {dqn_a} vs {dqn_b}")

        plt.tight_layout()

        png_path = os.path.join(
            SAVE_FOLDER,
            f"{dqn_a}_vs_{dqn_b}_RSA.png"
        )
        plt.savefig(png_path, dpi=300)
        plt.close()

        print("Saved heatmap:", png_path)


# Results are not simmetrics since we are not comparing same thinks:
# DQN2 L1  DQN2 L2
# DQN1 L1	A	B
# DQN1 L2	C	D

# Now:

# B = sim(DQN1 L1, DQN2 L2)

# C = sim(DQN1 L2, DQN2 L1)

# These are NOT the same comparison 