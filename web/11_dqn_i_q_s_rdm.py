import os
import numpy as np
import matplotlib.pyplot as plt
import rsatoolbox
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm

# =========================================================
# CONFIG
# =========================================================

GAMES = ["pong", "pacman", "spaceinvaders"]

seeds = ["seed_0", "seed_2"]


# =========================================================
# HELPERS
# =========================================================

def save_rdm(rdm, save_base):

    matrix = rdm.get_matrices()[0]

    np.save(save_base + ".npy", matrix)

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix)
    plt.colorbar()
    plt.title(os.path.basename(save_base))
    plt.tight_layout()
    plt.savefig(save_base + ".png", dpi=300)
    plt.close()

# =========================================================
# MAIN
# =========================================================

for seed in seeds:

    print("\n" + "=" * 60)
    print(f"Processing seed: {seed}")
    print("=" * 60) 

    BASE_FOLDER = f"../data/dqn_state_action_qvalue/{seed}/selected_subset_15" #"../data/dqn_state_action_qvalue/{seed}/big_rdm_equal_size" #"../data/dqn_state_action_qvalue/{seed}/internal_rdms"

    for game in GAMES:

        print("\n" + "=" * 60)
        print(game)
        print("=" * 60)

        input_folder = os.path.join(BASE_FOLDER, game)

        files = sorted([
            f for f in os.listdir(input_folder)
            if f.endswith(".npz")
            and f != "rdms.npz"
        ])

        states = []
        states_pca = []
        q_values = []
        actions = []
        values = []

        for file in files:

            data = np.load(os.path.join(input_folder, file))

            states.append(data["state"])
            states_pca.append(data["state_pca"])
            q_values.append(data["q_values"])
            actions.append([int(data["action"])])
            values.append([float(data["value"])])

        states = np.asarray(states)
        states_pca = np.asarray(states_pca)
        q_values = np.asarray(q_values)
        actions = np.asarray(actions)
        values = np.asarray(values)

        print("states:", states.shape)
        print("states_pca:", states_pca.shape)
        print("q_values:", q_values.shape)

        # =====================================================
        # CREATE DATASETS
        # =====================================================

        ds_state = Dataset(states)
        ds_state_pca = Dataset(states_pca)
        ds_qvalues = Dataset(q_values)
        ds_action = Dataset(actions)
        ds_value = Dataset(values)

        # =====================================================
        # COMPUTE RDMS
        # =====================================================

        print("Computing state RDM...")
        rdm_state = calc_rdm(ds_state, method='correlation')

        print("Computing PCA RDM...")
        rdm_state_pca = calc_rdm(ds_state_pca, method='correlation')

        print("Computing Q-value RDM...")
        rdm_qvalues = calc_rdm(ds_qvalues, method='correlation')

        print("Computing action RDM...")
        rdm_action = calc_rdm(ds_action, method='euclidean')

        print("Computing value RDM...")
        rdm_value = calc_rdm(ds_value, method='euclidean')

        # =====================================================
        # SAVE
        # =====================================================

        rdms_folder = os.path.join(input_folder, "rdms")

        os.makedirs(rdms_folder, exist_ok=True)

        save_rdm(
            rdm_state,
            os.path.join(rdms_folder, "pixel_rdm")
        )

        save_rdm(
            rdm_state_pca,
            os.path.join(rdms_folder, "pixel_pca_rdm")
        )

        save_rdm(
            rdm_qvalues,
            os.path.join(rdms_folder, "qvalue_rdm")
        )

        save_rdm(
            rdm_action,
            os.path.join(rdms_folder, "action_rdm")
        )

        save_rdm(
            rdm_value,
            os.path.join(rdms_folder, "state_value_rdm")
        )

        print("Saved all RDMs.")

    print("\nDone.")