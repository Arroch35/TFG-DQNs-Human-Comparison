"""
4_compute_pixel_qvalue_rdms.py
Compute RDMs for pixel state, PCA state, Q-values, action, and state value
for each game × seed.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm

from src.config import GAMES, SEEDS, get_path

# =========================================================
# CONFIG
# =========================================================
# Only a subset of seeds was used in the original — keep that
# flexibility by letting you pass a custom list, defaulting to all.
ACTIVE_SEEDS = ["seed_0", "seed_2"]   # change or use SEEDS for all

# Suggested addition to config.py PATHS:
#   "dqn_selected15": DATA / "dqn_state_action_qvalue" / "{seed}" / "selected_subset_15" / "{game}",
from src.config import DATA
def get_dqn_selected15(seed, game):
    return DATA / "dqn_state_action_qvalue" / seed / "selected_subset_15" / game

# =========================================================
# HELPERS
# =========================================================
def save_rdm(rdm, save_base):
    matrix = rdm.get_matrices()[0]
    np.save(str(save_base) + ".npy", matrix)

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix)
    plt.colorbar()
    plt.title(save_base.name)
    plt.tight_layout()
    plt.savefig(str(save_base) + ".png", dpi=300)
    plt.close()

# =========================================================
# MAIN
# =========================================================
for seed in ACTIVE_SEEDS:
    print(f"\n{'='*60}\nSeed: {seed}\n{'='*60}")

    for game in GAMES:
        print(f"\n{'='*60}\n{game}\n{'='*60}")

        input_folder = get_dqn_selected15(seed, game)

        files = sorted(
            f for f in os.listdir(input_folder)
            if f.endswith(".npz") and f != "rdms.npz"
        )

        states, states_pca, q_values, actions, values = [], [], [], [], []

        for file in files:
            data = np.load(input_folder / file)
            states.append(data["state"])
            states_pca.append(data["state_pca"])
            q_values.append(data["q_values"])
            actions.append([int(data["action"])])
            values.append([float(data["value"])])

        states     = np.asarray(states)
        states_pca = np.asarray(states_pca)
        q_values   = np.asarray(q_values)
        actions    = np.asarray(actions)
        values     = np.asarray(values)

        print(f"states: {states.shape}  states_pca: {states_pca.shape}  q_values: {q_values.shape}")

        # ── Compute RDMs ──────────────────────────────────
        rdm_state     = calc_rdm(Dataset(states),     method="correlation")
        rdm_state_pca = calc_rdm(Dataset(states_pca), method="correlation")
        rdm_qvalues   = calc_rdm(Dataset(q_values),   method="correlation")
        rdm_action    = calc_rdm(Dataset(actions),    method="euclidean")
        rdm_value     = calc_rdm(Dataset(values),     method="euclidean")

        # ── Save RDMs ─────────────────────────────────────
        rdms_folder = input_folder / "rdms"
        rdms_folder.mkdir(parents=True, exist_ok=True)

        save_rdm(rdm_state,     rdms_folder / "pixel_rdm")
        save_rdm(rdm_state_pca, rdms_folder / "pixel_pca_rdm")
        save_rdm(rdm_qvalues,   rdms_folder / "qvalue_rdm")
        save_rdm(rdm_action,    rdms_folder / "action_rdm")
        save_rdm(rdm_value,     rdms_folder / "state_value_rdm")

        print("Saved all RDMs.")

print("\nDone.")
