import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# =========================================================
# CONFIG
# =========================================================

GAMES = ["pong", "pacman", "spaceinvaders"]

SEED = "seed_42"

INTERNAL_RDM_FOLDER = f"../data/dqn_state_action_qvalue/{SEED}/big_rdm_equal_size"
DQN_LAYER_RDM_FOLDER = f"../data/test_16_rdms/big_rdm_equal_size/{SEED}"

HCF_FILE="../data/test_16_rdms/big_rdm_equal_size/pong/hcf/pong_hcf_rdm.npy"

SAVE_FOLDER = f"../data/dqn_state_action_qvalue/RSA/{SEED}/big_rdm_equal_size"

os.makedirs(SAVE_FOLDER, exist_ok=True)

pong_rdm = np.load(HCF_FILE)

# =========================================================
# HELPERS
# =========================================================

def upper_tri_vector(rdm):
    idx = np.triu_indices_from(rdm, k=1)
    return rdm[idx]

def extract_layer_name(path, game):
    return (
        os.path.basename(path)
        .replace(f"{game}_", "")
        .replace("_RDM.npy", "")
    )

# =========================================================
# MAIN
# =========================================================

all_results = []

for game in GAMES:

    print("\n" + "=" * 70)
    print(f"GAME: {game}")
    print("=" * 70)

    # -----------------------------------------------------
    # LOAD INTERNAL RDMs
    # -----------------------------------------------------

    internal_path = os.path.join(INTERNAL_RDM_FOLDER, game, "rdms")

    internal_rdms = {}

    expected_keys = [
        "pixel_rdm",
        "pixel_pca_rdm",
        "qvalue_rdm",
        "state_value_rdm"
    ]

    for key in expected_keys:
        file_path = os.path.join(internal_path, f"{key}.npy")

        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        internal_rdms[key] = np.load(file_path)

    # -----------------------------------------------------
    # LOAD DQN LAYER RDMs
    # -----------------------------------------------------

    dqn_folder = os.path.join(DQN_LAYER_RDM_FOLDER, game)
    dqn_files = sorted(glob.glob(os.path.join(dqn_folder, "*RDM.npy")))

    dqn_rdms = {}

    for f in dqn_files:
        layer = extract_layer_name(f, game)
        dqn_rdms[layer] = np.load(f)

    # -----------------------------------------------------
    # ORDERING: DQN FIRST, THEN INTERNAL
    # -----------------------------------------------------

    dqn_names = list(dqn_rdms.keys())
    internal_names = list(internal_rdms.keys())

    names = dqn_names + internal_names

    if game == "pong":
        names = names + ["HCF"]

    all_rdms = {**dqn_rdms, **internal_rdms}

    if game == "pong":
        all_rdms["HCF"] = pong_rdm

    print("Total RDMs:", len(names))

    # -----------------------------------------------------
    # RSA MATRIX
    # -----------------------------------------------------

    rsa_matrix = np.zeros((len(names), len(names)))

    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):

            rdm_i = all_rdms[name_i]
            rdm_j = all_rdms[name_j]

            if rdm_i.shape != rdm_j.shape:
                rsa_matrix[i, j] = np.nan
                continue

            v_i = upper_tri_vector(rdm_i)
            v_j = upper_tri_vector(rdm_j)

            # safe correlation
            if np.std(v_i) == 0 or np.std(v_j) == 0:
                rsa_matrix[i, j] = np.nan
                continue

            rsa_matrix[i, j] = np.corrcoef(v_i, v_j)[0, 1]

    # -----------------------------------------------------
    # SYMMETRY + DIAGONAL
    # -----------------------------------------------------

    rsa_matrix = (rsa_matrix + rsa_matrix.T) / 2
    np.fill_diagonal(rsa_matrix, 1.0)

    # -----------------------------------------------------
    # SAVE NUMPY
    # -----------------------------------------------------

    np.save(
        os.path.join(SAVE_FOLDER, f"{game}_FULL_RDM_RSA.npy"),
        rsa_matrix
    )

    # -----------------------------------------------------
    # PLOT (LOWER TRIANGLE ONLY + VALUES)
    # -----------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(max(8, len(names) * 0.6),
                 max(6, len(names) * 0.6))
    )

    # mask upper triangle
    mask = np.triu(np.ones_like(rsa_matrix, dtype=bool), k=0)
    masked = np.ma.array(rsa_matrix, mask=mask)

    cmap = cm.viridis.copy()
    cmap.set_bad(color="white")

    im = ax.imshow(
        masked,
        cmap=cmap,
        vmin=-0.1,
        vmax=1
    )

    plt.colorbar(im, label="RSA (correlation)")

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)

    # add values only in upper triangle
    for i in range(len(names)):
        for j in range(len(names)):

            if i > j:  # lower triangle only
                val = rsa_matrix[i, j]

                if not np.isnan(val):
                    ax.text(
                        j, i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="white" if val > 0.5 else "black"
                    )

    ax.set_title(f"{game}: Full RSA (DQN → Internal RDMs)")
    plt.tight_layout()

    plt.savefig(
        os.path.join(SAVE_FOLDER, f"{game}_FULL_RDM_RSA.png"),
        dpi=300
    )

    plt.close()

    # -----------------------------------------------------
    # STORE LONG FORMAT
    # -----------------------------------------------------

    for i, ni in enumerate(names):
        for j, nj in enumerate(names):

            all_results.append({
                "game": game,
                "rdm_1": ni,
                "rdm_2": nj,
                "rsa": rsa_matrix[i, j]
            })

    print(f"Saved RSA for {game}")

# =========================================================
# SAVE CSV
# =========================================================

df = pd.DataFrame(all_results)

df.to_csv(
    os.path.join(SAVE_FOLDER, "full_rdm_rsa_summary.csv"),
    index=False
)

print("\nDone. Full RSA pipeline complete.")