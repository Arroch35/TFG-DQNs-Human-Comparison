import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from rsatoolbox.rdm import RDMs, compare

# =========================================================
# CONFIG
# =========================================================

GAMES = ["pong", "pacman", "spaceinvaders"]

seeds = ["seed_42"] #["seed_0", "seed_42"]

specific_directory = "selected_subset_15" #"big_rdm_equal_size" #"selected_subset_15"

HCF_FILE=f"../data/test_16_rdms/{specific_directory}/pong/hcf/pong_hcf_rdm.npy"

HUMAN_RDM_FOLDER = "../data/triplets_results/final_experiment/cleaned_results/rdms_human_experiment_rsa"

 
pong_rdm = np.load(HCF_FILE)
 
# ==============================
# HELPERS
# =========================================================

def extract_layer_name(path, game):
    return (
        os.path.basename(path)
        .replace(f"{game}_", "")
        .replace("_RDM.npy", "")
    )

def to_rdm_object(matrix):
    return RDMs(
        dissimilarities=np.array([matrix[np.triu_indices_from(matrix, k=1)]]),
    )

def rsa_spearman(rdm_a, rdm_b):
    if rdm_a.shape != rdm_b.shape:
        return np.nan
    v_a = rdm_a[np.triu_indices_from(rdm_a, k=1)]
    v_b = rdm_b[np.triu_indices_from(rdm_b, k=1)]
    if np.std(v_a) == 0 or np.std(v_b) == 0:
        return np.nan
    return float(compare(to_rdm_object(rdm_a), to_rdm_object(rdm_b), method="spearman")[0, 0])


def clean_paper_name(name):
    # 1. Exact manual mapping for precise control
    mapping = {
        "conv1_correlation": "Conv 1",
        "conv2_correlation": "Conv 2",
        "conv3_correlation": "Conv 3",
        "fc_correlation": "FC",
        "pixel_rdm": "Pixel",
        "pixel_pca_rdm": "Pixel PCA",
        "HCF": "HCF",   # Keeping your existing keys safe
        "Human": "Human"
    }
    
    # Return the mapped name if it exists; otherwise, fallback to a clean default
    if name in mapping:
        return mapping[name]
    
    # Fallback cleanup just in case a new layer name appears
    return name.replace('_correlation', '').replace('_rdm', '').replace('_', ' ').title()

 
# ==============================
# MAIN
# =========================================================

for seed in seeds:

    print("\n" + "=" * 60)
    print(f"Processing seed: {seed}")
    print("=" * 60)

    INTERNAL_RDM_FOLDER = f"../data/dqn_state_action_qvalue/{seed}/{specific_directory}" #"../data/dqn_state_action_qvalue/{seed}/internal_rdms"
    DQN_LAYER_RDM_FOLDER = f"../data/test_16_rdms/{specific_directory}/{seed}"

    SAVE_FOLDER = f"../data/dqn_state_action_qvalue/RSA/{seed}/{specific_directory}"

    os.makedirs(SAVE_FOLDER, exist_ok=True)

    all_results = []
    
    for game in GAMES:
        print("\n" + "=" * 70)
        print(f"GAME: {game}")
        print("=" * 70)
    
        # ------------------------------
        # LOAD INTERNAL RDMs
        # ------------------------------
        internal_path = os.path.join(INTERNAL_RDM_FOLDER, game, "rdms")

        internal_rdms = {}
        expected_keys = [
            "pixel_rdm",
            "pixel_pca_rdm",
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
        # LOAD HUMAN RDM
        # -----------------------------------------------------
        human_rdm_path = os.path.join(HUMAN_RDM_FOLDER, f"{game}_rdm.npy")
        if not os.path.exists(human_rdm_path):
            raise FileNotFoundError(
                f"Human RDM not found for {game}: {human_rdm_path}"
            )
        human_rdm = np.load(human_rdm_path)
    
        # -----------------------------------------------------
        # ORDERING: DQN → INTERNAL → HCF (pong only) → HUMAN (last)
        # -----------------------------------------------------
        dqn_names = list(dqn_rdms.keys())
        internal_names = list(internal_rdms.keys())
    
        names = dqn_names + internal_names
        all_rdms = {**dqn_rdms, **internal_rdms}
    
        if game == "pong":
            names = names + ["HCF"]
            all_rdms["HCF"] = pong_rdm
    
        # Human always goes last
        names = names + ["Human"]
        all_rdms["Human"] = human_rdm
    
        cleaned_names = [clean_paper_name(name) for name in names]
        print("Total RDMs:", len(names))
    
        # -----------------------------------------------------
        # RSA MATRIX
        # -----------------------------------------------------
        rsa_matrix = np.zeros((len(names), len(names)))

        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):

                if i == j:
                    rsa_matrix[i, j] = 1.0
                    continue

                if j > i:
                    continue  # fill lower triangle only, mirror after

                rsa_matrix[i, j] = rsa_spearman(all_rdms[name_i], all_rdms[name_j])

        # mirror lower triangle to upper and set diagonal
        rsa_matrix = rsa_matrix + rsa_matrix.T
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

        plot_matrix = rsa_matrix.copy().astype(float)
        upper_mask = np.triu(np.ones_like(plot_matrix, dtype=bool), k=0)
        plot_matrix[upper_mask] = np.nan

        cmap = cm.viridis.copy()
        cmap.set_bad(color="white")

        im = ax.imshow(
            plot_matrix,
            cmap=cmap,
            vmin=-0.1,
            vmax=1
        )
        plt.colorbar(im, label="RSA (Spearman)")

        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(cleaned_names, rotation=45, ha="right")
        ax.set_yticklabels(cleaned_names)

        for i in range(len(names)):
            for j in range(i):
                val = rsa_matrix[i, j]
                if not np.isnan(val):
                    ax.text(
                        j, i,
                        f"{val:.3f}",
                        ha="center",
                        va="center",
                        fontsize=12,
                        color="white" if val > 0.5 else "black"
                    )
    
        ax.set_title(f"{game}: Full RSA (DQN → Internal RDMs → Human)")
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
    df = pd.DataFrame(all_results).round(4)
    df.to_csv(
        os.path.join(SAVE_FOLDER, "full_rdm_rsa_summary.csv"),
        index=False
    )
    print("\nDone. Full RSA pipeline complete.")