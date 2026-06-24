"""
exp15_full_rsa_matrix.py
Compute a full RSA matrix across DQN layers, pixel/PCA RDMs,
HCF (pong only), and the human RDM for each game × seed.
"""
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from rsatoolbox.rdm import RDMs, compare

from src.config import GAMES, SEEDS, REFERENCE_SEED, REPR, get_path, ensure
from src.utils import extract_layer_name

# =========================================================
# CONFIG
# =========================================================
# "selected_subset_15" is the directory variant used here.
# The config keys rdms_selected15 and full_rsa already encode this.
SPECIFIC_DIR = "selected_subset_15"

# Seeds to process (original only used seed_42)
ACTIVE_SEEDS = [REFERENCE_SEED]   # extend to SEEDS for all

# Suggested addition to config.py PATHS:
#   "dqn_rdms_game": DATA / "dqn_state_action_qvalue" / "{seed}" / "{variant}" / "{game}" / "rdms",
from src.config import DATA
def get_dqn_rdms_folder(seed, game):
    return DATA / "dqn_state_action_qvalue" / seed / SPECIFIC_DIR / game / "rdms"

# =========================================================
# HELPERS
# =========================================================
def to_rdm_object(matrix):
    return RDMs(dissimilarities=np.array([matrix[np.triu_indices_from(matrix, k=1)]]))


def rsa_spearman(rdm_a, rdm_b):
    if rdm_a.shape != rdm_b.shape:
        return np.nan
    v_a = rdm_a[np.triu_indices_from(rdm_a, k=1)]
    v_b = rdm_b[np.triu_indices_from(rdm_b, k=1)]
    if np.std(v_a) == 0 or np.std(v_b) == 0:
        return np.nan
    return float(compare(to_rdm_object(rdm_a), to_rdm_object(rdm_b), method="spearman")[0, 0])


CLEAN_NAME_MAP = {
    "conv1_correlation": "Conv 1",
    "conv2_correlation": "Conv 2",
    "conv3_correlation": "Conv 3",
    "fc_correlation":    "FC",
    "pixel_rdm":         "Pixel",
    "pixel_pca_rdm":     "Pixel PCA",
    "HCF":               "HCF",
    "Human":             "Human",
}

def clean_paper_name(name):
    if name in CLEAN_NAME_MAP:
        return CLEAN_NAME_MAP[name]
    return name.replace("_correlation", "").replace("_rdm", "").replace("_", " ").title()


# =========================================================
# MAIN
# =========================================================
# Load pong HCF RDM once (path already in config)
pong_hcf_rdm = np.load(get_path("rdms_hcf"))

for seed in ACTIVE_SEEDS:
    print(f"\n{'='*60}\nSeed: {seed}\n{'='*60}")

    save_folder = ensure("results_full_rsa", seed=seed)   # data/dqn_state_action_qvalue/RSA/{seed}/selected_subset_15
    all_results = []

    for game in GAMES:
        print(f"\n{'='*70}\nGAME: {game}\n{'='*70}")

        # ── Internal RDMs (pixel, pixel_pca) ──────────────
        internal_folder = get_dqn_rdms_folder(seed, game)
        internal_rdms   = {}
        for key in ["pixel_rdm", "pixel_pca_rdm"]:
            path = internal_folder / f"{key}.npy"
            if not path.exists():
                raise FileNotFoundError(str(path))
            internal_rdms[key] = np.load(path)

        # ── DQN layer RDMs ────────────────────────────────
        dqn_folder = get_path("rdms_selected15", seed=seed, game=game)
        dqn_files  = sorted(dqn_folder.glob("*RDM.npy"))
        dqn_rdms   = {extract_layer_name(str(f), game): np.load(f) for f in dqn_files}

        # ── Human RDM ─────────────────────────────────────
        human_rdm_path = get_path("rdms_human_game", game=game)
        if not human_rdm_path.exists():
            raise FileNotFoundError(f"Human RDM not found: {human_rdm_path}")
        human_rdm = np.load(human_rdm_path)

        # ── Assemble ordered dict: DQN → internal → HCF → human ──
        names    = list(dqn_rdms.keys()) + list(internal_rdms.keys())
        all_rdms = {**dqn_rdms, **internal_rdms}

        if game == "pong":
            names.append("HCF")
            all_rdms["HCF"] = pong_hcf_rdm

        names.append("Human")
        all_rdms["Human"] = human_rdm

        cleaned_names = [clean_paper_name(n) for n in names]
        print(f"Total RDMs: {len(names)}")

        # ── RSA matrix ────────────────────────────────────
        n_rdms     = len(names)
        rsa_matrix = np.zeros((n_rdms, n_rdms))
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                if i == j:
                    rsa_matrix[i, j] = 1.0
                elif j < i:
                    rsa_matrix[i, j] = rsa_spearman(all_rdms[ni], all_rdms[nj])

        rsa_matrix = rsa_matrix + rsa_matrix.T
        np.fill_diagonal(rsa_matrix, 1.0)

        # ── Save .npy ─────────────────────────────────────
        np.save(save_folder / f"{game}_FULL_RDM_RSA.npy", rsa_matrix)

        # ── Plot (lower triangle only) ────────────────────
        fig, ax = plt.subplots(figsize=(max(8, n_rdms * 0.6), max(6, n_rdms * 0.6)))
        plot_matrix = rsa_matrix.copy().astype(float)
        plot_matrix[np.triu(np.ones_like(plot_matrix, dtype=bool), k=0)] = np.nan

        cmap = cm.viridis.copy()
        cmap.set_bad(color="white")
        im = ax.imshow(plot_matrix, cmap=cmap, vmin=-0.1, vmax=1)
        plt.colorbar(im, label="RSA (Spearman)")

        ax.set_xticks(range(n_rdms)); ax.set_xticklabels(cleaned_names, rotation=45, ha="right")
        ax.set_yticks(range(n_rdms)); ax.set_yticklabels(cleaned_names)

        for i in range(n_rdms):
            for j in range(i):
                val = rsa_matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=12,
                            color="white" if val > 0.5 else "black")

        ax.set_title(f"{game}: Full RSA (DQN → Internal RDMs → Human)")
        plt.tight_layout()
        plt.savefig(save_folder / f"{game}_FULL_RDM_RSA.png", dpi=300)
        plt.close()

        # ── Long-format rows ──────────────────────────────
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                all_results.append({"game": game, "rdm_1": ni, "rdm_2": nj, "rsa": rsa_matrix[i, j]})

        print(f"Saved RSA for {game}")

    # ── Combined CSV ──────────────────────────────────────
    pd.DataFrame(all_results).round(4).to_csv(save_folder / "full_rdm_rsa_summary.csv", index=False)

print("\nDone. Full RSA pipeline complete.")
