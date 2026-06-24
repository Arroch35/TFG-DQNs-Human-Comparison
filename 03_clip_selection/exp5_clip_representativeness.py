import numpy as np
import os
import rsatoolbox
from scipy.stats import spearmanr

from src.config import GAMES, REFERENCE_SEED, REPR, get_path
from src.utils import upper_tri

# =========================================================
# CONFIG
# =========================================================
SEED = REFERENCE_SEED                   # "seed_42"
METHOD = REPR["rdm_method"]             # "correlation"

# =========================================================
# COMPARISON FUNCTION
# =========================================================
def compare_second_order_rsa(folder1, folder2, filename):
    path1 = os.path.join(folder1, filename)
    path2 = os.path.join(folder2, filename)

    rsa_mat1 = np.load(path1)
    rsa_mat2 = np.load(path2)

    if rsa_mat1.shape != rsa_mat2.shape:
        raise ValueError(
            f"Matrix mismatch: {rsa_mat1.shape} vs {rsa_mat2.shape}. "
            "Do both folders have the same number of layers?"
        )

    # Convert similarities (Spearman rho) to dissimilarities for comparison
    vec1 = upper_tri(1 - rsa_mat1)
    vec2 = upper_tri(1 - rsa_mat2)

    corr, _ = spearmanr(vec1, vec2)

    print(f"--- Third-Order RSA Comparison ---")
    print(f"Path 1: {path1}")
    print(f"Path 2: {path2}")
    print(f"File: {filename}")
    print(f"Similarity between RSA structures: {corr}")


# =========================================================
# MAIN
# NOTE: "rdms_pilot" is not yet in config.py — suggested addition:
#   "rdms_pilot": DATA / "test_16_rdms" / "pilot" / "{seed}" / "{game}",
# Until then, it falls back to a manual path derived from DATA.
# =========================================================
for game in GAMES:
    folder_pilot   = get_path("rdms_pilot", seed=SEED, game=game)   # once added to config
    folder_big_rdm = get_path("rdms_big",   seed=SEED, game=game)

    target_file = f"{game}_DQN_layer_RSA_{METHOD}_matrix.npy"

    compare_second_order_rsa(
        str(folder_pilot),
        str(folder_big_rdm),
        target_file,
    )
