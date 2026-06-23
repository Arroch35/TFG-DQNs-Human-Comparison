import numpy as np
import os
import rsatoolbox
from scipy.stats import spearmanr
from src.utils import upper_tri

def compare_second_order_rsa(folder1, folder2, filename):
    # 1. Load the RSA matrices
    path1 = os.path.join(folder1, filename)
    path2 = os.path.join(folder2, filename)
    
    # These are likely (n_layers, n_layers) matrices
    rsa_mat1 = np.load(path1)
    rsa_mat2 = np.load(path2)
    
    if rsa_mat1.shape != rsa_mat2.shape:
        raise ValueError(f"Matrix mismatch: {rsa_mat1.shape} vs {rsa_mat2.shape}. "
                         "Do both folders have the same number of layers?")

    # 2. Wrap into RDMs objects
    # Note: rsatoolbox RDMs expects dissimilarities. 
    # If your saved matrices are similarities (Spearman rho), 
    # we convert them to dissimilarity (1 - rho) for the toolbox.
    vec1 = upper_tri(1 - rsa_mat1)
    vec2 = upper_tri(1 - rsa_mat2)

    corr, _ = spearmanr(vec1, vec2)
    
    # 3. Compare the two hierarchical structures
    # We use 'spearman' to see if the RELATIVE similarity between layers 
    # is preserved across your two versions.
    #overall_similarity = rsatoolbox.rdm.compare(rdm1, rdm2, method='spearman')
    
    print(f"--- Third-Order RSA Comparison ---")
    print(f"Path 1: {path1}")
    print(f"Path 2: {path2}")
    print(f"File: {filename}")
    print(f"Similarity between RSA structures: {corr}")

# --- Configuration ---
methods=["correlation"] #["correlation", "euclidean"]
games=["pong","pacman","spaceinvaders"]
SEED="seed_42"
base_folder_a = "../data/test_16_rdms/pilot/"
base_folder_b = "../data/test_16_rdms/big_rdm_equal_size/"



for game in games:
    for method in methods:
        folder_experiment=os.path.join(base_folder_a, SEED, game)
        folder_big_rdms=os.path.join(base_folder_b, SEED, game)

        target_file = f"{game}_DQN_layer_RSA_{method}_matrix.npy" # The name shared by both files

        compare_second_order_rsa(folder_experiment, folder_big_rdms, target_file)
