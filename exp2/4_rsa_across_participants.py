import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from rsatoolbox.rdm import RDMs

# =========================
# CONFIGURATION
# =========================
games = ["pacman", "pong", "spaceinvaders"]
rdm_dir = "../data/triplets_results/exp2/cleaned_results/rdms_human_experiment_rsa"
output_dir = "../data/triplets_results/exp2/cleaned_results/inter_individual_rsa"
os.makedirs(output_dir, exist_ok=True)

# =========================
# COMPUTE INTER-INDIVIDUAL RSA
# =========================
for game in games:
    # Load all participant RDMs for this game
    rdm_files = sorted([
        f for f in os.listdir(rdm_dir)
        if f.startswith(f"{game}_participant_") and f.endswith("_rdm.npy")
    ])

    if len(rdm_files) < 2:
        print(f"Not enough participant RDMs for {game}, skipping.")
        continue

    participant_ids = [f.split("_participant_")[1].replace("_rdm.npy", "") for f in rdm_files]
    rdm_matrices = [np.load(os.path.join(rdm_dir, f)) for f in rdm_files]
    n_clips = rdm_matrices[0].shape[0]

    print(f"\n--- {game} | {len(rdm_matrices)} participants ---")

    # Stack into rsatoolbox RDMs object
    rdm_array = np.stack(rdm_matrices, axis=0)  # shape: (n_participants, n_clips, n_clips)
    # Extract upper triangular for each participant
    triu_idx = np.triu_indices(n_clips, k=1)
    rdm_vectors = np.stack([m[triu_idx] for m in rdm_matrices], axis=0)  # shape: (n_participants, n_pairs)

    rdms_obj = RDMs(
        dissimilarities=rdm_vectors,
        rdm_descriptors={"participant": participant_ids}
    )

    # Compute pairwise RSA (Spearman) between all participant pairs
    pairs = list(combinations(range(len(participant_ids)), 2))
    rsa_scores = []
    pair_labels = []

    for i, j in pairs:
        vec_i = rdm_vectors[i]
        vec_j = rdm_vectors[j]
        from scipy.stats import spearmanr
        rho, pval = spearmanr(vec_i, vec_j)
        rsa_scores.append(rho)
        pair_labels.append((participant_ids[i], participant_ids[j]))
        print(f"  {participant_ids[i]} vs {participant_ids[j]}: rho = {rho:.3f}, p = {pval:.3f}")

    rsa_scores = np.array(rsa_scores)
    print(f"\n  Mean pairwise RSA: {rsa_scores.mean():.3f} ± {rsa_scores.std():.3f}")
    print(f"  Min: {rsa_scores.min():.3f} | Max: {rsa_scores.max():.3f}")

    # Save pairwise RSA scores
    results_df = pd.DataFrame({
        "participant_i": [p[0] for p in pair_labels],
        "participant_j": [p[1] for p in pair_labels],
        "rsa_spearman": rsa_scores
    })
    csv_file = os.path.join(output_dir, f"{game}_inter_individual_rsa.csv")
    results_df.to_csv(csv_file, index=False)
    print(f"  Saved pairwise RSA to {csv_file}")

    # Build symmetric RSA matrix for heatmap
    n = len(participant_ids)
    rsa_matrix = np.zeros((n, n))
    for idx, (i, j) in enumerate(pairs):
        rsa_matrix[i, j] = rsa_scores[idx]
        rsa_matrix[j, i] = rsa_scores[idx]
    np.fill_diagonal(rsa_matrix, 1.0)

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        rsa_matrix,
        xticklabels=participant_ids,
        yticklabels=participant_ids,
        cmap="coolwarm",
        vmin=-1, vmax=1,
        annot=True, fmt=".2f"
    )
    plt.title(f"{game} — Inter-individual RSA (Spearman)")
    plt.tight_layout()
    png_file = os.path.join(output_dir, f"{game}_inter_individual_rsa_heatmap.png")
    plt.savefig(png_file, dpi=300)
    plt.close()
    print(f"  Saved heatmap: {png_file}")