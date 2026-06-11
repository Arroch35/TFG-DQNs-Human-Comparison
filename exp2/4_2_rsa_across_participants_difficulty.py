import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations
from scipy.stats import spearmanr

# =========================
# CONFIGURATION
# =========================
games = ["pacman", "pong", "spaceinvaders"]
difficulties = ["easy", "medium", "hard"]

rdm_dir = "../data/triplets_results/exp2/cleaned_results/rdms_human_experiment_rsa/difficulties" #"../data/triplets_results/final_experiment/cleaned_results/rdms_human_experiment_rsa"
output_dir = "../data/triplets_results/exp2/cleaned_results/rdms_human_experiment_rsa/difficulties/inter_individual_rsa"

os.makedirs(output_dir, exist_ok=True)

# =========================
# COMPUTE INTER-INDIVIDUAL RSA
# =========================
for game in games:

    print(f"\n{'='*60}")
    print(game.upper())
    print(f"{'='*60}")

    for difficulty in difficulties:

        # ----------------------------------------------------
        # Load RDMs corresponding to this difficulty
        # ----------------------------------------------------
        rdm_files = sorted([
            f for f in os.listdir(rdm_dir)
            if (
                f.startswith(f"{game}_participant_")
                and f.endswith(f"_{difficulty}_triplets_rdm.npy")
            )
        ])
        print(len(os.listdir(rdm_dir)), "total RDM files in directory.")
        print(len(rdm_files), "RDM files found for difficulty:", difficulty)
        if len(rdm_files) < 2:
            print(
                f"{game} | {difficulty}: "
                f"not enough participant RDMs."
            )
            continue

        participant_ids = [
            f.split("_participant_")[1].replace(
                f"_{difficulty}_rdm.npy", ""
            )
            for f in rdm_files
        ]

        rdm_matrices = [
            np.load(os.path.join(rdm_dir, f))
            for f in rdm_files
        ]

        n_clips = rdm_matrices[0].shape[0]

        print(
            f"\n--- {game} | {difficulty} "
            f"| {len(participant_ids)} participants ---"
        )

        # ----------------------------------------------------
        # Vectorize RDMs
        # ----------------------------------------------------
        triu_idx = np.triu_indices(n_clips, k=1)

        rdm_vectors = np.stack(
            [m[triu_idx] for m in rdm_matrices],
            axis=0
        )

        # ----------------------------------------------------
        # Pairwise RSA
        # ----------------------------------------------------
        pairs = list(
            combinations(range(len(participant_ids)), 2)
        )

        rsa_scores = []
        pair_labels = []

        for i, j in pairs:

            rho, pval = spearmanr(
                rdm_vectors[i],
                rdm_vectors[j]
            )

            rsa_scores.append(rho)

            pair_labels.append(
                (
                    participant_ids[i],
                    participant_ids[j]
                )
            )

            print(
                f"  {participant_ids[i]} vs "
                f"{participant_ids[j]}: "
                f"rho={rho:.3f}, p={pval:.3f}"
            )

        rsa_scores = np.array(rsa_scores)

        print(
            f"\nMean RSA: "
            f"{rsa_scores.mean():.3f} ± "
            f"{rsa_scores.std():.3f}"
        )

        # ----------------------------------------------------
        # Save pairwise results
        # ----------------------------------------------------
        results_df = pd.DataFrame({
            "participant_i": [p[0] for p in pair_labels],
            "participant_j": [p[1] for p in pair_labels],
            "difficulty": difficulty,
            "rsa_spearman": rsa_scores
        })

        csv_file = os.path.join(
            output_dir,
            f"{game}_{difficulty}_inter_individual_rsa.csv"
        )

        results_df.to_csv(csv_file, index=False)

        print(f"Saved: {csv_file}")

        # ----------------------------------------------------
        # RSA matrix
        # ----------------------------------------------------
        n = len(participant_ids)

        rsa_matrix = np.zeros((n, n))

        for idx, (i, j) in enumerate(pairs):
            rsa_matrix[i, j] = rsa_scores[idx]
            rsa_matrix[j, i] = rsa_scores[idx]

        np.fill_diagonal(rsa_matrix, 1)

        # ----------------------------------------------------
        # Heatmap
        # ----------------------------------------------------
        plt.figure(figsize=(8, 6))

        sns.heatmap(
            rsa_matrix,
            xticklabels=participant_ids,
            yticklabels=participant_ids,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f"
        )

        plt.title(
            f"{game} | {difficulty} "
            f"Inter-individual RSA"
        )

        plt.tight_layout()

        png_file = os.path.join(
            output_dir,
            f"{game}_{difficulty}_inter_individual_rsa_heatmap.png"
        )

        plt.savefig(png_file, dpi=300)
        plt.close()

        print(f"Saved heatmap: {png_file}")