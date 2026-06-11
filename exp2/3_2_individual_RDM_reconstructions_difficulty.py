import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cy_tste  # your t-STE library

from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm

#? Run this after 2_2_generate_achored_constrains.py and 2_1_create_triplet_csv.py to train t-STE on the human triplet constraints and compute RDMs for each game. The RDMs will be saved as .npy files in the specified directory. You can also visualize the RDMs as heatmaps.

# =========================
# CONFIGURATION
# =========================
games = ["pacman", "pong", "spaceinvaders"]
triplets_dir = "../data/triplets_results/exp2/cleaned_results" #"../data/cleaned_results"
rdm_dir = "../data/triplets_results/exp2/cleaned_results/rdms_human_experiment_rsa/difficulties" #"../data/rdms_human_experiment_rsa"
lopo_summary_file = "../data/tste_cv_results/all_games_lopo_summary.csv"
max_iter = 1000
TICK_STEP = 50  # For heatmap ticks

os.makedirs(rdm_dir, exist_ok=True)

# =========================
# 1) SELECT BEST DIMENSION
# =========================
# summary_df = pd.read_csv(lopo_summary_file)

# # Choose dimension with highest mean_test_accuracy, break ties with lowest std_test_accuracy
# best_dims = (
#     summary_df.groupby("dimension")
#     .agg(mean_test=("mean_test_accuracy", "mean"),
#          std_test=("std_test_accuracy", "mean"))
#     .sort_values(["mean_test", "std_test"], ascending=[False, True])
# )
# best_dimension = best_dims.index[0]
best_dimension=2
print(f"Selected best dimension for all games: {best_dimension}")

# =========================
# 2) TRAIN t-STE AND COMPUTE RDMs USING rsatoolbox
# =========================
for game in games:
    input_file = os.path.join(
        triplets_dir,
        f"{game}_triplets_constraints_with_difficulty.csv"
    )

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Skipping {game}.")
        continue

    df = pd.read_csv(input_file)

    triplet_cols = ["reference", "near", "far"]
    n_clips = np.max(df[triplet_cols].values) + 1

    for participant_id, participant_df in df.groupby("participant_id"):

        for difficulty, diff_df in participant_df.groupby("difficulty"):

            triplets = np.ascontiguousarray(
                diff_df[triplet_cols].values,
                dtype=np.int32
            )

            if len(triplets) < 10:
                print(
                    f"Skipping {game} | participant {participant_id} | "
                    f"{difficulty}: too few triplets ({len(triplets)})"
                )
                continue

            print(
                f"\nTraining t-STE for {game} | participant {participant_id} "
                f"| difficulty={difficulty} "
                f"({len(triplets)} triplets)"
            )

            X = cy_tste.tste(
                triplets,
                no_dims=best_dimension,
                max_iter=max_iter,
                verbose=True,
                use_log=True
            )

            dataset = Dataset(
                X,
                obs_descriptors={"clips": np.arange(n_clips)},
                channel_descriptors={"dims": np.arange(best_dimension)}
            )

            rdm_obj = calc_rdm(dataset, method="euclidean")
            rdm_matrix = rdm_obj.get_matrices()[0]

            # Save RDM
            rdm_file = os.path.join(
                rdm_dir,
                f"{game}_participant_{participant_id}_{difficulty}_rdm.npy"
            )
            np.save(rdm_file, rdm_matrix)

            print(
                f"Saved RDM for {game} | participant {participant_id} "
                f"| {difficulty}"
            )

            # Heatmap
            plt.figure(figsize=(8, 8))
            im = plt.imshow(
                rdm_matrix,
                cmap="coolwarm",
                origin="upper"
            )
            plt.colorbar(im)

            plt.title(
                f"{game} | participant {participant_id} "
                f"| {difficulty} RDM"
            )

            ticks = np.arange(
                0,
                n_clips,
                max(1, n_clips // TICK_STEP)
            )

            plt.xticks(ticks)
            plt.yticks(ticks)

            plt.tight_layout()

            png_file = os.path.join(
                rdm_dir,
                f"{game}_participant_{participant_id}_{difficulty}_rdm_heatmap.png"
            )

            plt.savefig(png_file, dpi=300)
            plt.close()
        print(f"Saved heatmap: {png_file}")