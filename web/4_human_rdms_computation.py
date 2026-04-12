import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cy_tste  # your t-STE library

from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm

# =========================
# CONFIGURATION
# =========================
games = ["pacman", "pong", "spaceinvaders"]
triplets_dir = "../data/cleaned_results"
rdm_dir = "../data/rdms_human_experiment_rsa"
lopo_summary_file = "../data/tste_cv_results/all_games_lopo_summary.csv"
max_iter = 1000
TICK_STEP = 50  # For heatmap ticks

os.makedirs(rdm_dir, exist_ok=True)

# =========================
# 1) SELECT BEST DIMENSION
# =========================
summary_df = pd.read_csv(lopo_summary_file)

# Choose dimension with highest mean_test_accuracy, break ties with lowest std_test_accuracy
best_dims = (
    summary_df.groupby("dimension")
    .agg(mean_test=("mean_test_accuracy", "mean"),
         std_test=("std_test_accuracy", "mean"))
    .sort_values(["mean_test", "std_test"], ascending=[False, True])
)
best_dimension = best_dims.index[0]
print(f"Selected best dimension for all games: {best_dimension}")

# =========================
# 2) TRAIN t-STE AND COMPUTE RDMs USING rsatoolbox
# =========================
for game in games:
    input_file = os.path.join(triplets_dir, f"{game}_tste_constraints.csv")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Skipping {game}.")
        continue

    df = pd.read_csv(input_file)
    triplet_cols = ["reference", "near", "far"]
    triplets = np.ascontiguousarray(df[triplet_cols].values, dtype=np.int32)
    n_clips = np.max(triplets) + 1
    
    print(f"\nTraining t-STE for {game} ({n_clips} clips, {len(triplets)} triplets)")

    # -------------------------------
    # Train t-STE embedding
    # -------------------------------
    X = cy_tste.tste(
        triplets,
        no_dims=best_dimension,
        max_iter=max_iter,
        verbose=True,
        use_log=True
    )

    # -------------------------------
    # Wrap t-STE embeddings into Dataset
    # -------------------------------
    dataset = Dataset(
        X,
        obs_descriptors={"clips": np.arange(n_clips)},
        channel_descriptors={"dims": np.arange(best_dimension)}
    )

    print("Embeddings X:\n", X)
    print("Min/max per dimension:", X.min(axis=0), X.max(axis=0))

    #!ESTO NO ES EUCLIDIANT DISTANCE, ASÍ QUE TENDRÉ MIRAR MEJOR SI ESTO ES LO MEJOR O ES MEJOR LA EUCLIDIAN
    # -------------------------------
    # Compute correlation-distance RDM
    # -------------------------------
    rdm_obj = calc_rdm(dataset, method="euclidean") # method="correlation"
    rdm_matrix = rdm_obj.get_matrices()[0]

    # -------------------------------
    # Save RDM as .npy
    # -------------------------------
    rdm_file = os.path.join(rdm_dir, f"{game}_rdm.npy")
    np.save(rdm_file, rdm_matrix)
    print(f"Saved RDM for {game} to {rdm_file}")

    # -------------------------------
    # Optional: Plot heatmap
    # -------------------------------
    plt.figure(figsize=(8, 8))
    im = plt.imshow(rdm_matrix, cmap="coolwarm", origin="upper")
    plt.colorbar(im)
    plt.title(f"{game} RDM (t-STE)")
    plt.xlabel("Clip")
    plt.ylabel("Clip")
    ticks = np.arange(0, n_clips, max(1, n_clips // TICK_STEP))
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.tight_layout()
    png_file = os.path.join(rdm_dir, f"{game}_rdm_heatmap.png")
    plt.savefig(png_file, dpi=300)
    plt.close()
    print(f"Saved heatmap: {png_file}")