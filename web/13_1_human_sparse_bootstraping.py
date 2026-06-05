import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cy_tste
from scipy.stats import spearmanr
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm

# =========================
# CONFIGURATION
# =========================
games = ["pacman", "pong", "spaceinvaders"]
triplets_dir = "../data/triplets_results/final_experiment/cleaned_results"
model_rdm_dir = "../data/triplets_results/final_experiment/cleaned_results/rdms_human_experiment_rsa/"  # <-- CHANGE IF NEEDED
rdm_dir = "../data/triplets_results/final_experiment/cleaned_results/rdms_human_experiment_rsa/bootstraping"

os.makedirs(rdm_dir, exist_ok=True)

best_dimension = 2
max_iter = 1000
N_BOOTSTRAPS = 100

# =========================
# MAIN LOOP
# =========================
for game in games:

    input_file = os.path.join(triplets_dir, f"{game}_triplets_constraints.csv")
    model_file = os.path.join(model_rdm_dir, f"{game}_rdm.npy")

    if not os.path.exists(input_file):
        print(f"Missing {input_file}, skipping {game}")
        continue

    if not os.path.exists(model_file):
        print(f"Missing model RDM {model_file}, skipping {game}")
        continue

    df = pd.read_csv(input_file)

    triplets_full = np.ascontiguousarray(
        df[["reference", "near", "far"]].values,
        dtype=np.int32
    )

    n_clips = np.max(triplets_full) + 1

    print(f"\nGame: {game}")
    print(f"Clips: {n_clips}, Triplets: {len(triplets_full)}")
    print(f"Bootstraps: {N_BOOTSTRAPS}")

    # load model RDM
    model_rdm = np.load(model_file)

    # vectorize model once
    model_vec = model_rdm[np.triu_indices(n_clips, k=1)]

    # store RSA values
    rsa_values = np.zeros(N_BOOTSTRAPS)

    # =========================
    # BOOTSTRAP LOOP
    # =========================
    for b in range(N_BOOTSTRAPS):

        # 1. resample triplets
        idx = np.random.choice(
            len(triplets_full),
            size=len(triplets_full),
            replace=True
        )

        triplets_b = np.ascontiguousarray(triplets_full[idx], dtype=np.int32)

        # 2. t-STE
        X = cy_tste.tste(
            triplets_b,
            no_dims=best_dimension,
            max_iter=max_iter,
            verbose=False,
            use_log=True
        )

        # 3. RDM from t-STE embedding
        dataset = Dataset(
            X,
            obs_descriptors={"clips": np.arange(n_clips)},
            channel_descriptors={"dims": np.arange(best_dimension)}
        )

        rdm_b = calc_rdm(dataset, method="euclidean").get_matrices()[0]

        # 4. RSA
        human_vec = rdm_b[np.triu_indices(n_clips, k=1)]

        rsa_values[b] = spearmanr(model_vec, human_vec)[0]

        if (b + 1) % 10 == 0:
            print(f"  Completed bootstrap {b+1}/{N_BOOTSTRAPS}")

    # =========================
    # SUMMARY STATISTICS
    # =========================
    rsa_mean = np.mean(rsa_values)
    rsa_std = np.std(rsa_values)
    rsa_ci = np.percentile(rsa_values, [2.5, 97.5])

    print("\n===== RSA BOOTSTRAP RESULTS =====")
    print(f"Mean RSA: {rsa_mean:.4f}")
    print(f"Std RSA: {rsa_std:.4f}")
    print(f"95% CI: [{rsa_ci[0]:.4f}, {rsa_ci[1]:.4f}]")

    # =========================
    # SAVE RESULTS
    # =========================
    np.save(os.path.join(rdm_dir, f"{game}_rsa_bootstrap.npy"), rsa_values)

    # =========================
    # PLOT RSA DISTRIBUTION
    # =========================
    plt.figure(figsize=(7, 5))
    plt.hist(rsa_values, bins=20)
    plt.title(f"{game} RSA bootstrap distribution")
    plt.xlabel("RSA (Spearman)")
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig(os.path.join(rdm_dir, f"{game}_rsa_bootstrap_hist.png"), dpi=300)
    plt.close()

    print(f"Saved RSA bootstrap results for {game}")