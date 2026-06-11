import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cy_tste
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm, RDMs, compare

# =========================
# CONFIGURATION
# =========================
games = ["pacman", "pong", "spaceinvaders"]
triplets_dir = "../data/triplets_results/final_experiment/cleaned_results"
SEED = "seed_42"
DQN_RDM_BASE_FOLDER = f"../data/test_16_rdms/selected_subset_15/{SEED}"
REAL_RSA_CSV = f"../data/triplets_results/final_experiment/cleaned_results/test_16_RSA/optimized_RDMs/{SEED}/human_vs_DQN_RSA_summary.csv"
SAVE_FOLDER = f"../data/triplets_results/final_experiment/cleaned_results/test_16_RSA/permutation_baseline/{SEED}"
os.makedirs(SAVE_FOLDER, exist_ok=True)

best_dimension = 2
max_iter = 1000
n_permutations = 1000

# =========================
# HELPER: raw triplets -> constraints (double)
# =========================
def to_constraints(df):
    rows = []
    for _, row in df.iterrows():
        sim1 = int(row["similar_clip_1_idx"])
        sim2 = int(row["similar_clip_2_idx"])
        odd = int(row["odd_clip_idx"])
        rows.append({"reference": sim1, "near": sim2, "far": odd})
        rows.append({"reference": sim2, "near": sim1, "far": odd})
    return pd.DataFrame(rows)

# =========================
# HELPER: constraints -> RDM
# =========================
def compute_rdm_from_constraints(constraints_df, n_clips):
    triplets = np.ascontiguousarray(
        constraints_df[["reference", "near", "far"]].values, dtype=np.int32
    )
    X = cy_tste.tste(
        triplets,
        no_dims=best_dimension,
        max_iter=max_iter,
        verbose=False,
        use_log=True
    )
    dataset = Dataset(
        X,
        obs_descriptors={"clips": np.arange(n_clips)},
        channel_descriptors={"dims": np.arange(best_dimension)}
    )
    rdm_obj = calc_rdm(dataset, method="euclidean")
    return rdm_obj.get_matrices()[0]

# =========================
# HELPER: load RDM as rsatoolbox object
# =========================
def load_rdm_as_object(rdm_matrix, name="rdm"):
    return RDMs(
        dissimilarities=np.array([rdm_matrix]),
        rdm_descriptors={"name": [name]}
    )

# =========================
# LOAD REAL RSA SCORES
# =========================
real_rsa_df = pd.read_csv(REAL_RSA_CSV)
# Build lookup: (game, layer) -> real rsa score
real_rsa_lookup = {
    (row["game"], row["layer"]): row["rsa_spearman"]
    for _, row in real_rsa_df.iterrows()
}
print("Loaded real RSA scores:")
print(real_rsa_df)

# =========================
# MAIN
# =========================
all_results = []

for game in games:
    print("\n" + "=" * 60)
    print(f"GAME: {game}")
    print("=" * 60)

    # -------------------------
    # Load raw triplets
    # -------------------------
    raw_file = os.path.join(triplets_dir, f"{game}_triplets_indexed.csv")
    if not os.path.exists(raw_file):
        print(f"Raw triplets file not found for {game}, skipping.")
        continue

    raw_df = pd.read_csv(raw_file)
    clip_indices = np.unique(
        raw_df[["similar_clip_1_idx", "similar_clip_2_idx", "odd_clip_idx"]].values
    )
    n_clips = int(clip_indices.max()) + 1
    print(f"Loaded {len(raw_df)} raw triplets, {n_clips} clips")

    # -------------------------
    # Load DQN layer RDMs
    # -------------------------
    game_dqn_folder = os.path.join(DQN_RDM_BASE_FOLDER, game)
    dqn_rdm_files = sorted(glob.glob(os.path.join(game_dqn_folder, f"{game}_*_correlation_rdm.npy")))
    dqn_rdm_files = [f for f in dqn_rdm_files if "RSA" not in os.path.basename(f)]

    if len(dqn_rdm_files) == 0:
        print(f"No DQN RDM files found for {game}, skipping.")
        continue

    dqn_rdms = {}
    for dqn_file in dqn_rdm_files:
        layer_name = os.path.basename(dqn_file).replace(f"{game}_", "").replace("_RDM.npy", "")
        dqn_rdms[layer_name] = np.load(dqn_file)
    print(f"Loaded {len(dqn_rdms)} DQN layer RDMs: {list(dqn_rdms.keys())}")

    # -------------------------
    # Permutation loop
    # -------------------------
    null_distributions = {layer: [] for layer in dqn_rdms}

    for i in range(n_permutations):
        if (i + 1) % 100 == 0:
            print(f"  Permutation {i + 1}/{n_permutations}")

        # Shuffle clip indices globally across all triplet columns
        all_clips = raw_df[["similar_clip_1_idx", "similar_clip_2_idx", "odd_clip_idx"]].values.flatten()
        shuffled_clips = np.random.permutation(all_clips).reshape(-1, 3)

        shuffled_df = raw_df.copy()
        shuffled_df["similar_clip_1_idx"] = shuffled_clips[:, 0]
        shuffled_df["similar_clip_2_idx"] = shuffled_clips[:, 1]
        shuffled_df["odd_clip_idx"] = shuffled_clips[:, 2]

        constraints_df = to_constraints(shuffled_df)

        try:
            null_rdm = compute_rdm_from_constraints(constraints_df, n_clips)
        except Exception as e:
            print(f"  t-STE failed on permutation {i + 1}: {e}")
            continue

        null_rdm_obj = load_rdm_as_object(null_rdm, name="null")

        for layer_name, dqn_rdm in dqn_rdms.items():
            if null_rdm.shape != dqn_rdm.shape:
                continue
            dqn_rdm_obj = load_rdm_as_object(dqn_rdm, name=layer_name)
            rsa_score = compare(null_rdm_obj, dqn_rdm_obj, method="spearman")[0, 0]
            null_distributions[layer_name].append(rsa_score)

    # -------------------------
    # Compute p-values and save results
    # -------------------------
    for layer_name, null_scores in null_distributions.items():
        null_array = np.array(null_scores)

        # Save null distribution
        npy_path = os.path.join(SAVE_FOLDER, f"{game}_{layer_name}_null_distribution.npy")
        np.save(npy_path, null_array)

        # Get real RSA score
        real_rsa = real_rsa_lookup.get((game, layer_name), None)
        if real_rsa is None:
            print(f"Real RSA score not found for {game} {layer_name}, skipping p-value.")
            continue

        # p-value: proportion of null scores >= real RSA score
        pval = np.mean(null_array >= real_rsa)

        print(f"{game} | {layer_name}: real RSA = {real_rsa:.4f}, null mean = {null_array.mean():.4f} ± {null_array.std():.4f}, p = {pval:.4f}")

        all_results.append({
            "game": game,
            "layer": layer_name,
            "real_rsa": real_rsa,
            "null_mean": null_array.mean(),
            "null_std": null_array.std(),
            "null_95th": np.percentile(null_array, 95),
            "null_99th": np.percentile(null_array, 99),
            "pvalue": pval
        })

    # -------------------------
    # Plot null distributions with real RSA scores
    # -------------------------
    n_layers = len(null_distributions)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4), sharey=False, squeeze=False)

    for ax, (layer_name, null_scores) in zip(axes[0], null_distributions.items()):
        null_array = np.array(null_scores)
        real_rsa = real_rsa_lookup.get((game, layer_name), None)
        pval = all_results[-1]["pvalue"] if all_results else None

        ax.hist(null_array, bins=50, color="steelblue", edgecolor="white", alpha=0.8, label="Null distribution")
        ax.axvline(np.percentile(null_array, 95), color="orange", linestyle="--", linewidth=1.2, label="95th percentile")
        ax.axvline(np.percentile(null_array, 99), color="red", linestyle="--", linewidth=1.2, label="99th percentile")

        if real_rsa is not None:
            ax.axvline(real_rsa, color="green", linestyle="-", linewidth=2,
                       label=f"Real RSA = {real_rsa:.3f}")
            if pval is not None:
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                ax.set_title(f"{layer_name}\np = {pval:.4f} {sig}")
            else:
                ax.set_title(f"{layer_name}")
        else:
            ax.set_title(f"{layer_name}")

        ax.set_xlabel("RSA (Spearman)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)

    plt.suptitle(f"{game} — Null Distribution vs Real RSA", fontsize=13)
    plt.tight_layout()
    png_path = os.path.join(SAVE_FOLDER, f"{game}_null_vs_real_rsa.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Saved plot: {png_path}")

# =========================
# SAVE SUMMARY CSV
# =========================
summary_df = pd.DataFrame(all_results)
csv_path = os.path.join(SAVE_FOLDER, "permutation_baseline_summary.csv")
summary_df.to_csv(csv_path, index=False)
print(f"\nSaved summary to {csv_path}")
print("\nDone!")