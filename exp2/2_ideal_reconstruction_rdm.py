import numpy as np
import os
import json
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import cy_tste

# =========================================================
# CONFIG
# =========================================================
GAME = "pong"
LAYER = "fc"
METHOD = "correlation"
SEED = "seed_42"

BASE_RDM_FOLDER = f"../data/test_16_rdms/selected_subset_15/{SEED}"
SAVE_FOLDER = f"../data/triplet_results/exp2/rdms/{SEED}"

os.makedirs(SAVE_FOLDER, exist_ok=True)

DIM = 2
MAX_ITER = 1000
N_REPEATS = 100

# =========================================================
# LOAD HUMAN TRIPLETS JSON
# =========================================================
JSON_PATH = "../data/jsons/pong_final_triplet_exp.json"

with open(JSON_PATH, "r") as f:
    final_trials = json.load(f)

# =========================================================
# LOAD ORIGINAL RDM
# =========================================================
rdm_path = os.path.join(
    BASE_RDM_FOLDER,
    GAME,
    f"{GAME}_{LAYER}_{METHOD}_RDM.npy"
)

rdm = np.load(rdm_path)

N = rdm.shape[0]

print(f"Loaded RDM: {rdm.shape}")

# =========================================================
# BUILD T-STE TRIPLETS
# =========================================================
# IMPORTANT:
# t-STE expects:
#
# (anchor, positive, negative)
#
# meaning:
# d(anchor, positive) < d(anchor, negative)
#
# Your triplets are:
# [similar1, similar2, odd]
#
# so we create:
#
# (similar1, similar2, odd)
# (similar2, similar1, odd)
#
# for symmetry
# =========================================================
triplets = []

game_triplets = final_trials["PongNoFrameskip-v4"]

for difficulty in [
    "easy_triplets",
    "medium_triplets",
    "hard_triplets"
]:

    for t in game_triplets[difficulty]:

        s1 = t[0]
        s2 = t[1]
        odd = t[2]

        # symmetric constraints
        triplets.append((s1, s2, odd))
        triplets.append((s2, s1, odd))

triplets = np.array(triplets, dtype=np.int32)

print(f"Total triplets used: {len(triplets)}")

# =========================================================
# EMBEDDING → RDM
# =========================================================
def embedding_to_rdm(X):

    diff = X[:, None, :] - X[None, :, :]

    return np.linalg.norm(diff, axis=-1)

# =========================================================
# COMPARE RDMS
# =========================================================
def compare_rdms(rdm1, rdm2):

    idx = np.triu_indices_from(rdm1, k=1)

    return spearmanr(
        rdm1[idx],
        rdm2[idx]
    ).correlation

# =========================================================
# RUN MULTIPLE T-STE INITIALIZATIONS
# =========================================================
results = []

best_score = -np.inf
best_rdm = None
best_embedding = None

for repeat in range(N_REPEATS):

    print(f"\nRepeat {repeat+1}/{N_REPEATS}")

    # -----------------------------------------------------
    # Fit t-STE
    # -----------------------------------------------------
    X = cy_tste.tste(
        triplets,
        no_dims=DIM,
        max_iter=MAX_ITER,
        verbose=False,
        use_log=True
    )

    # -----------------------------------------------------
    # Reconstruct RDM
    # -----------------------------------------------------
    rdm_rec = embedding_to_rdm(X)

    # -----------------------------------------------------
    # Compare against original RDM
    # -----------------------------------------------------
    score = compare_rdms(rdm, rdm_rec)

    results.append({
        "repeat": repeat,
        "score": score
    })

    print(f"Spearman correlation: {score:.4f}")

    if score > best_score:

        best_score = score
        best_rdm = rdm_rec.copy()
        best_embedding = X.copy()

# =========================================================
# SAVE RAW RESULTS
# =========================================================
df = pd.DataFrame(results)

raw_csv = os.path.join(
    SAVE_FOLDER,
    "ideal_human_triplet_reconstruction_raw.csv"
)

df.to_csv(raw_csv, index=False)

print(f"\nSaved raw results → {raw_csv}")


# =========================================================
# SAVE BEST RECONSTRUCTED RDM
# =========================================================
best_rdm_path = os.path.join(
    SAVE_FOLDER,
    "ideal_best_reconstructed_rdm.npy"
)

np.save(best_rdm_path, best_rdm)

print(f"Saved best reconstructed RDM → {best_rdm_path}")

png_path = os.path.join(
    SAVE_FOLDER,
    "ideal_best_reconstructed_rdm.png"
)

plt.figure(figsize=(6, 6))

plt.imshow(best_rdm, interpolation="nearest")
plt.colorbar()

plt.title(
    f"Best Reconstructed RDM\nSpearman = {best_score:.4f}"
)

plt.xlabel("Clip index")
plt.ylabel("Clip index")

plt.tight_layout()
plt.savefig(png_path, dpi=300)
plt.close()

print(f"Saved best reconstructed RDM PNG → {png_path}")

# =========================================================
# SUMMARY
# =========================================================
mean_score = df["score"].mean()
std_score = df["score"].std()
min_score = df["score"].min()
max_score = df["score"].max()

summary = pd.DataFrame([{
    "game": GAME,
    "layer": LAYER,
    "n_triplets": len(triplets),
    "mean_score": mean_score,
    "std_score": std_score,
    "min_score": min_score,
    "max_score": max_score
}])

summary_csv = os.path.join(
    SAVE_FOLDER,
    "ideal_human_triplet_reconstruction_summary.csv"
)

summary.to_csv(summary_csv, index=False)

print(f"Saved summary → {summary_csv}")

# =========================================================
# PRINT FINAL RESULT
# =========================================================
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

print(f"Game: {GAME}")
print(f"Layer: {LAYER}")
print(f"Triplets used: {len(triplets)}")

print(f"\nMean Spearman correlation: {mean_score:.4f}")
print(f"Std: {std_score:.4f}")
print(f"Min: {min_score:.4f}")
print(f"Max: {max_score:.4f}")

print(f"\nRDM reconstruction quality:")
print(f"{mean_score*100:.2f}%")

# =========================================================
# HISTOGRAM
# =========================================================
plt.figure(figsize=(7,5))

plt.hist(df["score"], bins=15)

plt.xlabel("Spearman correlation")
plt.ylabel("Frequency")
plt.title("Human triplet → RDM reconstruction")

plot_path = os.path.join(
    SAVE_FOLDER,
    "human_triplet_histogram.png"
)

plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"Saved histogram → {plot_path}")