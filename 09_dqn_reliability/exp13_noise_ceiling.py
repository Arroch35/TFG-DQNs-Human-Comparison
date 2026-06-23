import numpy as np
import os
import pandas as pd
from scipy.stats import spearmanr
import cy_tste

from src.utils import embedding_to_rdm, add_symmetric_triplets

# =========================================================
# CONFIG
# =========================================================
GAMES        = ["pacman", "pong", "spaceinvaders"]
SEEDS        = ["seed_42", "seed_0", "seed_1", "seed_2", "seed_3"]  # seed_42 must be first

BASE_RDM_FOLDER  = "../data/test_16_rdms/selected_subset_15"
CLIP_MAP_FOLDER = f"../data/maps/selected_15/"
TRIPLET_SCORES_DIR = "../data/triplets_results/triplet_scores"       # output of 9_5
SPARSE_RESPONSES_DIR = "../data/triplets_results/final_experiment/cleaned_results"

OUTPUT_DIR = "../data/triplets_results/noise_ceiling"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# t-STE params — same as 9_4
DIM      = 2
N_REPEATS = 100
MAX_ITER  = 1000

# =========================================================
# HELPERS
# =========================================================

def run_tste(triplets_local, rdm_42_sub):
    """Run t-STE N_REPEATS times, return RDM from the run that best
    correlates with seed_42's RDM — same selection criterion used in 9_4."""
    sym_triplets = add_symmetric_triplets(triplets_local)
    best_X    = None
    best_corr = -np.inf
    for _ in range(N_REPEATS):
        X = cy_tste.tste(
            sym_triplets,
            no_dims=DIM,
            max_iter=MAX_ITER,
            verbose=False,
            use_log=True,
        )
        rdm = embedding_to_rdm(X)
        corr = rdm_correlation(rdm_42_sub, rdm)
        if corr > best_corr:
            best_corr = corr
            best_X    = X
    return embedding_to_rdm(best_X)


def rdm_correlation(rdm1, rdm2):
    idx = np.triu_indices_from(rdm1, k=1)
    return spearmanr(rdm1[idx], rdm2[idx]).correlation


# =========================================================
# MAIN LOOP
# =========================================================
summary_rows = []

for game in GAMES:

    print("\n" + "=" * 60)
    print(f"GAME: {game}")
    print("=" * 60)

    # --- Load clip index map ---
    clip_map_csv = os.path.join(CLIP_MAP_FOLDER, f"{game}_clip_map.csv")
    if not os.path.exists(clip_map_csv):
        print(f"  Missing clip index map: {clip_map_csv}, skipping.")
        continue
    clip_map_df = pd.read_csv(clip_map_csv)
    orig_to_local = {orig: local for local, orig in enumerate(clip_map_df["clip_index"].tolist())}
    n_clips = len(clip_map_df)

    # --- Load seed_42 full RDM ---
    rdm42_path = os.path.join(BASE_RDM_FOLDER, "seed_42", game, f"{game}_fc_correlation_RDM.npy")
    if not os.path.exists(rdm42_path):
        print(f"  Missing seed_42 RDM: {rdm42_path}, skipping.")
        continue
    rdm_42_full = np.load(rdm42_path)
    # Subselect to the 15 clips used in the experiment
    local_indices = list(range(n_clips))
    rdm_42_sub = rdm_42_full[np.ix_(local_indices, local_indices)]

    # --- Load 9_5 master CSV ---
    master_csv = os.path.join(TRIPLET_SCORES_DIR, f"triplet_scores_{game}.csv")
    if not os.path.exists(master_csv):
        print(f"  Missing 9_5 output: {master_csv}, skipping.")
        continue
    master_df = pd.read_csv(master_csv)

    # --- Load sparse human responses ---
    sparse_csv = os.path.join(SPARSE_RESPONSES_DIR, f"{game}_triplets_indexed_with_difficulty.csv")
    if not os.path.exists(sparse_csv):
        print(f"  Missing sparse responses: {sparse_csv}, skipping.")
        continue
    sparse_df = pd.read_csv(sparse_csv)

    # --- Extract unique triplets from sparse responses (majority vote) ---
    # Build frozenset key for matching
    sparse_df["triplet_fs"] = sparse_df.apply(
        lambda r: frozenset({int(r["similar_clip_1_idx"]),
                              int(r["similar_clip_2_idx"]),
                              int(r["odd_clip_idx"])}), axis=1
    )
    unique_fs = sparse_df["triplet_fs"].unique()
    print(f"  Unique sparse triplets: {len(unique_fs)}")

    # Majority vote per unique triplet
    majority_votes = {}
    for fs, grp in sparse_df.groupby("triplet_fs"):
        majority_votes[fs] = grp["odd_clip_idx"].value_counts().index[0]

    # --- Filter master_df to sparse triplets ---
    master_df["triplet_fs"] = master_df.apply(
        lambda r: frozenset({int(r["similar_1_idx"]),
                              int(r["similar_2_idx"]),
                              int(r["odd_idx"])}), axis=1
    )
    sparse_master = master_df[master_df["triplet_fs"].isin(unique_fs)].copy()
    print(f"  Sparse triplets found in master: {len(sparse_master)}")

    # Identify seed answer columns from 9_5
    seed_answer_cols = [c for c in master_df.columns
                        if c.endswith("_answer") and c != "seed_42_answer"]
    other_seeds = [c.replace("_answer", "") for c in seed_answer_cols]

    if not other_seeds:
        print("  No other seed answer columns found in master CSV — skipping.")
        continue

    # -----------------------------------------------
    # For each non-42 seed: build triplet list using
    # THEIR answer, run t-STE, correlate with seed_42
    # -----------------------------------------------
    seed_correlations = []

    for seed, ans_col in zip(other_seeds, seed_answer_cols):
        print(f"\n  Processing {seed}...")

        # Build triplet array in local indices using seed's answers
        # Format: (similar_1_local, similar_2_local, odd_local)
        # where odd_local = seed's chosen odd clip
        triplets_local = []
        skipped = 0
        for _, row in sparse_master.iterrows():
            s1   = int(row["similar_1_idx"])
            s2   = int(row["similar_2_idx"])
            odd  = int(row["odd_idx"])          # seed_42's odd (defines the triplet)
            seed_odd = int(row[ans_col])         # this seed's answer

            # The two "similar" clips for this seed are the two that are NOT seed_odd
            clips = [s1, s2, odd]
            similar_clips = [c for c in clips if c != seed_odd]
            if len(similar_clips) != 2:
                skipped += 1
                continue

            l_s1  = orig_to_local.get(similar_clips[0])
            l_s2  = orig_to_local.get(similar_clips[1])
            l_odd = orig_to_local.get(seed_odd)

            if None in (l_s1, l_s2, l_odd):
                skipped += 1
                continue

            triplets_local.append((l_s1, l_s2, l_odd))

        if skipped:
            print(f"    Skipped {skipped} triplets (index not found)")

        if len(triplets_local) < 10:
            print(f"    Too few triplets ({len(triplets_local)}), skipping.")
            continue

        triplets_arr = np.array(triplets_local, dtype=np.int32)
        print(f"    Running t-STE on {len(triplets_arr)} triplets...")

        tste_rdm = run_tste(triplets_arr, rdm_42_sub)
        corr     = rdm_correlation(rdm_42_sub, tste_rdm)
        print(f"    Correlation with seed_42 RDM: {corr:.4f}")
        seed_correlations.append(corr)

        summary_rows.append({
            "game":        game,
            "seed":        seed,
            "n_triplets":  len(triplets_arr),
            "correlation": corr,
        })

    if seed_correlations:
        noise_ceiling = np.mean(seed_correlations)
        noise_ceiling_std = np.std(seed_correlations)
        print(f"\n  === Noise ceiling ({game}): {noise_ceiling:.4f} ± {noise_ceiling_std:.4f}"
              f"(mean ± std over {len(seed_correlations)} seeds) ===")
        summary_rows.append({
            "game":        game,
            "seed":        "NOISE_CEILING",
            "n_triplets":  len(sparse_master),
            "correlation": noise_ceiling,
            "std": noise_ceiling_std,
        })

# =========================================================
# SAVE
# =========================================================
summary_df = pd.DataFrame(summary_rows).round(4)
out_csv    = os.path.join(OUTPUT_DIR, "dqn_noise_ceiling.csv")
summary_df.to_csv(out_csv, index=False)
print(f"\nSaved summary → {out_csv}")

print("\n--- Final noise ceilings ---")
nc = summary_df[summary_df["seed"] == "NOISE_CEILING"]
print(nc[["game", "correlation"]].to_string(index=False))

print("\nDONE.")
