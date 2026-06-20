import os
import numpy as np
import pandas as pd
from scipy.stats import binomtest

# =========================================================
# CONFIG
# =========================================================
GAMES = ["pacman", "pong", "spaceinvaders"]

TRIPLET_SCORES_DIR   = "../data/triplets_results/triplet_scores"
SPARSE_RESPONSES_DIR = "../data/triplets_results/final_experiment/cleaned_results"
OUTPUT_DIR           = "../data/triplets_results/human_agreement_by_difficulty"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHANCE = 1 / 3


# =========================================================
# HELPERS
# =========================================================

def majority_vote_with_ties(group):
    """
    Returns list of all candidates tied for majority.
    Length 1  → clean majority.
    Length >1 → tie: any of the candidates could be the majority vote.
    """
    counts    = group["odd_clip_idx"].value_counts()
    max_count = counts.iloc[0]
    return counts[counts == max_count].index.tolist()


def agreement_with_bounds(sparse_candidates_series, ground_truth_series):
    """
    Given a series of candidate lists (from majority_vote_with_ties) and
    the ground-truth odd_idx for each triplet, compute:
      - observed  : first candidate used as tie-break (matches 7_ script)
      - min_agree : worst-case tie resolution
      - max_agree : best-case tie resolution
    Returns (observed, min_agree, max_agree) as fractions.
    """
    n          = len(sparse_candidates_series)
    fixed_agrees  = 0
    min_extra  = 0
    max_extra  = 0
    observed_agrees = 0

    for candidates, gt in zip(sparse_candidates_series, ground_truth_series):
        if len(candidates) == 1:
            # clean majority — deterministic
            match = int(candidates[0] == gt)
            fixed_agrees    += match
            observed_agrees += match
        else:
            # tie — compute bounds
            observed_agrees += int(candidates[0] == gt)   # arbitrary first
            if gt in candidates:
                max_extra += 1   # best case: correct candidate wins tie
                # min_extra += 0   # worst case: wrong candidate wins tie
            # if gt not in candidates, neither case contributes

    observed  = observed_agrees / n
    min_agree = (fixed_agrees + 0)          / n   # min_extra always 0 worst case
    max_agree = (fixed_agrees + max_extra)  / n

    return observed, min_agree, max_agree


# =========================================================
# MAIN LOOP
# =========================================================
all_summary = []

for game in GAMES:

    print("\n" + "=" * 60)
    print(f"GAME: {game}")
    print("=" * 60)

    # --- Load master CSV (triplet scores from 9_5) ---
    master_csv = os.path.join(TRIPLET_SCORES_DIR, f"triplet_scores_{game}.csv")
    if not os.path.exists(master_csv):
        print(f"  Missing master CSV: {master_csv}, skipping.")
        continue
    master_df = pd.read_csv(master_csv)

    # Build frozenset triplet ID on master
    master_df["triplet_fs"] = master_df.apply(
        lambda r: frozenset({int(r["similar_1_idx"]),
                              int(r["similar_2_idx"]),
                              int(r["odd_idx"])}), axis=1
    )

    # Assign difficulty buckets from percentile thresholds
    master_df["bucket"] = master_df["difficulty"].str.replace("_triplets", "").str.capitalize()

    # --- Load sparse human responses ---
    sparse_csv = os.path.join(
        SPARSE_RESPONSES_DIR,
        f"{game}_triplets_indexed_with_difficulty.csv"
    )
    if not os.path.exists(sparse_csv):
        print(f"  Missing sparse responses: {sparse_csv}, skipping.")
        continue
    sparse_df = pd.read_csv(sparse_csv)

    # Build frozenset triplet ID on sparse
    sparse_df["triplet_fs"] = sparse_df.apply(
        lambda r: frozenset({int(r["similar_clip_1_idx"]),
                              int(r["similar_clip_2_idx"]),
                              int(r["odd_clip_idx"])}), axis=1
    )

    # --- Tie-aware majority vote per triplet ---
    majority = (
        sparse_df.groupby("triplet_fs")
        .apply(majority_vote_with_ties)
        .reset_index()
        .rename(columns={0: "candidates"})
    )

    # --- Merge with master to get odd_idx (ground truth) and bucket ---
    merged = master_df[["triplet_fs", "odd_idx", "bucket", "difficulty_score"]].merge(
        majority, on="triplet_fs", how="inner"
    )

    print(f"  Sparse triplets matched to master: {len(merged)}")
    print(f"  Bucket distribution among matched triplets:")
    print(merged["bucket"].value_counts().to_string())
    print()

    # --- Per-bucket analysis ---
    for bucket_label in ["Easy", "Medium", "Hard"]:
        subset = merged[merged["bucket"] == bucket_label].copy()
        n = len(subset)
        if n == 0:
            print(f"  {bucket_label}: no triplets, skipping.")
            continue

        n_ties = subset["candidates"].apply(lambda x: len(x) > 1).sum()

        observed, min_agree, max_agree = agreement_with_bounds(
            subset["candidates"], subset["odd_idx"]
        )

        # Binomial tests against chance (1/3)
        p_obs = binomtest(round(observed  * n), n, CHANCE, alternative="greater").pvalue
        p_min = binomtest(round(min_agree * n), n, CHANCE, alternative="greater").pvalue
        p_max = binomtest(round(max_agree * n), n, CHANCE, alternative="greater").pvalue

        print(f"  {bucket_label}: n={n}, ties={n_ties}")
        print(f"    Observed agreement : {observed:.3f}  (p={p_obs:.4f})")
        print(f"    Min possible       : {min_agree:.3f}  (p={p_min:.4f})")
        print(f"    Max possible       : {max_agree:.3f}  (p={p_max:.4f})")
        print()

        all_summary.append({
            "game":       game,
            "bucket":     bucket_label,
            "n_triplets": n,
            "n_ties":     n_ties,
            "observed":   observed,
            "min":        min_agree,
            "max":        max_agree,
            "p_observed": p_obs,
            "p_min":      p_min,
            "p_max":      p_max,
        })

# =========================================================
# SAVE SUMMARY
# =========================================================
summary_df = pd.DataFrame(all_summary).round(4)
out_csv = os.path.join(OUTPUT_DIR, "human_agreement_by_difficulty.csv")
summary_df.to_csv(out_csv, index=False)
print(f"\nSaved summary → {out_csv}")