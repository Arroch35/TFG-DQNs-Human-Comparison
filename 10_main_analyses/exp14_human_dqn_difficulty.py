"""
exp14_human_dqn_difficulty.py
Compute human agreement with the DQN seed_42 answer per difficulty bucket,
with tie-aware bounds using majority_vote_with_ties.
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.config import GAMES, get_path, ensure
from src.utils import majority_vote_with_ties

# =========================================================
# CONFIG
# =========================================================
CHANCE = 1 / 3

# Suggested additions to config.py PATHS:
#   "human_agreement": DATA / "triplets_results" / "human_agreement_by_difficulty",
from src.config import DATA
OUTPUT_DIR = DATA / "triplets_results" / "human_agreement_by_difficulty"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# HELPERS
# =========================================================
def agreement_with_bounds(sparse_candidates_series, ground_truth_series):
    """
    Returns (observed, min_agree, max_agree) as fractions.
    observed  : first candidate as tie-break
    min_agree : worst-case tie resolution
    max_agree : best-case tie resolution
    """
    n = len(sparse_candidates_series)
    fixed_agrees = observed_agrees = max_extra = 0

    for candidates, gt in zip(sparse_candidates_series, ground_truth_series):
        if len(candidates) == 1:
            match            = int(candidates[0] == gt)
            fixed_agrees    += match
            observed_agrees += match
        else:
            observed_agrees += int(candidates[0] == gt)
            if gt in candidates:
                max_extra += 1

    observed  = observed_agrees / n
    min_agree = fixed_agrees / n
    max_agree = (fixed_agrees + max_extra) / n
    return observed, min_agree, max_agree


# =========================================================
# MAIN
# =========================================================
all_summary = []

for game in GAMES:
    print(f"\n{'='*60}\nGAME: {game}\n{'='*60}")

    master_csv = get_path("triplets_scores_csv", game=game)
    if not master_csv.exists():
        print(f"  Missing master CSV: {master_csv}, skipping."); continue
    master_df = pd.read_csv(master_csv)

    master_df["triplet_fs"] = master_df.apply(
        lambda r: frozenset({int(r["similar_1_idx"]), int(r["similar_2_idx"]), int(r["odd_idx"])}), axis=1)
    master_df["bucket"] = master_df["difficulty"].str.replace("_triplets", "").str.capitalize()

    sparse_csv = get_path("experiment_cleaned") / f"{game}_triplets_indexed_with_difficulty.csv"
    if not sparse_csv.exists():
        print(f"  Missing sparse responses: {sparse_csv}, skipping."); continue
    sparse_df = pd.read_csv(sparse_csv)

    sparse_df["triplet_fs"] = sparse_df.apply(
        lambda r: frozenset({int(r["similar_clip_1_idx"]), int(r["similar_clip_2_idx"]), int(r["odd_clip_idx"])}), axis=1)

    majority = (
        sparse_df.groupby("triplet_fs")
        .apply(majority_vote_with_ties)
        .reset_index()
        .rename(columns={0: "candidates"})
    )

    merged = master_df[["triplet_fs", "odd_idx", "bucket", "difficulty_score"]].merge(
        majority, on="triplet_fs", how="inner"
    )

    print(f"  Sparse triplets matched: {len(merged)}")
    print(merged["bucket"].value_counts().to_string())

    for bucket_label in ["Easy", "Medium", "Hard"]:
        subset = merged[merged["bucket"] == bucket_label].copy()
        n = len(subset)
        if n == 0:
            print(f"  {bucket_label}: no triplets, skipping."); continue

        n_ties = subset["candidates"].apply(lambda x: len(x) > 1).sum()
        observed, min_agree, max_agree = agreement_with_bounds(subset["candidates"], subset["odd_idx"])

        p_obs = binomtest(round(observed  * n), n, CHANCE, alternative="greater").pvalue
        p_min = binomtest(round(min_agree * n), n, CHANCE, alternative="greater").pvalue
        p_max = binomtest(round(max_agree * n), n, CHANCE, alternative="greater").pvalue

        print(f"\n  {bucket_label}: n={n}, ties={n_ties}")
        print(f"    Observed  : {observed:.3f}  (p={p_obs:.4f})")
        print(f"    Min       : {min_agree:.3f}  (p={p_min:.4f})")
        print(f"    Max       : {max_agree:.3f}  (p={p_max:.4f})")

        all_summary.append({"game": game, "bucket": bucket_label, "n_triplets": n, "n_ties": n_ties,
                             "observed": observed, "min": min_agree, "max": max_agree,
                             "p_observed": p_obs, "p_min": p_min, "p_max": p_max})

# =========================================================
# SAVE
# =========================================================
out_csv = OUTPUT_DIR / "human_agreement_by_difficulty.csv"
pd.DataFrame(all_summary).round(4).to_csv(out_csv, index=False)
print(f"\nSaved summary → {out_csv}")
