"""
exp10_sparse_individual_agreement.py
Compare sparse-data majority votes against full individual-data majority
votes, computing agreement bounds that account for ties.
Currently scoped to pong only.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.config import GAME_TO_GYM_ID, get_path, ensure
from src.utils import majority_vote_with_ties

# =========================================================
# CONFIG
# =========================================================
GAME         = "pong"
DIFFICULTIES = ["easy_triplets", "medium_triplets", "hard_triplets"]
CHANCE       = 1 / 3

# Paths
EXP2_DIR  = get_path("experiment_exp2")       # data/triplets_results/exp2/cleaned_results
DQN_JSON  = get_path("jsons_pong_triplets") # data/jsons/pong_final_triplet_exp.json

# Suggested addition to config.py PATHS:
#   "exp2_sparse_subset": DATA / "triplets_results" / "exp2" / "cleaned_results" / "sparse_individual_subset",
from src.config import DATA
SPARSE_DIR = DATA / "triplets_results" / "exp2" / "cleaned_results" / "sparse_individual_subset"

SPARSE_FILE     = SPARSE_DIR / f"{GAME}_sparse_individual_subset.csv"
INDIVIDUAL_FILE = EXP2_DIR   / f"{GAME}_triplets_indexed_with_difficulty.csv"
OUTPUT_DIR      = SPARSE_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# LOAD DQN TRIPLETS → difficulty label per clip-set
# =========================================================
with open(DQN_JSON, "r") as f:
    dqn_triplets_raw = json.load(f)

gym_key = GAME_TO_GYM_ID[GAME]   # "PongNoFrameskip-v4"

triplet_to_difficulty = {
    frozenset(triplet): difficulty
    for difficulty in DIFFICULTIES
    for triplet in dqn_triplets_raw[gym_key][difficulty]
}

# =========================================================
# LOAD DATA & ADD TRIPLET IDs
# =========================================================
def add_triplet_id(df):
    df["triplet_id"] = df.apply(
        lambda row: frozenset({row["similar_clip_1_idx"], row["similar_clip_2_idx"], row["odd_clip_idx"]}),
        axis=1,
    )
    return df

sparse_df     = add_triplet_id(pd.read_csv(SPARSE_FILE))
individual_df = add_triplet_id(pd.read_csv(INDIVIDUAL_FILE))

# =========================================================
# MAJORITY VOTES
# =========================================================
def majority_vote(group):
    return group["odd_clip_idx"].value_counts().idxmax()

individual_votes = (individual_df.groupby("triplet_id")
                    .apply(majority_vote)
                    .reset_index()
                    .rename(columns={0: "individual_majority"}))

sparse_votes_raw = (sparse_df.groupby("triplet_id")
                    .apply(majority_vote_with_ties)
                    .reset_index()
                    .rename(columns={0: "sparse_candidates"}))

# =========================================================
# MERGE & ANNOTATE
# =========================================================
merged = sparse_votes_raw.merge(individual_votes, on="triplet_id")
merged["difficulty"]    = merged["triplet_id"].map(triplet_to_difficulty)
merged["n_candidates"]  = merged["sparse_candidates"].apply(len)
merged["has_tie"]       = merged["n_candidates"] > 1

print(f"Total triplets:              {len(merged)}")
print(f"Triplets with ties (sparse): {merged['has_tie'].sum()} / {len(merged)}\n")

# =========================================================
# AGREEMENT BOUNDS PER DIFFICULTY
# =========================================================
summary_rows = []

for difficulty in DIFFICULTIES:
    label   = difficulty.replace("_triplets", "").capitalize()
    subset  = merged[merged["difficulty"] == difficulty].copy()
    n_total = len(subset)
    n_ties  = subset["has_tie"].sum()

    fixed        = subset[~subset["has_tie"]]
    fixed_agrees = (fixed["sparse_candidates"].apply(lambda x: x[0]) == fixed["individual_majority"]).sum()

    tied      = subset[subset["has_tie"]]
    min_extra = max_extra = 0
    for _, row in tied.iterrows():
        if row["individual_majority"] in row["sparse_candidates"]:
            max_extra += 1   # best case: matching candidate chosen

    min_agree = (fixed_agrees + min_extra) / n_total
    max_agree = (fixed_agrees + max_extra) / n_total

    observed_agrees = fixed_agrees + tied.apply(
        lambda row: int(row["sparse_candidates"][0] == row["individual_majority"]), axis=1
    ).sum()
    observed_agree = observed_agrees / n_total

    p_min = binomtest(int(min_agree * n_total), n_total, CHANCE, alternative="greater").pvalue
    p_max = binomtest(int(max_agree * n_total), n_total, CHANCE, alternative="greater").pvalue
    p_obs = binomtest(int(observed_agree * n_total), n_total, CHANCE, alternative="greater").pvalue

    print(f"{label}: n={n_total}, ties={n_ties}, fixed_agrees={fixed_agrees}")
    print(f"  Observed (arbitrary tie-break): {observed_agree:.3f}  (p={p_obs:.4f})")
    print(f"  Min possible agreement:         {min_agree:.3f}  (p={p_min:.4f})")
    print(f"  Max possible agreement:         {max_agree:.3f}  (p={p_max:.4f})\n")

    summary_rows.append({
        "difficulty": label, "n_total": n_total, "n_ties": n_ties,
        "observed": observed_agree, "min": min_agree, "max": max_agree,
        "p_observed": p_obs, "p_min": p_min, "p_max": p_max,
    })

# =========================================================
# SAVE
# =========================================================
out_csv = OUTPUT_DIR / "sparse_individual_agreement_with_bounds.csv"
pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
print(f"Saved → {out_csv}")
