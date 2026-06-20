import os
import json
import pandas as pd
import numpy as np
from scipy.stats import binomtest

# =========================
# CONFIGURATION
# =========================
game = "pong"
json_key = "PongNoFrameskip-v4"
difficulties = ["easy_triplets", "medium_triplets", "hard_triplets"]
difficulty_labels = ["Easy", "Medium", "Hard"]
chance_level = 1 / 3

sparse_subset_file = "../data/triplets_results/exp2/cleaned_results/sparse_individual_subset/pong_sparse_individual_subset.csv"
individual_file = "../data/triplets_results/exp2/cleaned_results/pong_triplets_indexed_with_difficulty.csv"
dqn_triplets_path = "../data/jsons/pong_final_triplet_exp.json"
output_dir = "../data/triplets_results/exp2/cleaned_results/sparse_individual_subset"
os.makedirs(output_dir, exist_ok=True)

# =========================
# LOAD DQN TRIPLETS — for difficulty labels
# =========================
with open(dqn_triplets_path, "r") as f:
    dqn_triplets_raw = json.load(f)

triplet_to_difficulty = {}
for difficulty in difficulties:
    for triplet in dqn_triplets_raw[json_key][difficulty]:
        triplet_to_difficulty[frozenset(triplet)] = difficulty

# =========================
# LOAD DATA
# =========================
sparse_df = pd.read_csv(sparse_subset_file)
individual_df = pd.read_csv(individual_file)

def add_triplet_id(df):
    df["triplet_id"] = df.apply(
        lambda row: frozenset({row["similar_clip_1_idx"], row["similar_clip_2_idx"], row["odd_clip_idx"]}),
        axis=1
    )
    return df

sparse_df = add_triplet_id(sparse_df)
individual_df = add_triplet_id(individual_df)

# =========================
# INDIVIDUAL MAJORITY VOTE (deterministic — many responses per triplet)
# =========================
def majority_vote(group):
    return group["odd_clip_idx"].value_counts().idxmax()

individual_votes = individual_df.groupby("triplet_id").apply(majority_vote).reset_index()
individual_votes.columns = ["triplet_id", "individual_majority"]

# =========================
# SPARSE MAJORITY VOTE WITH TIE DETECTION
# Returns list of all tied candidates (length 1 = no tie, >1 = tie)
# =========================
def majority_vote_with_ties(group):
    counts = group["odd_clip_idx"].value_counts()
    max_count = counts.iloc[0]
    tied = counts[counts == max_count].index.tolist()
    return tied

sparse_votes_raw = sparse_df.groupby("triplet_id").apply(majority_vote_with_ties).reset_index()
sparse_votes_raw.columns = ["triplet_id", "sparse_candidates"]

# =========================
# MERGE AND ANNOTATE
# =========================
merged = sparse_votes_raw.merge(individual_votes, on="triplet_id")
merged["difficulty"] = merged["triplet_id"].map(triplet_to_difficulty)
merged["n_candidates"] = merged["sparse_candidates"].apply(len)
merged["has_tie"] = merged["n_candidates"] > 1

print(f"Total triplets: {len(merged)}")
print(f"Triplets with ties in sparse: {merged['has_tie'].sum()} / {len(merged)}")
print()

# =========================
# COMPUTE AGREEMENT BOUNDS PER DIFFICULTY
# =========================
summary_rows = []

for difficulty in difficulties:
    subset = merged[merged["difficulty"] == difficulty].copy()
    label = difficulty.replace("_triplets", "").capitalize()
    n_total = len(subset)
    n_ties = subset["has_tie"].sum()

    # Fixed agreements: triplets with no tie — outcome is deterministic
    fixed = subset[~subset["has_tie"]]
    fixed_agrees = (
        fixed["sparse_candidates"].apply(lambda x: x[0]) == fixed["individual_majority"]
    ).sum()

    # Tied triplets: compute best and worst case
    tied = subset[subset["has_tie"]]
    min_extra = 0
    max_extra = 0
    for _, row in tied.iterrows():
        ind_vote = row["individual_majority"]
        candidates = row["sparse_candidates"]
        if ind_vote in candidates:
            max_extra += 1  # best case: the matching candidate is chosen
            min_extra += 0  # worst case: a non-matching candidate is chosen
        # if no candidate matches, both min and max get 0 regardless

    min_agree = (fixed_agrees + min_extra) / n_total
    max_agree = (fixed_agrees + max_extra) / n_total

    # Observed: what original code computed (arbitrary first candidate in case of tie)
    observed_agrees = fixed_agrees + tied.apply(
        lambda row: int(row["sparse_candidates"][0] == row["individual_majority"]), axis=1
    ).sum()
    observed_agree = observed_agrees / n_total

    p_min = binomtest(int(min_agree * n_total), n_total, chance_level, alternative="greater").pvalue
    p_max = binomtest(int(max_agree * n_total), n_total, chance_level, alternative="greater").pvalue
    p_obs = binomtest(int(observed_agree * n_total), n_total, chance_level, alternative="greater").pvalue

    print(f"{label}: n={n_total}, ties={n_ties}, fixed_agrees={fixed_agrees}")
    print(f"  Observed (arbitrary tie-break): {observed_agree:.3f}  (p={p_obs:.4f})")
    print(f"  Min possible agreement:         {min_agree:.3f}  (p={p_min:.4f})")
    print(f"  Max possible agreement:         {max_agree:.3f}  (p={p_max:.4f})")
    print()

    summary_rows.append({
        "difficulty": label,
        "n_total": n_total,
        "n_ties": n_ties,
        "observed": observed_agree,
        "min": min_agree,
        "max": max_agree,
        "p_observed": p_obs,
        "p_min": p_min,
        "p_max": p_max,
    })

# =========================
# SAVE
# =========================
summary_df = pd.DataFrame(summary_rows)
out_csv = os.path.join(output_dir, "sparse_individual_agreement_with_bounds.csv")
summary_df.to_csv(out_csv, index=False)
print(f"Saved to {out_csv}")
