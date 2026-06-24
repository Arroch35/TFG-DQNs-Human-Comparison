"""
5_filter_sparse_to_pong60_subset.py
Filter the sparse pong results down to only the 60 reference triplets
defined in the DQN triplet JSON.
"""
import json
import os
import pandas as pd

from src.config import REFERENCE_SEED, get_path, ensure

# =========================================================
# CONFIG
# =========================================================
GAME        = "pong"
GYM_KEY     = "PongNoFrameskip-v4"
DIFFICULTIES = ["easy_triplets", "medium_triplets", "hard_triplets"]

# Paths — all from config
SPARSE_FILE      = get_path("experiment_sparse") / "pong_triplets_indexed_with_difficulty.csv"
DQN_TRIPLETS_JSON = get_path("jsons_pong_triplets")
OUTPUT_DIR = ensure("experiment_sparse_individual_subset")

# =========================================================
# LOAD REFERENCE TRIPLETS (60 DQN-defined triplets)
# =========================================================
with open(DQN_TRIPLETS_JSON, "r") as f:
    dqn_triplets_raw = json.load(f)

reference_triplets = set()
for difficulty in DIFFICULTIES:
    for triplet in dqn_triplets_raw[GYM_KEY][difficulty]:
        reference_triplets.add(frozenset(triplet))

print(f"Reference triplets loaded: {len(reference_triplets)}")

# =========================================================
# FILTER
# =========================================================
df = pd.read_csv(SPARSE_FILE)

df["triplet_id"] = df.apply(
    lambda row: frozenset({row["similar_clip_1_idx"], row["similar_clip_2_idx"], row["odd_clip_idx"]}),
    axis=1,
)

filtered_df = df[df["triplet_id"].isin(reference_triplets)].drop(columns=["triplet_id"])

n_total  = len(df)
n_kept   = len(filtered_df)
n_unique = filtered_df.apply(
    lambda row: frozenset({row["similar_clip_1_idx"], row["similar_clip_2_idx"], row["odd_clip_idx"]}),
    axis=1,
).nunique()

print(f"{GAME}: {n_total} sparse rows → {n_kept} kept ({n_unique} unique triplets matched out of 60)")

output_file = OUTPUT_DIR / f"{GAME}_sparse_individual_subset.csv"
filtered_df.to_csv(output_file, index=False)
print(f"Saved → {output_file}")
