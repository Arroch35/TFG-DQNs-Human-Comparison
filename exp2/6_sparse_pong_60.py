import os
import json
import pandas as pd

# =========================
# CONFIGURATION
# =========================
game = "pong"
json_key = "PongNoFrameskip-v4"
difficulties = ["easy_triplets", "medium_triplets", "hard_triplets"]

sparse_file = "../data/triplets_results/final_experiment/cleaned_results/pong_triplets_indexed_with_difficulty.csv"
dqn_triplets_path = "../data/jsons/pong_final_triplet_exp.json"
output_dir = "../data/triplets_results/exp2/cleaned_results/sparse_individual_subset"
os.makedirs(output_dir, exist_ok=True)

# =========================
# LOAD JSON — 60 REFERENCE TRIPLETS
# =========================
with open(dqn_triplets_path, "r") as f:
    dqn_triplets_raw = json.load(f)

reference_triplets = set()
for difficulty in difficulties:
    for triplet in dqn_triplets_raw[json_key][difficulty]:
        reference_triplets.add(frozenset(triplet))

print(f"Reference triplets loaded: {len(reference_triplets)}")

# =========================
# FILTER SPARSE CSV
# =========================
df = pd.read_csv(sparse_file)

df["triplet_id"] = df.apply(
    lambda row: frozenset({row["similar_clip_1_idx"], row["similar_clip_2_idx"], row["odd_clip_idx"]}),
    axis=1
)

mask = df["triplet_id"].apply(lambda x: x in reference_triplets)
filtered_df = df[mask].drop(columns=["triplet_id"])

n_total = len(df)
n_kept = len(filtered_df)
n_unique = filtered_df.apply(
    lambda row: frozenset({row["similar_clip_1_idx"], row["similar_clip_2_idx"], row["odd_clip_idx"]}),
    axis=1
).nunique()

print(f"{game}: {n_total} sparse rows → {n_kept} kept ({n_unique} unique triplets matched out of 60)")

output_file = os.path.join(output_dir, f"{game}_sparse_individual_subset.csv")
filtered_df.to_csv(output_file, index=False)
print(f"Saved to {output_file}")