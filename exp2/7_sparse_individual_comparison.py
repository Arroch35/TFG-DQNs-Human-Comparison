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
# LOAD DQN TRIPLETS — for difficulty labels and triplet identity
# =========================
with open(dqn_triplets_path, "r") as f:
    dqn_triplets_raw = json.load(f)

# Build lookup: frozenset -> difficulty
triplet_to_difficulty = {}
for difficulty in difficulties:
    for triplet in dqn_triplets_raw[json_key][difficulty]:
        triplet_to_difficulty[frozenset(triplet)] = difficulty

# =========================
# LOAD DATA
# =========================
sparse_df = pd.read_csv(sparse_subset_file)
individual_df = pd.read_csv(individual_file)

# Add triplet_id to both
def add_triplet_id(df):
    df["triplet_id"] = df.apply(
        lambda row: frozenset({row["similar_clip_1_idx"], row["similar_clip_2_idx"], row["odd_clip_idx"]}),
        axis=1
    )
    return df

sparse_df = add_triplet_id(sparse_df)
individual_df = add_triplet_id(individual_df)

# =========================
# COMPUTE MAJORITY VOTE PER TRIPLET FOR EACH EXPERIMENT
# =========================
def majority_vote(group):
    return group["odd_clip_idx"].value_counts().idxmax()

sparse_votes = sparse_df.groupby("triplet_id").apply(majority_vote).reset_index()
sparse_votes.columns = ["triplet_id", "sparse_majority"]

individual_votes = individual_df.groupby("triplet_id").apply(majority_vote).reset_index()
individual_votes.columns = ["triplet_id", "individual_majority"]

# =========================
# MERGE AND COMPARE
# =========================
merged = sparse_votes.merge(individual_votes, on="triplet_id")
merged["difficulty"] = merged["triplet_id"].map(triplet_to_difficulty)
merged["agree"] = (merged["sparse_majority"] == merged["individual_majority"]).astype(int)

print(f"\nTotal matched triplets: {len(merged)}")
print(f"Overall agreement: {merged['agree'].mean():.3f} ({merged['agree'].sum()}/{len(merged)})")

# =========================
# AGGREGATE BY DIFFICULTY
# =========================
summary_rows = []

for difficulty in difficulties:
    subset = merged[merged["difficulty"] == difficulty]
    if subset.empty:
        continue
    n_total = len(subset)
    n_agree = subset["agree"].sum()
    agreement = n_agree / n_total

    binom_result = binomtest(n_agree, n_total, chance_level, alternative="greater")
    pval = binom_result.pvalue

    print(f"{difficulty}: agreement = {agreement:.3f} ({n_agree}/{n_total}), p = {pval:.4f}")
    summary_rows.append({
        "difficulty": difficulty,
        "n_total": n_total,
        "n_agree": n_agree,
        "agreement": agreement,
        "pvalue": pval
    })

summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(output_dir, "pong_sparse_vs_individual_agreement.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"\nSaved summary to {summary_csv}")

# =========================
# PLOT
# =========================
import matplotlib.pyplot as plt

agreements = []
pvals = []
for difficulty in difficulties:
    row = summary_df[summary_df["difficulty"] == difficulty]
    agreements.append(row["agreement"].values[0] if not row.empty else 0)
    pvals.append(row["pvalue"].values[0] if not row.empty else 1)

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(difficulty_labels, agreements, color=["#2ecc71", "#f39c12", "#e74c3c"], width=0.5)
ax.axhline(chance_level, color="black", linestyle="--", linewidth=1.2, label="Chance (1/3)")
ax.set_ylim(0, 1)
ax.set_xlabel("Difficulty")
ax.set_ylabel("Agreement Rate")
ax.set_title("Pong — Sparse vs Individual Experiment Agreement")
ax.legend()

for bar, pval in zip(bars, pvals):
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, sig,
            ha="center", va="bottom", fontsize=12)

plt.tight_layout()
plot_file = os.path.join(output_dir, "pong_sparse_vs_individual_agreement.png")
plt.savefig(plot_file, dpi=300)
plt.close()
print(f"Saved plot to {plot_file}")