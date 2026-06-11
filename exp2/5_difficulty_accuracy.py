import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest, ttest_1samp

# =========================
# CONFIGURATION
# =========================
games = ["pong"]
game_to_json_key = {
    "pong": "PongNoFrameskip-v4",
}
difficulties = ["easy_triplets", "medium_triplets", "hard_triplets"]
difficulty_order = ["easy_triplets", "medium_triplets", "hard_triplets"]
difficulty_labels = ["Easy", "Medium", "Hard"]
chance_level = 1 / 3

responses_dir = "../data/triplets_results/exp2/cleaned_results"
dqn_triplets_path = "../data/jsons/pong_final_triplet_exp.json"
output_dir = "../data/triplets_results/exp2/accuracy_results"
os.makedirs(output_dir, exist_ok=True)

# =========================
# LOAD DQN TRIPLETS
# =========================
with open(dqn_triplets_path, "r") as f:
    dqn_triplets_raw = json.load(f)

# Build lookup: game -> difficulty -> frozenset(clip_a, clip_b, clip_c) -> odd_clip
dqn_lookup = {}
for game in games:
    json_key = game_to_json_key[game]
    dqn_lookup[game] = {}
    for difficulty in difficulties:
        dqn_lookup[game][difficulty] = {}
        for triplet in dqn_triplets_raw[json_key][difficulty]:
            clip_set = frozenset(triplet)
            odd_clip = triplet[-1]
            dqn_lookup[game][difficulty][clip_set] = odd_clip

# =========================
# LOAD PARTICIPANT RESPONSES
# =========================
all_results = []

for game in games:
    responses_file = os.path.join(responses_dir, f"{game}_triplets_indexed_with_difficulty.csv")
    if not os.path.exists(responses_file):
        print(f"Responses file not found for {game}, skipping.")
        continue

    df = pd.read_csv(responses_file)

    for _, row in df.iterrows():
        participant_id = row["participant_id"]
        difficulty = row["difficulty"]
        similar1 = row["similar_clip_1_idx"]
        similar2 = row["similar_clip_2_idx"]
        odd_chosen = row["odd_clip_idx"]

        clip_set = frozenset({similar1, similar2, odd_chosen})

        dqn_odd = dqn_lookup[game][difficulty].get(clip_set, None)

        if dqn_odd is None:
            print(f"Warning: triplet {clip_set} not found in DQN lookup for {game} {difficulty}")
            continue

        correct = int(odd_chosen == dqn_odd)

        all_results.append({
            "game": game,
            "participant_id": participant_id,
            "difficulty": difficulty,
            "similar_clip_1_idx": similar1,
            "similar_clip_2_idx": similar2,
            "odd_chosen": odd_chosen,
            "dqn_odd": dqn_odd,
            "correct": correct
        })

results_df = pd.DataFrame(all_results)

# Add triplet_id column as frozenset of all three clips
results_df["triplet_id"] = results_df.apply(
    lambda row: frozenset({row["similar_clip_1_idx"], row["similar_clip_2_idx"], row["odd_chosen"]}),
    axis=1
)

results_csv = os.path.join(output_dir, "all_accuracy_results.csv")
results_df.to_csv(results_csv, index=False)
print(f"Saved all results to {results_csv}")

# =========================
# AGGREGATE ACCURACY
# =========================
summary_rows = []

for game in games:
    for difficulty in difficulty_order:
        subset = results_df[(results_df["game"] == game) & (results_df["difficulty"] == difficulty)]
        if subset.empty:
            continue
        n_total = len(subset)
        n_correct = subset["correct"].sum()
        accuracy = n_correct / n_total

        binom_result = binomtest(n_correct, n_total, chance_level, alternative="greater")
        pval = binom_result.pvalue

        print(f"{game} | {difficulty}: accuracy = {accuracy:.3f} ({n_correct}/{n_total}), p = {pval:.4f}")
        summary_rows.append({
            "game": game,
            "difficulty": difficulty,
            "n_total": n_total,
            "n_correct": n_correct,
            "accuracy": accuracy,
            "pvalue": pval
        })

summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(output_dir, "accuracy_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"\nSaved summary to {summary_csv}")

# =========================
# HUMAN-HUMAN CONSISTENCY
# =========================
consistency_rows = []

for game in games:
    for difficulty in difficulty_order:
        subset = results_df[(results_df["game"] == game) & (results_df["difficulty"] == difficulty)]
        if subset.empty:
            continue

        triplet_consistencies = []
        for clip_set, triplet_df in subset.groupby("triplet_id"):
            n_participants = len(triplet_df)
            vote_counts = triplet_df["odd_chosen"].value_counts()
            plurality = vote_counts.iloc[0] / n_participants
            triplet_consistencies.append(plurality)

        mean_consistency = np.mean(triplet_consistencies)
        std_consistency = np.std(triplet_consistencies)

        tstat, pval = ttest_1samp(triplet_consistencies, chance_level)

        print(f"{game} | {difficulty}: consistency = {mean_consistency:.3f} ± {std_consistency:.3f}, p = {pval:.4f}")
        consistency_rows.append({
            "game": game,
            "difficulty": difficulty,
            "mean_consistency": mean_consistency,
            "std_consistency": std_consistency,
            "pvalue": pval
        })

consistency_df = pd.DataFrame(consistency_rows)
consistency_csv = os.path.join(output_dir, "consistency_summary.csv")
consistency_df.to_csv(consistency_csv, index=False)
print(f"\nSaved consistency summary to {consistency_csv}")

# =========================
# PLOT BOTH ACCURACY AND CONSISTENCY SIDE BY SIDE
# =========================
fig, axes = plt.subplots(2, len(games), figsize=(5 * len(games), 10), sharey=True, squeeze=False)

for col, game in enumerate(games):
    # --- Row 0: Human-DQN Accuracy ---
    ax = axes[0, col]
    game_df = summary_df[summary_df["game"] == game]
    accuracies = []
    pvals_acc = []
    for difficulty in difficulty_order:
        row = game_df[game_df["difficulty"] == difficulty]
        accuracies.append(row["accuracy"].values[0] if not row.empty else 0)
        pvals_acc.append(row["pvalue"].values[0] if not row.empty else 1)

    bars = ax.bar(difficulty_labels, accuracies, color=["#2ecc71", "#f39c12", "#e74c3c"], width=0.5)
    ax.axhline(chance_level, color="black", linestyle="--", linewidth=1.2, label="Chance (1/3)")
    ax.set_title(f"{game.capitalize()}")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Human-DQN Agreement" if col == 0 else "")
    for bar, pval in zip(bars, pvals_acc):
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, sig,
                ha="center", va="bottom", fontsize=12)
    if col == 0:
        ax.legend()

    # --- Row 1: Human-Human Consistency ---
    ax = axes[1, col]
    game_df_c = consistency_df[consistency_df["game"] == game]
    consistencies = []
    pvals_con = []
    for difficulty in difficulty_order:
        row = game_df_c[game_df_c["difficulty"] == difficulty]
        consistencies.append(row["mean_consistency"].values[0] if not row.empty else 0)
        pvals_con.append(row["pvalue"].values[0] if not row.empty else 1)

    bars = ax.bar(difficulty_labels, consistencies, color=["#2ecc71", "#f39c12", "#e74c3c"], width=0.5)
    ax.axhline(chance_level, color="black", linestyle="--", linewidth=1.2, label="Chance (1/3)")
    ax.set_xlabel("Difficulty")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Human-Human Consistency" if col == 0 else "")
    for bar, pval in zip(bars, pvals_con):
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, sig,
                ha="center", va="bottom", fontsize=12)
    if col == 0:
        ax.legend()

plt.suptitle("Human-DQN Agreement and Human-Human Consistency by Difficulty", fontsize=14)
plt.tight_layout()
plot_file = os.path.join(output_dir, "accuracy_and_consistency_by_difficulty.png")
plt.savefig(plot_file, dpi=300)
plt.close()
print(f"Saved combined plot to {plot_file}")