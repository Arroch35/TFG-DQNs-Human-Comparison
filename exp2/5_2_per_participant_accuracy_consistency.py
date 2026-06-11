import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import binomtest

# =========================
# CONFIGURATION
# =========================
games = ["pong"]
game_to_json_key = {"pong": "PongNoFrameskip-v4"}
difficulties = ["easy_triplets", "medium_triplets", "hard_triplets"]
difficulty_order = ["easy_triplets", "medium_triplets", "hard_triplets"]
difficulty_labels = {"easy_triplets": "Easy", "medium_triplets": "Medium", "hard_triplets": "Hard"}
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

dqn_lookup = {}
for game in games:
    json_key = game_to_json_key[game]
    dqn_lookup[game] = {}
    for difficulty in difficulties:
        dqn_lookup[game][difficulty] = {}
        for triplet in dqn_triplets_raw[json_key][difficulty]:
            clip_set = frozenset(triplet)
            dqn_lookup[game][difficulty][clip_set] = triplet[-1]

# =========================
# LOAD RESPONSES
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
            continue
        all_results.append({
            "game": game,
            "participant_id": participant_id,
            "difficulty": difficulty,
            "odd_chosen": odd_chosen,
            "dqn_odd": dqn_odd,
            "correct": int(odd_chosen == dqn_odd),
            "triplet_id": clip_set,
        })

results_df = pd.DataFrame(all_results)

# =========================
# COMPUTE PLURALITY VOTE PER TRIPLET (leave-one-out)
# =========================
def get_plurality_loo(triplet_df, participant_id):
    """Majority vote excluding the current participant."""
    others = triplet_df[triplet_df["participant_id"] != participant_id]
    if others.empty:
        return None
    return others["odd_chosen"].value_counts().index[0]

# =========================
# PER-PARTICIPANT ACCURACY + CONSISTENCY
# =========================
participant_rows = []

for game in games:
    for difficulty in difficulty_order:
        subset = results_df[(results_df["game"] == game) & (results_df["difficulty"] == difficulty)]
        if subset.empty:
            continue

        # Pre-group by triplet for LOO efficiency
        triplet_groups = {clip_set: grp for clip_set, grp in subset.groupby("triplet_id")}

        for pid, pdata in subset.groupby("participant_id"):
            acc_list = []
            con_list = []

            for _, row in pdata.iterrows():
                clip_set = row["triplet_id"]
                triplet_df = triplet_groups[clip_set]

                # Accuracy vs DQN
                acc_list.append(row["correct"])

                # Consistency vs majority (leave-one-out)
                plurality = get_plurality_loo(triplet_df, pid)
                if plurality is not None:
                    con_list.append(int(row["odd_chosen"] == plurality))

            n = len(acc_list)
            participant_rows.append({
                "game": game,
                "participant_id": pid,
                "difficulty": difficulty,
                "n_triplets": n,
                # Accuracy
                "n_correct": sum(acc_list),
                "accuracy": np.mean(acc_list),
                # Consistency
                "n_consistent": sum(con_list),
                "n_consistent_total": len(con_list),
                "consistency": np.mean(con_list) if con_list else np.nan,
                # Profile
                "profile": None,  # filled below
            })

participant_df = pd.DataFrame(participant_rows)

# =========================
# ASSIGN QUADRANT PROFILE
# =========================
def assign_profile(row):
    acc = row["accuracy"]
    con = row["consistency"]
    if acc >= chance_level and con >= chance_level:
        return "High accuracy / High consistency"
    elif acc >= chance_level and con < chance_level:
        return "High accuracy / Low consistency"
    elif acc < chance_level and con >= chance_level:
        return "Low accuracy / High consistency"
    else:
        return "Low accuracy / Low consistency"

participant_df["profile"] = participant_df.apply(assign_profile, axis=1)

# =========================
# PRINT COMBINED TABLE
# =========================
print("=" * 80)
print("PER-PARTICIPANT ACCURACY (vs DQN) × CONSISTENCY (vs majority, LOO)")
print("=" * 80)

for difficulty in difficulty_order:
    label = difficulty_labels[difficulty]
    sub = participant_df[participant_df["difficulty"] == difficulty].sort_values("accuracy", ascending=False)
    print(f"\n{'─'*80}")
    print(f"  {label.upper()}")
    print(f"{'─'*80}")
    print(f"  {'Participant':<25} {'Accuracy':>10} {'Consistency':>13} {'Profile'}")
    print(f"  {'─'*25} {'─'*10} {'─'*13} {'─'*35}")
    for _, r in sub.iterrows():
        acc_str = f"{r['accuracy']:.2f} ({int(r['n_correct'])}/{int(r['n_triplets'])})"
        con_str = f"{r['consistency']:.2f} ({int(r['n_consistent'])}/{int(r['n_consistent_total'])})"
        print(f"  {str(r['participant_id']):<25} {acc_str:>10} {con_str:>13}   {r['profile']}")

# =========================
# PROFILE SUMMARY COUNT
# =========================
print(f"\n{'='*80}")
print("PROFILE COUNTS PER DIFFICULTY")
print(f"{'='*80}")
profile_summary = participant_df.groupby(["difficulty", "profile"]).size().reset_index(name="count")
for difficulty in difficulty_order:
    label = difficulty_labels[difficulty]
    sub = profile_summary[profile_summary["difficulty"] == difficulty]
    print(f"\n  {label}:")
    for _, r in sub.iterrows():
        print(f"    {r['profile']}: {r['count']}")

# =========================
# SAVE CSV
# =========================
out_csv = os.path.join(output_dir, "per_participant_accuracy_consistency.csv")
participant_df.to_csv(out_csv, index=False)
print(f"\nSaved to {out_csv}")

# =========================
# PLOT: Scatter per difficulty (accuracy vs consistency)
# =========================
profile_colors = {
    "High accuracy / High consistency": "#2ecc71",
    "High accuracy / Low consistency":  "#3498db",
    "Low accuracy / High consistency":  "#f39c12",
    "Low accuracy / Low consistency":   "#e74c3c",
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

for ax, difficulty in zip(axes, difficulty_order):
    label = difficulty_labels[difficulty]
    sub = participant_df[participant_df["difficulty"] == difficulty]

    for _, r in sub.iterrows():
        color = profile_colors[r["profile"]]
        ax.scatter(r["accuracy"], r["consistency"], color=color, s=100, edgecolors="white", linewidths=0.5, zorder=3)
        ax.annotate(str(r["participant_id"]), (r["accuracy"], r["consistency"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7, alpha=0.8)

    # Chance lines
    ax.axvline(chance_level, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(chance_level, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    # Quadrant shading
    ax.axvspan(chance_level, 1.05, ymin=(chance_level - 0) / 1.05, ymax=1, alpha=0.04, color="#2ecc71")
    ax.axvspan(0, chance_level, ymin=(chance_level - 0) / 1.05, ymax=1, alpha=0.04, color="#f39c12")
    ax.axvspan(chance_level, 1.05, ymin=0, ymax=(chance_level) / 1.05, alpha=0.04, color="#3498db")
    ax.axvspan(0, chance_level, ymin=0, ymax=(chance_level) / 1.05, alpha=0.04, color="#e74c3c")

    ax.set_title(label, fontsize=13)
    ax.set_xlabel("Accuracy (vs DQN)", fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, chance_level, 0.5, 1.0])
    ax.set_xticklabels(["0", "1/3\n(chance)", "0.5", "1.0"], fontsize=8)
    ax.set_yticks([0, chance_level, 0.5, 1.0])
    ax.set_yticklabels(["0", "1/3\n(chance)", "0.5", "1.0"], fontsize=8)

axes[0].set_ylabel("Consistency (vs majority, LOO)", fontsize=11)

# Legend
legend_patches = [mpatches.Patch(color=c, label=l) for l, c in profile_colors.items()]
fig.legend(handles=legend_patches, loc="lower center", ncol=2, fontsize=9,
           bbox_to_anchor=(0.5, -0.12), frameon=True)

plt.suptitle("Per-Participant Accuracy vs Consistency — Pong", fontsize=14, y=1.02)
plt.tight_layout()
plot_file = os.path.join(output_dir, "per_participant_accuracy_vs_consistency.png")
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved scatter plot to {plot_file}")
