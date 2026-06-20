import os
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp

# =========================
# CONFIGURATION — match original script
# =========================
games = ["pong"]
game_to_json_key = {"pong": "PongNoFrameskip-v4"}
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
# LOAD PARTICIPANT RESPONSES (same as original)
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
results_df["triplet_id"] = results_df.apply(
    lambda row: frozenset({row["similar_clip_1_idx"], row["similar_clip_2_idx"], row["odd_chosen"]}),
    axis=1
)

# =========================
# STEP 1: Compute plurality vote per triplet (same as original)
# =========================
plurality_votes = {}
for game in games:
    plurality_votes[game] = {}
    for difficulty in difficulty_order:
        subset = results_df[(results_df["game"] == game) & (results_df["difficulty"] == difficulty)]
        plurality_votes[game][difficulty] = {}
        for clip_set, triplet_df in subset.groupby("triplet_id"):
            vote_counts = triplet_df["odd_chosen"].value_counts()
            plurality_clip = vote_counts.index[0]  # the clip most participants chose
            plurality_votes[game][difficulty][clip_set] = plurality_clip

# =========================
# STEP 2: Per-participant consistency
# For each participant × difficulty, compute what fraction of their
# triplets matched the plurality vote.
# =========================
participant_rows = []
for game in games:
    for difficulty in difficulty_order:
        subset = results_df[(results_df["game"] == game) & (results_df["difficulty"] == difficulty)]
        if subset.empty:
            continue
        for pid, pdata in subset.groupby("participant_id"):
            matches = []
            for _, row in pdata.iterrows():
                clip_set = row["triplet_id"]
                plurality_choice = plurality_votes[game][difficulty].get(clip_set, None)
                if plurality_choice is None:
                    continue
                matched = int(row["odd_chosen"] == plurality_choice)
                matches.append(matched)
            if not matches:
                continue
            participant_rows.append({
                "game": game,
                "participant_id": pid,
                "difficulty": difficulty,
                "n_triplets": len(matches),
                "n_consistent": sum(matches),
                "consistency": np.mean(matches),
            })

participant_df = pd.DataFrame(participant_rows)

# =========================
# STEP 3: Summary stats per difficulty (how spread out are participants?)
# =========================
print("=" * 60)
print("PER-PARTICIPANT CONSISTENCY SUMMARY")
print("=" * 60)
for difficulty in difficulty_order:
    label = difficulty.replace("_triplets", "").capitalize()
    sub = participant_df[participant_df["difficulty"] == difficulty]
    if sub.empty:
        continue
    tstat, pval = ttest_1samp(sub["consistency"], chance_level)
    print(f"\n--- {label} ---")
    print(f"  N participants : {len(sub)}")
    print(f"  Mean           : {sub['consistency'].mean():.3f}")
    print(f"  Std            : {sub['consistency'].std():.3f}")
    print(f"  t({len(sub)-1})          : {tstat:.3f},  p = {pval:.6f}")
    print(f"  Min            : {sub['consistency'].min():.3f}  (participant: {sub.loc[sub['consistency'].idxmin(), 'participant_id']})")
    print(f"  Max            : {sub['consistency'].max():.3f}  (participant: {sub.loc[sub['consistency'].idxmax(), 'participant_id']})")
    print(f"  Median         : {sub['consistency'].median():.3f}")
    print(f"\n  Per-participant breakdown:")
    for _, r in sub.sort_values("consistency").iterrows():
        flag = " <-- LOW" if r["consistency"] < chance_level else ""
        print(f"    {str(r['participant_id']):>20s}  {r['consistency']:.3f}  ({int(r['n_consistent'])}/{int(r['n_triplets'])} consistent){flag}")

# =========================
# STEP 4: Save
# =========================
out_path = os.path.join(output_dir, "per_participant_consistency.csv")
participant_df.to_csv(out_path, index=False)
print(f"\nSaved per-participant consistency to {out_path}")

# =========================
# STEP 5: Flag outliers (>1.5 IQR below Q1 per difficulty)
# =========================
print("\n" + "=" * 60)
print("OUTLIER CHECK (below Q1 - 1.5*IQR per difficulty)")
print("=" * 60)
for difficulty in difficulty_order:
    label = difficulty.replace("_triplets", "").capitalize()
    sub = participant_df[participant_df["difficulty"] == difficulty]
    q1 = sub["consistency"].quantile(0.25)
    q3 = sub["consistency"].quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    outliers = sub[sub["consistency"] < lower_fence]
    print(f"\n{label}: Q1={q1:.3f}, Q3={q3:.3f}, IQR={iqr:.3f}, fence={lower_fence:.3f}")
    if outliers.empty:
        print("  No outliers.")
    else:
        for _, r in outliers.iterrows():
            print(f"  OUTLIER: {r['participant_id']}  consistency={r['consistency']:.3f}")
