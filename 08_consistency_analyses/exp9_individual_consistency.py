"""
exp9_individual_consistency.py
Compute per-participant consistency against the plurality vote,
run t-tests against chance, and flag outliers.
Currently scoped to pong only (extend GAMES to include others).
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from src.config import GAME_TO_GYM_ID, get_path, ensure

# =========================================================
# CONFIG
# =========================================================
GAMES      = ["pong"]   # extend as needed
CHANCE     = 1 / 3

DIFFICULTIES   = ["easy_triplets", "medium_triplets", "hard_triplets"]

# Paths
INDIVIDUAL_DIR = get_path("experiment_individual")                    
DQN_JSON = get_path("jsons_pong_triplets")              
OUTPUT_DIR = ensure("experiment_individual_accuracy")

# =========================================================
# LOAD DQN TRIPLETS → lookup: game × difficulty × clip_set → odd_clip
# =========================================================
with open(DQN_JSON, "r") as f:
    dqn_triplets_raw = json.load(f)

dqn_lookup = {}
for game in GAMES:
    gym_key = GAME_TO_GYM_ID[game]              # e.g. "PongNoFrameskip-v4"
    dqn_lookup[game] = {}
    for difficulty in DIFFICULTIES:
        dqn_lookup[game][difficulty] = {}
        for triplet in dqn_triplets_raw[gym_key][difficulty]:
            clip_set = frozenset(triplet)
            odd_clip = triplet[-1]
            dqn_lookup[game][difficulty][clip_set] = odd_clip

# =========================================================
# LOAD PARTICIPANT RESPONSES
# =========================================================
all_results = []
for game in GAMES:
    responses_file = INDIVIDUAL_DIR / f"{game}_triplets_indexed_with_difficulty.csv"
    if not responses_file.exists():
        print(f"Responses file not found for {game}, skipping.")
        continue

    df = pd.read_csv(responses_file)
    for _, row in df.iterrows():
        participant_id = row["participant_id"]
        difficulty     = row["difficulty"]
        similar1       = row["similar_clip_1_idx"]
        similar2       = row["similar_clip_2_idx"]
        odd_chosen     = row["odd_clip_idx"]
        clip_set       = frozenset({similar1, similar2, odd_chosen})

        dqn_odd = dqn_lookup[game][difficulty].get(clip_set)
        if dqn_odd is None:
            continue

        all_results.append({
            "game":              game,
            "participant_id":    participant_id,
            "difficulty":        difficulty,
            "similar_clip_1_idx": similar1,
            "similar_clip_2_idx": similar2,
            "odd_chosen":        odd_chosen,
            "dqn_odd":           dqn_odd,
            "correct":           int(odd_chosen == dqn_odd),
        })

results_df = pd.DataFrame(all_results)
results_df["triplet_id"] = results_df.apply(
    lambda row: frozenset({row["similar_clip_1_idx"], row["similar_clip_2_idx"], row["odd_chosen"]}),
    axis=1,
)

# =========================================================
# STEP 1 — Plurality vote per triplet
# =========================================================
plurality_votes = {}
for game in GAMES:
    plurality_votes[game] = {}
    for difficulty in DIFFICULTIES:
        subset = results_df[(results_df["game"] == game) & (results_df["difficulty"] == difficulty)]
        plurality_votes[game][difficulty] = {}
        for clip_set, triplet_df in subset.groupby("triplet_id"):
            plurality_votes[game][difficulty][clip_set] = triplet_df["odd_chosen"].value_counts().index[0]

# =========================================================
# STEP 2 — Per-participant consistency against plurality
# =========================================================
participant_rows = []
for game in GAMES:
    for difficulty in DIFFICULTIES:
        subset = results_df[(results_df["game"] == game) & (results_df["difficulty"] == difficulty)]
        if subset.empty:
            continue

        for pid, pdata in subset.groupby("participant_id"):
            matches = [
                int(row["odd_chosen"] == plurality_votes[game][difficulty].get(row["triplet_id"]))
                for _, row in pdata.iterrows()
                if row["triplet_id"] in plurality_votes[game][difficulty]
            ]
            if not matches:
                continue

            participant_rows.append({
                "game":           game,
                "participant_id": pid,
                "difficulty":     difficulty,
                "n_triplets":     len(matches),
                "n_consistent":   sum(matches),
                "consistency":    np.mean(matches),
            })

participant_df = pd.DataFrame(participant_rows)

# =========================================================
# STEP 3 — Summary stats per difficulty
# =========================================================
print("=" * 60)
print("PER-PARTICIPANT CONSISTENCY SUMMARY")
print("=" * 60)

for difficulty in DIFFICULTIES:
    label = difficulty.replace("_triplets", "").capitalize()
    sub   = participant_df[participant_df["difficulty"] == difficulty]
    if sub.empty:
        continue

    tstat, pval = ttest_1samp(sub["consistency"], CHANCE)
    print(f"\n--- {label} ---")
    print(f"  N participants : {len(sub)}")
    print(f"  Mean           : {sub['consistency'].mean():.3f}")
    print(f"  Std            : {sub['consistency'].std():.3f}")
    print(f"  t({len(sub)-1})          : {tstat:.3f},  p = {pval:.6f}")
    print(f"  Min            : {sub['consistency'].min():.3f}  "
          f"(participant: {sub.loc[sub['consistency'].idxmin(), 'participant_id']})")
    print(f"  Max            : {sub['consistency'].max():.3f}  "
          f"(participant: {sub.loc[sub['consistency'].idxmax(), 'participant_id']})")
    print(f"  Median         : {sub['consistency'].median():.3f}")
    print(f"\n  Per-participant breakdown:")
    for _, r in sub.sort_values("consistency").iterrows():
        flag = " <-- LOW" if r["consistency"] < CHANCE else ""
        print(f"    {str(r['participant_id']):>20s}  {r['consistency']:.3f}  "
              f"({int(r['n_consistent'])}/{int(r['n_triplets'])} consistent){flag}")

# =========================================================
# STEP 4 — Save
# =========================================================
out_path = OUTPUT_DIR / "per_participant_consistency.csv"
participant_df.to_csv(out_path, index=False)
print(f"\nSaved per-participant consistency → {out_path}")

# =========================================================
# STEP 5 — Outlier check (below Q1 − 1.5×IQR per difficulty)
# =========================================================
print("\n" + "=" * 60)
print("OUTLIER CHECK (below Q1 - 1.5*IQR per difficulty)")
print("=" * 60)

for difficulty in DIFFICULTIES:
    label = difficulty.replace("_triplets", "").capitalize()
    sub   = participant_df[participant_df["difficulty"] == difficulty]
    q1, q3 = sub["consistency"].quantile(0.25), sub["consistency"].quantile(0.75)
    iqr    = q3 - q1
    fence  = q1 - 1.5 * iqr
    outliers = sub[sub["consistency"] < fence]

    print(f"\n{label}: Q1={q1:.3f}, Q3={q3:.3f}, IQR={iqr:.3f}, fence={fence:.3f}")
    if outliers.empty:
        print("  No outliers.")
    else:
        for _, r in outliers.iterrows():
            print(f"  OUTLIER: {r['participant_id']}  consistency={r['consistency']:.3f}")
