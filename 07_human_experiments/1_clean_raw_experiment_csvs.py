"""
1_clean_raw_experiment_csvs.py
Merge per-participant raw CSVs from the experiment download folder into
two cleaned tables: triplet results and survey responses.
"""
import glob
import os
import pandas as pd

from src.config import get_path, ensure

# =========================================================
# PATHS
# =========================================================
INPUT_FOLDER  = get_path("experiment_raw")       # data/triplets_results/final_experiment
OUTPUT_FOLDER = ensure("experiment_cleaned")     # data/triplets_results/final_experiment/cleaned_results

OUTPUT_TRIPLETS_FILE = OUTPUT_FOLDER / "all_participants_triplets.csv"
OUTPUT_SURVEY_FILE   = OUTPUT_FOLDER / "all_participants_survey.csv"

# =========================================================
# DISCOVER FILES
# =========================================================
all_files = glob.glob(str(INPUT_FOLDER / "*.csv"))

if not all_files:
    print(f"No CSV files found in {INPUT_FOLDER}. Please check the path!")
else:
    print(f"Found {len(all_files)} files. Starting merge...")

# =========================================================
# PROCESSING LOOP
# =========================================================
all_triplets_list = []
all_survey_list   = []

TRIPLET_COLS = [
    "game_name", "trial_in_game", "total_trials_in_game", "difficulty",
    "clip_1", "clip_2", "clip_3", "chosen_position",
    "chosen_clip", "similar_clip_1", "similar_clip_2",
    "odd_clip", "rt", "screen_w", "screen_h", "is_mobile",
]
SURVEY_COLS = [
    "video_game_frequency", "prior_game_familiarity",
    "clip_interpretation_difficulty", "decision_confidence",
    "perceived_experiment_length",
]

for file_path in all_files:
    participant_id = os.path.basename(file_path).replace(".csv", "")
    print(f"Processing {file_path} → participant {participant_id}...")

    df = pd.read_csv(file_path)

    # ── Triplets ──────────────────────────────────────────
    triplets_df = df[df["task"] == "odd_one_out"].copy()
    if not triplets_df.empty:
        triplets_df = triplets_df[[c for c in TRIPLET_COLS if c in triplets_df.columns]].copy()
        triplets_df.insert(0, "participant_id", participant_id)
        all_triplets_list.append(triplets_df)

    # ── Survey ────────────────────────────────────────────
    likert_df = df[df["task"] == "post_survey_likert"].copy()
    if not likert_df.empty:
        likert_df = likert_df[[c for c in SURVEY_COLS if c in likert_df.columns]].copy()
        likert_df.insert(0, "participant_id", participant_id)
        all_survey_list.append(likert_df)

# =========================================================
# SAVE
# =========================================================
if all_triplets_list:
    final_triplets_df = pd.concat(all_triplets_list, ignore_index=True)
    final_triplets_df.to_csv(OUTPUT_TRIPLETS_FILE, index=False)
    print(f"Saved {len(final_triplets_df)} rows → {OUTPUT_TRIPLETS_FILE}")

if all_survey_list:
    final_survey_df = pd.concat(all_survey_list, ignore_index=True)
    final_survey_df.to_csv(OUTPUT_SURVEY_FILE, index=False)
    print(f"Saved {len(final_survey_df)} rows → {OUTPUT_SURVEY_FILE}")

print("\nProcessing complete.")
