import pandas as pd
import glob
import os

# =========================
# 1) CONFIGURATION & PATHS
# =========================
input_folder = "../data/triplets_results/final_experiment/" #"../data/triplets_results/final_experiment/" #"../data/triplets_results/" # En este file pego los resultados descargados del experimento
output_folder= "../data/triplets_results/final_experiment/cleaned_results/" #"../data/triplets_results/final_experiment/cleaned_results/" #"../data/cleaned_results/"
output_triplets_file = "all_participants_triplets.csv"
output_survey_file = "all_participants_survey.csv"

os.makedirs(output_folder, exist_ok=True)

# Get a list of all CSV files in the folder
all_files = glob.glob(os.path.join(input_folder, "*.csv"))

if not all_files:
    print(f"No CSV files found in {input_folder}. Please check the path!")
else:
    print(f"Found {len(all_files)} files. Starting merge...")

# Lists to hold dataframes for concatenation
all_triplets_list = []
all_survey_list = []

# =========================
# 2) PROCESSING LOOP
# =========================
for file_path in all_files:
    # Load individual participant file
    df = pd.read_csv(file_path)
    
    # Extract a Participant ID from the filename (e.g., 'P01' from 'P01.csv')
    participant_id = os.path.basename(file_path).replace(".csv", "")
    
    # --- Process Triplet Results ---
    print(f"Processing {file_path} for participant {participant_id}...")
    triplets_df = df[df["task"] == "odd_one_out"].copy()
    if not triplets_df.empty:
        # Define important columns
        cols_to_keep = [
            "game_name", "trial_in_game", "total_trials_in_game", "difficulty",
            "clip_1", "clip_2", "clip_3", "chosen_position",
            "chosen_clip", "similar_clip_1", "similar_clip_2",
            "odd_clip", "rt", "screen_w", "screen_h", "is_mobile"
        ]
        
        # Keep only what exists in the df (prevents errors if a column is missing)
        triplets_df = triplets_df[[c for c in cols_to_keep if c in triplets_df.columns]].copy()
        
        # Inject Participant ID at the start
        triplets_df.insert(0, "participant_id", participant_id)
        all_triplets_list.append(triplets_df)

    # --- Process Survey Results ---
    likert_df = df[df["task"] == "post_survey_likert"].copy()
    if not likert_df.empty:
        survey_cols = [
            "video_game_frequency", "prior_game_familiarity", "clip_interpretation_difficulty", 
            "decision_confidence", "perceived_experiment_length", 
        ]
        likert_df = likert_df[[c for c in survey_cols if c in likert_df.columns]].copy()
        
        # Inject Participant ID at the start
        likert_df.insert(0, "participant_id", participant_id)
        all_survey_list.append(likert_df)

# =========================
# 3) CONCATENATE & SAVE
# =========================

# Combine all participants into one big table
if all_triplets_list:
    final_triplets_df = pd.concat(all_triplets_list, ignore_index=True)
    final_triplets_df.to_csv(output_folder+output_triplets_file, index=False)
    print(f"Success: {output_folder+output_triplets_file} saved with {len(final_triplets_df)} rows.")

if all_survey_list:
    final_survey_df = pd.concat(all_survey_list, ignore_index=True)
    final_survey_df.to_csv(output_folder+output_survey_file, index=False)
    print(f"Success: {output_folder+output_survey_file} saved with {len(final_survey_df)} rows.")

print("\nProcessing complete.")