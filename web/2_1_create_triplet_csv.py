import pandas as pd
import os

# =========================
# 1) CONFIGURATION
# =========================
triplets_file = "../data/triplets_results/final_experiment/cleaned_results/all_participants_triplets.csv" #"../data/cleaned_results/all_participants_triplets.csv"
games = ["pacman", "spaceinvaders", "pong"]
output_file = "../data/triplets_results/final_experiment/cleaned_results/all_participants_triplets_indexed.csv" # "../data/cleaned_results/all_participants_triplets_indexed.csv"

game_name_map = {
    "MsPacmanNoFrameskip-v4": "pacman",
    "SpaceInvadersNoFrameskip-v4": "spaceinvaders",
    "PongNoFrameskip-v4": "pong"
}

# =========================
# 2) LOAD MAPPINGS
# =========================
master_map = {}
for game in games:
    map_path = f"../data/maps/selected_15/{game}_clip_map.csv"
    if os.path.exists(map_path):
        map_df = pd.read_csv(map_path)
        master_map[game] = pd.Series(map_df.clip_index.values, index=map_df.clip_name).to_dict()
        print(f"Loaded mapping for {game}.")
    else:
        print(f"Warning: {map_path} not found.")


# =========================
# 3) MAPPING LOGIC
# =========================
def get_clip_index(row, col_name):
    # Translate technical name to short name (e.g., PongNoFrameskip-v4 -> pong)
    raw_game = row['game_name']
    short_game = game_name_map.get(raw_game, raw_game) 
    
    clip_name = str(row[col_name])
    
    # Try finding the clip in the appropriate map
    if short_game in master_map:
        # Check for exact match
        if clip_name in master_map[short_game]:
            return master_map[short_game][clip_name]
        
        # Check if index exists without path (just in case)
        base_name = os.path.basename(clip_name)
        if base_name in master_map[short_game]:
            return master_map[short_game][base_name]
            
    return None

# =========================
# 4) RUN, SPLIT & SAVE
# =========================
if os.path.exists(triplets_file):
    df = pd.read_csv(triplets_file)
    cols_to_map = ["similar_clip_1", "similar_clip_2", "odd_clip"]
    
    # Generate the index columns for the whole dataframe first
    for col in cols_to_map:
        print(f"Indexing {col}...")
        df[f"{col}_idx"] = df.apply(lambda row: get_clip_index(row, col), axis=1)

    # We need 'game_name' to do the splitting, then we'll drop it if needed
    output_columns = ["participant_id", "similar_clip_1_idx", "similar_clip_2_idx", "odd_clip_idx"]

    # Loop through each game and save a separate file
    for tech_name, short_name in game_name_map.items():
        # Filter rows for this specific game
        game_df = df[df["game_name"] == tech_name].copy()
        
        if not game_df.empty:
            # Select only the relevant columns
            final_game_df = game_df[[col for col in output_columns if col in game_df.columns]]
            
            # Create a specific filename (e.g., ../data/cleaned_results/pacman_triplets_indexed.csv)
            game_output_path = f"../data/triplets_results/final_experiment/cleaned_results/{short_name}_triplets_indexed.csv" #f"../data/cleaned_results/{short_name}_triplets_indexed.csv"
            
            os.makedirs(os.path.dirname(game_output_path), exist_ok=True)
            final_game_df.to_csv(game_output_path, index=False)
            
            # Validation for this game
            missing = final_game_df["odd_clip_idx"].isna().sum()
            print(f"--- {short_name.upper()} ---")
            print(f"Saved to: {game_output_path}")
            if missing > 0:
                print(f"Warning: {missing} rows unmapped in {short_name}.")
            else:
                print(f"All {len(final_game_df)} rows indexed successfully.")
        else:
            print(f"No data found for {short_name}, skipping file creation.")

    print("\nProcessing complete.")