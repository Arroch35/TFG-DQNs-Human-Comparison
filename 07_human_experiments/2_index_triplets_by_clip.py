"""
2_index_triplets_by_clip.py
Convert clip filenames in the merged triplets CSV into numeric indices
using the per-game clip maps, then split output by game.
"""
import os
import pandas as pd

from src.config import GAMES, GYM_ID_TO_GAME, get_path, ensure

# =========================================================
# PATHS
# =========================================================
# Input: merged triplets from script 1, stored in exp2_cleaned
TRIPLETS_FILE  = get_path("experiment_exp2") / "all_participants_triplets.csv"
OUTPUT_DIR     = ensure("experiment_exp2")   # data/triplets_results/exp2/cleaned_results

# GYM_ID_TO_GAME from config already maps gym IDs → short names, e.g.
# "PongNoFrameskip-v4" → "pong", used below as game_name_map
game_name_map = GYM_ID_TO_GAME   # {"PongNoFrameskip-v4": "pong", ...}

# =========================================================
# LOAD CLIP MAPS
# =========================================================
master_map = {}
for game in GAMES:
    map_path = get_path("maps_selected15_game", game=game)
    if map_path.exists():
        map_df = pd.read_csv(map_path)
        master_map[game] = pd.Series(map_df.clip_index.values, index=map_df.clip_name).to_dict()
        print(f"Loaded mapping for {game}.")
    else:
        print(f"Warning: {map_path} not found.")

# =========================================================
# MAPPING HELPER
# =========================================================
def get_clip_index(row, col_name):
    short_game = game_name_map.get(row["game_name"], row["game_name"])
    clip_name  = str(row[col_name])

    if short_game in master_map:
        if clip_name in master_map[short_game]:
            return master_map[short_game][clip_name]
        base_name = os.path.basename(clip_name)
        if base_name in master_map[short_game]:
            return master_map[short_game][base_name]

    return None

# =========================================================
# RUN, SPLIT & SAVE
# =========================================================
if not TRIPLETS_FILE.exists():
    print(f"Input file not found: {TRIPLETS_FILE}")
else:
    df = pd.read_csv(TRIPLETS_FILE)

    for col in ["similar_clip_1", "similar_clip_2", "odd_clip"]:
        print(f"Indexing {col}...")
        df[f"{col}_idx"] = df.apply(lambda row, c=col: get_clip_index(row, c), axis=1)

    OUTPUT_COLS = [
        "participant_id", "difficulty",
        "similar_clip_1_idx", "similar_clip_2_idx", "odd_clip_idx",
    ]

    for gym_id, short_name in game_name_map.items():
        game_df = df[df["game_name"] == gym_id].copy()

        if game_df.empty:
            print(f"No data for {short_name}, skipping.")
            continue

        final_df     = game_df[[c for c in OUTPUT_COLS if c in game_df.columns]]
        out_path     = OUTPUT_DIR / f"{short_name}_triplets_indexed_with_difficulty.csv"
        final_df.to_csv(out_path, index=False)

        missing = final_df["odd_clip_idx"].isna().sum()
        print(f"\n--- {short_name.upper()} ---")
        print(f"Saved {len(final_df)} rows → {out_path}")
        if missing:
            print(f"Warning: {missing} rows unmapped.")
        else:
            print("All rows indexed successfully.")

    print("\nProcessing complete.")
