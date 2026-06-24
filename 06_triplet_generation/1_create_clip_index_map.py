"""
3_create_clip_index_map.py
Build a clip_index → clip_name CSV for each game from the clips that were
copied into the triplet-visualisation folder by script 1.
"""
import os
import pandas as pd

from src.config import GAMES, REFERENCE_SEED, get_path, ensure

# =========================================================
# CONFIG
# =========================================================
SEED = REFERENCE_SEED   # "seed_42"

# Source folder: clips copied by 1_score_and_bucket_triplets.py
# Suggested addition to config.py PATHS:
#   "triplet_viz": DATA / "triplet_visualization_subset" / "selected_15" / "{seed}" / "filtered_all_difficulties",
# Until then, derived manually:
from src.config import DATA
def get_triplet_viz_clips(seed, game):
    return DATA / "triplet_visualization_subset" / "selected_15" / seed / "filtered_all_difficulties" / game / "clips"

# =========================================================
# MAIN
# =========================================================
for game in GAMES:
    folder_path = get_triplet_viz_clips(SEED, game)

    if not folder_path.exists():
        print(f"Skipping {game}: folder not found at {folder_path}")
        continue

    video_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".mp4"))

    if not video_files:
        print(f"No .mp4 files found for {game}.")
        continue

    df_map = pd.DataFrame([
        {"clip_index": idx, "clip_name": filename}
        for idx, filename in enumerate(video_files)
    ])

    # Output path already in config: "clip_maps" → data/maps/selected_15/{game}_clip_map.csv
    output_path = get_path("maps_selected15_game", game=game)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_map.to_csv(output_path, index=False)

    print(f"Created {output_path} with {len(df_map)} clips.")

print("\nAll mappings generated.")
