import os
import pandas as pd

# =========================
# 1) CONFIGURATION
# =========================
games = ["pacman", "spaceinvaders", "pong"]
base_path_template = "../data/clips/{game_name}/buenos/" #"../data/test_16_clips/big_rdm/{game_name}/" #"../data/clips/{game_name}/buenos/pilot/"


# =========================
# 2) PROCESSING LOOP
# =========================
for game in games:
    # Format the path for the specific game
    folder_path = base_path_template.format(game_name=game)
    
    # Check if the folder actually exists to avoid errors
    if not os.path.exists(folder_path):
        print(f"Skipping {game}: Folder not found at {folder_path}")
        continue

    # Get all .mp4 files and sort them (to ensure index consistency)
    video_files = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
    video_files.sort()

    if not video_files:
        print(f"No .mp4 files found for {game}.")
        continue

    # Create a list of dictionaries for the mapping
    mapping_data = []
    for index, filename in enumerate(video_files):
        mapping_data.append({
            "clip_index": index,
            "clip_name": filename
        })

    # Convert to DataFrame
    df_map = pd.DataFrame(mapping_data)

    # =========================
    # 3) SAVE MAPPING CSV
    # =========================
    output_filename = f"../data/maps/buenos_25/{game}_clip_map.csv"
    os.makedirs("../data/maps/buenos_25/", exist_ok=True)
    df_map.to_csv(output_filename, index=False)
    
    print(f"Created {output_filename} with {len(df_map)} clips.")

print("\nAll mappings generated.")