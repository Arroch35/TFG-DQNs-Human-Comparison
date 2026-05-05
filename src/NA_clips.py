import os
import shutil

def organize_pilot_files(source_folder, search_folder):
    # 1. Get the list of .mp4 files from the first folder
    if not os.path.exists(source_folder):
        print(f"Source folder '{source_folder}' not found.")
        return

    mp4_names = [f.replace("_frames12", "") for f in os.listdir(source_folder) if f.lower().endswith(".mp4")]
    print(len(mp4_names))
    
    if not mp4_names:
        print("No .mp4 files found in the source folder to search for.")
        return

    # 2. Prepare the 'buenos_25' subfolder in folder2 #!BEFORE pilot
    pilot_folder = os.path.join(search_folder, "buenos_25") #!BEFORE pilot
    if not os.path.exists(pilot_folder):
        os.makedirs(pilot_folder)
        print(f"Created subfolder: {pilot_folder}")

    # 3. Search for these names in folder2 and move them
    moved_count = 0
    for name in mp4_names:
        target_path = os.path.join(search_folder, name)
        print(target_path)
        
        # Check if the file exists in the second folder
        if os.path.exists(target_path):
            destination = os.path.join(pilot_folder, name)
            
            # Move the file
            shutil.move(target_path, destination)
            print(f"Moved: {name} -> pilot/")
            moved_count += 1

    print(f"\nTask complete. Moved {moved_count} files to the 'pilot' folder.")

# --- Configuration ---
games=["pacman", "pong", "spaceinvaders"]
for game in games:
    folder1 = f"../data/clips/{game}/buenos/" # Where we get the names from
    folder2 = f"../data/test_16_clips/{game}" # Where we search and move files from
    organize_pilot_files(folder1, folder2)