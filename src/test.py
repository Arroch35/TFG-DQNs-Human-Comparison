import os

# Define the path to your clips
folder_path = r'C:\Users\arroc\OneDrive\Escritorio\Apuntes\UAB\4th_year\TFG\Code_repo\data\clips\spaceinvaders\buenos\pilot'

# List comprehension to filter for .mp4 files
# We use .lower() just in case some files use .MP4
video_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.mp4')]

# Print the results
print(f"Found {len(video_files)} .mp4 files:\n")
for name in video_files:
    print(name)

print(video_files)