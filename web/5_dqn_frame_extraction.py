import os
import cv2
import numpy as np

# =========================
# CONFIGURATION
# =========================
games = ["pacman", 
         "pong", 
         "spaceinvaders"
         ]
base_input_folder = "../data/test_16_clips/pca_training/" #"../data/test_16_clips/" #"../data/test_16_clips/big_rdm" #"../data/clips"
n_frames_to_extract = 16

# =========================
# PROCESS ALL GAMES
# =========================
for game in games:
    input_folder = os.path.join(base_input_folder, game) #, "buenos", "pilot" , game, "buenos_25"
    output_folder = os.path.join("../data/test_16_arrays/pca_training", game) #"../data/frame_arrays" buenos_25
    os.makedirs(output_folder, exist_ok=True)

    clip_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]
    #print(clip_files) #orden correcto!
    print(f"\nProcessing {len(clip_files)} clips for game: {game}")

    for clip_file in clip_files:
        clip_path = os.path.join(input_folder, clip_file)

        # Open video
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            print(f"Cannot open {clip_file}, skipping.")
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Seek to last frames
        start_frame = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for i in range(n_frames_to_extract):
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {start_frame + i}, skipping clip.")
                frames = []
                break
            # Convert to RGB (OpenCV reads as BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) != n_frames_to_extract:
            continue  # Skip incomplete clips

        # Convert to numpy array: shape (n_frames, height, width, channels)
        frames_array = np.stack(frames, axis=0).astype(np.uint8)

        # Save as .npy file
        base_name = os.path.splitext(clip_file)[0]
        out_file = os.path.join(output_folder, f"{base_name}.npy")
        np.save(out_file, frames_array)
        print(f"Saved frames array: {out_file}, shape: {frames_array.shape}")

print("\nAll games processed successfully.")