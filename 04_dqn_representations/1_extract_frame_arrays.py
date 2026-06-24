import os
import cv2
import numpy as np

from src.config import GAMES, REPR, get_path, ensure

# =========================
# CONFIGURATION
# =========================
N_FRAMES = REPR["frame_stack"] * 4      # 16  (4 stacks × 4 selected frames)

# =========================
# PROCESS ALL GAMES
# =========================
for game in GAMES:
    input_folder  = get_path("clips_pca", game=game) 
    output_folder = ensure("arrays_pca_game", game=game)

    clip_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]
    print(f"\nProcessing {len(clip_files)} clips for game: {game}")

    for clip_file in clip_files:
        clip_path = input_folder / clip_file

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            print(f"Cannot open {clip_file}, skipping.")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frames = []
        for i in range(N_FRAMES):
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {i}, skipping clip.")
                frames = []
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()

        if len(frames) != N_FRAMES:
            continue

        frames_array = np.stack(frames, axis=0).astype(np.uint8)

        base_name = os.path.splitext(clip_file)[0]
        out_file  = output_folder / f"{base_name}.npy"
        np.save(out_file, frames_array)
        print(f"Saved: {out_file}  shape: {frames_array.shape}")

print("\nAll games processed successfully.")
