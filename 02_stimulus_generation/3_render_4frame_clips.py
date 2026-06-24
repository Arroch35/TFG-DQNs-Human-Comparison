import os
import cv2
import imageio
import numpy as np

from src.config import GAMES, get_path
from src.utils import dqn_preprocess_from_16_frames

# ============================================================
# PROCESSING LOOP
# ============================================================

def process_npy_folder(input_folder, output_folder, fps=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    npy_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".npy")]

    if not npy_files:
        print("No .npy files found in the source folder.")
        return

    for npy_name in npy_files:
        input_path = os.path.join(input_folder, npy_name)
        video_name = npy_name.rsplit('.', 1)[0] + ".mp4"
        output_path = os.path.join(output_folder, video_name)

        print(f"Processing: {npy_name}...")

        try:
            frames_16 = np.load(input_path)

            # Output 'stack' is (4, 84, 84) float32 [0,1]
            stack = dqn_preprocess_from_16_frames(frames_16, human=True)

            video_data = (stack * 255).astype(np.uint8)

            with imageio.get_writer(output_path, fps=fps, macro_block_size=None) as writer:
                for i in range(video_data.shape[0]):
                    writer.append_data(video_data[i])

            print(f"Successfully saved 4-frame video to: {output_path}")

        except Exception as e:
            print(f"Error processing {npy_name}: {e}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    for game in GAMES:
        input_folder  = get_path("arrays_pool25_game", game=game)   
        output_folder = get_path("clips_pool25_game",  game=game)
        os.makedirs(output_folder, exist_ok=True)
        process_npy_folder(str(input_folder), str(output_folder), fps=2)
