import os
import cv2
import imageio
import numpy as np

#!Usar web/5_... antes de este fichero

# ============================================================
# 1. YOUR PREPROCESSING FUNCTION
# ============================================================
def dqn_preprocess_from_16_frames(frames_16):
    """
    frames_16: np.array of shape (16, H, W, 3), dtype uint8

    Returns:
        stack: np.array of shape (4, 84, 84), float32 in [0,1]
    """
    print(frames_16.shape)
    assert frames_16.shape[0] == 16, "Expected 16 frames"

    processed_frames = []

    # Indices corresponding to frame skipping = 4
    selected_indices = [3, 7, 11, 15]

    for t in selected_indices:
        # Max-pooling over last 2 frames
        pooled = np.maximum(frames_16[t], frames_16[t - 1])

        # Grayscale
        #gray = cv2.cvtColor(pooled, cv2.COLOR_RGB2GRAY)

        # Resize to 84x84
        #resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

        processed_frames.append(pooled)

    stack = np.stack(processed_frames, axis=0).astype(np.float32) / 255.0

    return stack

# ============================================================
# 2. THE PROCESSING LOOP
# ============================================================
def process_npy_folder(input_folder, output_folder, fps=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Search for .npy files instead of .mp4
    npy_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".npy")]

    if not npy_files:
        print("No .npy files found in the source folder.")
        return

    for npy_name in npy_files:
        input_path = os.path.join(input_folder, npy_name)
        # We change the extension to .mp4 for the output file
        video_name = npy_name.rsplit('.', 1)[0] + ".mp4"
        output_path = os.path.join(output_folder, video_name)
        
        print(f"Processing: {npy_name}...")

        try:
            # Load the numpy array
            frames_16 = np.load(input_path)

            # Apply your preprocessing
            # Output 'stack' is (4, 84, 84) float32 [0,1]
            stack = dqn_preprocess_from_16_frames(frames_16)

            # IMPORTANT: imageio expects uint8 [0-255] and (H, W, C) or (H, W) for video
            # We convert float32 [0,1] back to uint8 [0,255]
            video_data = (stack * 255).astype(np.uint8)

            # Save the 4 frames as an mp4
            with imageio.get_writer(output_path, fps=fps, macro_block_size=None) as writer:
                for i in range(video_data.shape[0]):
                    # video_data[i] is (84, 84)
                    writer.append_data(video_data[i])
            
            print(f"Successfully saved 4-frame video to: {output_path}")

        except Exception as e:
            print(f"Error processing {npy_name}: {e}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    SOURCE_DIR = "data/test_16_arrays"  # Folder with your .npy files
    TARGET_DIR = "data/test_16_clips" # Where the 4-frame clips go

    games = ["pacman", 
         "pong", 
         "spaceinvaders"
         ]
    for game in games:
        input_folder = os.path.join(SOURCE_DIR, "buenos_25", game) #, "buenos", "pilot"
        output_folder = os.path.join(TARGET_DIR, game, "buenos_25", "human_dqn_visualitzation") #"../data/frame_arrays"
        os.makedirs(output_folder, exist_ok=True)
        process_npy_folder(input_folder, output_folder, fps=2)