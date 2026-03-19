import numpy as np
import imageio
import os

# ----------------------------
# CONSTANTS
# ----------------------------
# The index of the first frame you want to extract
START_FRAME_INDEX = 500  

# Total number of frames to extract (matching DQN input stack)
NUM_FRAMES = 4           

# Total duration of the resulting clip in seconds
CLIP_DURATION_SECONDS = 2.0 

# Path to your recorded data
DATA_PATH = "../data/sub01_Pong-v5_block1.npz"
OUTPUT_FILENAME = f"../data/clips/stimulus_frame_{START_FRAME_INDEX}.mp4"

# ----------------------------
# EXTRACTION LOGIC
# ----------------------------

def extract_stimulus(data_path, start_idx, num_frames, duration, output_path):
    # Load the compressed numpy file
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} not found.")
        return

    with np.load(data_path) as data:
        frames = data['frames']
        
        # Check if indices are within bounds
        if start_idx + num_frames > len(frames):
            print(f"Error: Requested {num_frames} frames starting at {start_idx}, "
                  f"but only {len(frames)} frames available.")
            return

        # Extract the sequence
        clip_frames = frames[start_idx : start_idx + num_frames]

    # Calculate FPS to achieve the desired duration
    # If 4 frames must last 2 seconds, FPS = 4 / 2 = 2.0
    calc_fps = num_frames / duration

    # Write to video file
    # We use macro_block_size=None to prevent issues with non-standard dimensions
    with imageio.get_writer(output_path, fps=calc_fps, macro_block_size=None) as writer:
        for frame in clip_frames:
            writer.append_data(frame)

    print(f"Successfully saved {num_frames} frames to {output_path}")
    print(f"Frame duration: {1/calc_fps}s | Total clip duration: {duration}s")

if __name__ == "__main__":
    extract_stimulus(
        DATA_PATH, 
        START_FRAME_INDEX, 
        NUM_FRAMES, 
        CLIP_DURATION_SECONDS, 
        OUTPUT_FILENAME
    )