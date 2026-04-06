import os
import re
import glob
import gc

import numpy as np
import imageio

# ============================================================
# CONFIG
# ============================================================

# ---- Gameplay identity ----
SUBJECT_ID = "sub_clipsEjemplo" 
GAME = "SpaceInvadersNoFrameskip-v4" # e.g. "Pong-v5", "MsPacman-v5", "SpaceInvadersNoFrameskip-v4"
BLOCK_INDEX = 1

# ---- Paths ----
DATA_DIR = "data"
OUTPUT_DIR = "data/clips"

# ---- Clip settings ----
START_FRAME = 10                 # global frame index to start from
NUM_FRAMES_TO_EXTRACT = 1000     # how many frames to include in the clip
CLIP_DURATION_SECONDS = 10.0    # final video duration in seconds

# Example:
# 300 frames / 10 sec = 30 fps
# 600 frames / 10 sec = 60 fps (looks faster if source gameplay is same pace)

# ============================================================
# HELPERS
# ============================================================

def sanitize_game_name(game_name: str) -> str:
    game_lower = game_name.lower()

    if "pacman" in game_lower:
        return "pacman"
    elif "pong" in game_lower:
        return "pong"
    elif "spaceinvaders" in game_lower:
        return "spaceinvaders"
    else:
        return re.sub(r'[^a-zA-Z0-9_-]', '_', game_lower)


def find_gameplay_chunks(data_dir, subject_id, game, block_index):
    """
    Find all chunk files corresponding to one gameplay:
    {SUBJECT_ID}_{GAME}_block{BLOCK_INDEX}_chunkXXXX.npz
    """
    pattern = os.path.join(
        data_dir,
        f"{subject_id}_{game}_block{block_index}_chunk*.npz"
    )

    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"No chunk files found for pattern:\n{pattern}"
        )

    def chunk_sort_key(path):
        fname = os.path.basename(path)
        match = re.search(r'chunk(\d+)', fname)

        if not match:
            raise ValueError(
                f"Unexpected file matched pattern but has no chunk number: {fname}"
            )

        return int(match.group(1))

    return sorted(files, key=chunk_sort_key)


def save_clip(frames, output_path, duration_seconds):
    """
    Save frames as one mp4 clip with fixed duration.
    Playback speed depends on how many frames are packed into that duration.
    """
    fps = len(frames) / duration_seconds

    with imageio.get_writer(output_path, fps=fps, macro_block_size=None) as writer:
        for frame in frames:
            writer.append_data(frame)


# ============================================================
# MAIN EXTRACTION LOGIC
# ============================================================

def extract_single_clip_streaming(
    chunk_paths,
    output_dir,
    subject_id,
    game,
    block_index,
    start_frame,
    num_frames_to_extract,
    clip_duration_seconds
):
    """
    Stream through chunk files and extract ONE clip:
    - starts at global frame index `start_frame`
    - contains exactly `num_frames_to_extract` frames
    - saved as a 10-second video (or chosen duration)
    """

    collected_frames = []
    global_frame_idx = -1
    end_frame = start_frame + num_frames_to_extract - 1

    print("\nStarting single clip extraction...")
    print(f" - Start frame: {start_frame}")
    print(f" - End frame:   {end_frame}")
    print(f" - Frames:      {num_frames_to_extract}")
    print(f" - Duration:    {clip_duration_seconds:.2f}s")
    print(f" - Output FPS:  {num_frames_to_extract / clip_duration_seconds:.2f}\n")

    for chunk_path in chunk_paths:
        fname = os.path.basename(chunk_path)

        with np.load(chunk_path) as data:
            if "frames" not in data:
                raise KeyError(f"'frames' key not found in {chunk_path}")

            frames = data["frames"]
            num_chunk_frames = len(frames)

            print(f"Processing {fname} ({num_chunk_frames} frames)...")

            for frame in frames:
                global_frame_idx += 1

                # Skip until start frame
                if global_frame_idx < start_frame:
                    continue

                # Stop after collecting enough frames
                if global_frame_idx > end_frame:
                    break

                collected_frames.append(frame)

            del frames
            gc.collect()

        # If done, stop reading more chunks
        if len(collected_frames) >= num_frames_to_extract:
            break

    if len(collected_frames) < num_frames_to_extract:
        raise ValueError(
            f"Not enough frames available.\n"
            f"Requested {num_frames_to_extract} frames starting at frame {start_frame}, "
            f"but only collected {len(collected_frames)}."
        )

    collected_frames = np.array(collected_frames)

    output_filename = (
        f"{subject_id}_{game}_block{block_index}"
        f"_start{start_frame:06d}_end{end_frame:06d}"
        f"_frames{num_frames_to_extract}_dur{int(clip_duration_seconds)}s.mp4"
    )
    output_path = os.path.join(output_dir, output_filename)

    save_clip(collected_frames, output_path, clip_duration_seconds)

    print(f"\nDone! Saved clip to:\n{output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    # 1) Find all chunks for this gameplay
    chunk_paths = find_gameplay_chunks(
        DATA_DIR,
        SUBJECT_ID,
        GAME,
        BLOCK_INDEX
    )

    print("Found chunk files:")
    for p in chunk_paths:
        print(" -", os.path.basename(p))

    # 2) Create output folder
    game_folder = sanitize_game_name(GAME)
    output_dir = os.path.join(OUTPUT_DIR, game_folder)
    os.makedirs(output_dir, exist_ok=True)

    # 3) Extract one single clip
    extract_single_clip_streaming(
        chunk_paths=chunk_paths,
        output_dir=output_dir,
        subject_id=SUBJECT_ID,
        game=GAME,
        block_index=BLOCK_INDEX,
        start_frame=START_FRAME,
        num_frames_to_extract=NUM_FRAMES_TO_EXTRACT,
        clip_duration_seconds=CLIP_DURATION_SECONDS
    )


if __name__ == "__main__":
    main()