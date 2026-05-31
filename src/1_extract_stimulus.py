import os
import re
import glob
import gc
from collections import deque

import numpy as np
import imageio

# ============================================================
# CONFIG
# ============================================================

# ---- Gameplay identity ----
SUBJECT_ID = "sub_training_set_pca"
GAME = "SpaceInvadersNoFrameskip-v4"   # e.g. "Pong-v5", "MsPacman-v5", "SpaceInvadersNoFrameskip-v4" data/sub01_MsPacman-v5_block1.npz
BLOCK_INDEX = 1

# ---- Paths ----
FRAMES_DIR = "data/human_plays/pca_training" #"data/human_plays/big_rdm" #/human_plays
CLIPS_ROOT = "data/test_16_clips/pca_training" #"data/human_plays/clips"

# ---- Clip settings ----
NUM_FRAMES = 16
CLIP_DURATION_SECONDS = 2.0
STEP_FRAMES = 50          # extract one clip every 100 frames
MIN_END_FRAME = 1000       # skip very early gameplay if desired

# Optional: force garbage collection every N saved clips
GC_EVERY_N_CLIPS = 25

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


def find_gameplay_chunks(frames_dir, subject_id, game, block_index):
    """
    Find all chunk files corresponding to one gameplay:
    {SUBJECT_ID}_{GAME}_block{BLOCK_INDEX}_chunkXXXX.npz
    """
    pattern = os.path.join(
        frames_dir,
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

    files = sorted(files, key=chunk_sort_key)
    return files


def save_clip(frames, output_path, duration):
    """
    Save a clip as mp4.
    """
    fps = len(frames) / duration

    with imageio.get_writer(output_path, fps=fps, macro_block_size=None) as writer:
        for frame in frames:
            writer.append_data(frame)


# ============================================================
# MAIN EXTRACTION LOGIC (MEMORY-SAFE)
# ============================================================

def extract_clips_streaming(
    chunk_paths,
    output_dir,
    subject_id,
    game,
    block_index,
    num_frames,
    step_frames,
    duration,
    min_end_frame=0,
    gc_every_n_clips=25
):
    from collections import deque

    frames_buffer = deque(maxlen=num_frames)

    global_frame_idx = -1
    saved_count = 0

    for chunk_path in chunk_paths:
            with np.load(chunk_path) as data:
                frames = data["frames"]

                for i in range(len(frames)):
                    global_frame_idx += 1

                    # ----------------------------
                    # NATURAL FRAME BUFFER
                    # ----------------------------
                    # We always append frames to the buffer so that when we reach 
                    # min_end_frame, we have the history needed to make a clip.
                    frames_buffer.append(frames[i])

                    # 1. Only start considering clips once we hit your specific end frame
                    if global_frame_idx < min_end_frame:
                        continue

                    # 2. Ensure we have enough frames in the buffer to actually save
                    if len(frames_buffer) < num_frames:
                        continue

                    # 3. Check if this frame follows the stepping interval starting FROM min_end_frame
                    # This ensures: 1011, 1111, 1211, etc.
                    if (global_frame_idx - min_end_frame) % step_frames != 0:
                        continue

                    # =========================================================
                    # SAVE (ALIGNED BY SAME END FRAME)
                    # =========================================================
                    clip_frames = np.array(frames_buffer)
                    
                    base_name = (
                        f"{subject_id}_{game}_block{block_index}"
                        f"_end{global_frame_idx:06d}"
                    )

                    save_clip(
                        clip_frames,
                        os.path.join(output_dir, base_name + ".mp4"),
                        duration
                    )

                    saved_count += 1
                    print(f"Saved clip #{saved_count:03d} (end={global_frame_idx})")

                    if saved_count % gc_every_n_clips == 0:
                        gc.collect()


# ============================================================
# MAIN
# ============================================================

def main():
    # 1) Find all chunks for this gameplay
    chunk_paths = find_gameplay_chunks(
        FRAMES_DIR,
        SUBJECT_ID,
        GAME,
        BLOCK_INDEX
    )

    print("Found chunk files:")
    for p in chunk_paths:
        print(" -", os.path.basename(p))

    # 2) Create output folder for this game
    game_folder = sanitize_game_name(GAME)
    output_dir = os.path.join(CLIPS_ROOT, game_folder)
    os.makedirs(output_dir, exist_ok=True)

    # 3) Extract clips in streaming mode (memory-safe)
    extract_clips_streaming(
        chunk_paths=chunk_paths,
        output_dir=output_dir,
        subject_id=SUBJECT_ID,
        game=GAME,
        block_index=BLOCK_INDEX,
        num_frames=NUM_FRAMES,
        step_frames=STEP_FRAMES,
        duration=CLIP_DURATION_SECONDS,
        min_end_frame=MIN_END_FRAME,
        gc_every_n_clips=GC_EVERY_N_CLIPS
    )


if __name__ == "__main__":
    main()