import os
import re
import glob
import gc
from collections import deque

import numpy as np
import imageio

from src.config import GAME_TO_GYM_ID, GYM_ID_TO_GAME, REPR, get_path, ensure

# ============================================================
# CONFIG
# ============================================================

# ---- Gameplay identity ----
SUBJECT_ID = "sub_training_set_pca"
GYM_ID = "SpaceInvadersNoFrameskip-v4"
GAME = GYM_ID_TO_GAME[GYM_ID]             # -> "spaceinvaders"
BLOCK_INDEX = 1

# ---- Paths (from config) ----
FRAMES_DIR = get_path("recordings_pca")         
CLIPS_ROOT = str(ensure("clips_pca")) 

# ---- Clip settings (frame count from REPR config) ----
NUM_FRAMES = REPR["frame_stack"] * 4      # 16 frames (4 stacks × 4)
CLIP_DURATION_SECONDS = 2.0
STEP_FRAMES = 50
MIN_END_FRAME = 1000

GC_EVERY_N_CLIPS = 25

# ============================================================
# HELPERS
# ============================================================

def sanitize_game_name(gym_id: str) -> str:
    """Map a gym ID to the canonical short name used in folder paths."""
    return GYM_ID_TO_GAME.get(gym_id, re.sub(r'[^a-zA-Z0-9_-]', '_', gym_id.lower()))


def find_gameplay_chunks(frames_dir, subject_id, gym_id, block_index):
    """
    Find all chunk files corresponding to one gameplay:
    {SUBJECT_ID}_{GYM_ID}_block{BLOCK_INDEX}_chunkXXXX.npz
    """
    pattern = os.path.join(
        frames_dir,
        f"{subject_id}_{gym_id}_block{block_index}_chunk*.npz"
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


def save_clip(frames, output_path, duration):
    """Save a clip as mp4."""
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
    gym_id,
    block_index,
    num_frames,
    step_frames,
    duration,
    min_end_frame=0,
    gc_every_n_clips=25
):
    frames_buffer = deque(maxlen=num_frames)

    global_frame_idx = -1
    saved_count = 0

    for chunk_path in chunk_paths:
        with np.load(chunk_path) as data:
            frames = data["frames"]

            for i in range(len(frames)):
                global_frame_idx += 1

                frames_buffer.append(frames[i])

                if global_frame_idx < min_end_frame:
                    continue

                if len(frames_buffer) < num_frames:
                    continue

                if (global_frame_idx - min_end_frame) % step_frames != 0:
                    continue

                clip_frames = np.array(frames_buffer)

                base_name = (
                    f"{subject_id}_{gym_id}_block{block_index}"
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
    chunk_paths = find_gameplay_chunks(
        FRAMES_DIR,
        SUBJECT_ID,
        GYM_ID,
        BLOCK_INDEX
    )

    print("Found chunk files:")
    for p in chunk_paths:
        print(" -", os.path.basename(p))

    game_folder = sanitize_game_name(GYM_ID)
    output_dir = os.path.join(CLIPS_ROOT, game_folder)
    os.makedirs(output_dir, exist_ok=True)

    extract_clips_streaming(
        chunk_paths=chunk_paths,
        output_dir=output_dir,
        subject_id=SUBJECT_ID,
        gym_id=GYM_ID,
        block_index=BLOCK_INDEX,
        num_frames=NUM_FRAMES,
        step_frames=STEP_FRAMES,
        duration=CLIP_DURATION_SECONDS,
        min_end_frame=MIN_END_FRAME,
        gc_every_n_clips=GC_EVERY_N_CLIPS
    )


if __name__ == "__main__":
    main()
