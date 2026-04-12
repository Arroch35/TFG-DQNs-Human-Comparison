import gymnasium as gym
import ale_py
from gymnasium.utils.play import play
import numpy as np
import time
import os
import threading
import queue

# ----------------------------
# CONFIGURATION
# ----------------------------
GAMES = ["PongNoFrameskip-v4", "MsPacmanNoFrameskip-v4", "SpaceInvadersNoFrameskip-v4", "ALE/Galaxian-v5"]
GAME = GAMES[2]

SUBJECT_ID = "sub_clipsEjemplo"
BLOCK_INDEX = 1
BLOCK_DURATION_MINUTES = 30
FPS = 60
FRAME_SKIP = 1
SAVE_FOLDER = "../data/human_plays"
CHUNK_SIZE = 5000

os.makedirs(SAVE_FOLDER, exist_ok=True)

# ----------------------------
# AUTOMATIC CONTROLS
# ----------------------------
KEYS_TO_ACTION = {
    GAMES[0]: {
        (ord("w"),): 2,
        (ord("s"),): 3
    },
    GAMES[1]: {
        (ord("w"),): 1,
        (ord("d"),): 2,
        (ord("a"),): 3,
        (ord("s"),): 4
    },
    GAMES[2]: {
        (ord("w"),): 1,  
        (ord("d"),): 2,  
        (ord("a"),): 3   
    },
    GAMES[3]: {
        (ord("w"),): 1,  
        (ord("d"),): 2,  
        (ord("a"),): 3   
    }
}

if GAME not in KEYS_TO_ACTION:
    raise ValueError(f"No key mapping defined for {GAME}")

keys_to_action = KEYS_TO_ACTION[GAME]

# ----------------------------
# DATA BUFFER (CURRENT CHUNK ONLY)
# ----------------------------
buffer = {
    "frames": [],
    "actions": [],
    "rewards": [],
    "terminated": [],
    "truncated": [],
    "timestamps": [],
    "episode_ids": []
}

start_time = None
episode_counter = 0
chunk_counter = 0
MAX_SECONDS = BLOCK_DURATION_MINUTES * 60

# ----------------------------
# ASYNC SAVE SYSTEM
# ----------------------------
save_queue = queue.Queue()
stop_saver = object()  # sentinel


def saver_worker():
    while True:
        item = save_queue.get()
        if item is stop_saver:
            save_queue.task_done()
            break

        chunk_data, suffix = item

        filename = f"{SUBJECT_ID}_{GAME.split('/')[-1]}_block{BLOCK_INDEX}_{suffix}.npz"
        filepath = os.path.join(SAVE_FOLDER, filename)

        print(f"[Saver] Saving {len(chunk_data['actions'])} frames to {filepath}...")

        np.savez_compressed(
            filepath,
            frames=np.array(chunk_data["frames"], dtype=np.uint8),
            actions=np.array(chunk_data["actions"], dtype=np.int16),
            rewards=np.array(chunk_data["rewards"], dtype=np.float32),
            terminated=np.array(chunk_data["terminated"], dtype=bool),
            truncated=np.array(chunk_data["truncated"], dtype=bool),
            timestamps=np.array(chunk_data["timestamps"], dtype=np.float64),
            episode_ids=np.array(chunk_data["episode_ids"], dtype=np.int32)
        )

        save_queue.task_done()


saver_thread = threading.Thread(target=saver_worker, daemon=True)
saver_thread.start()

# ----------------------------
# SAVE FUNCTION (NON-BLOCKING)
# ----------------------------
def flush_buffer_to_queue(final=False):
    global buffer, chunk_counter

    if len(buffer["actions"]) == 0:
        return

    suffix = "final" if final else f"chunk{chunk_counter:04d}"

    # Swap buffers so gameplay continues immediately
    chunk_data = buffer
    buffer = {
        "frames": [],
        "actions": [],
        "rewards": [],
        "terminated": [],
        "truncated": [],
        "timestamps": [],
        "episode_ids": []
    }

    save_queue.put((chunk_data, suffix))
    chunk_counter += 1


# ----------------------------
# CALLBACK FUNCTION
# ----------------------------
def my_callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    global start_time, episode_counter, buffer

    if start_time is None:
        start_time = time.time()
        print(f"Timer started! Play for {BLOCK_DURATION_MINUTES} minutes...")

    current_time = time.time()
    elapsed = current_time - start_time

    # Record current step
    buffer["frames"].append(obs_tp1.copy())
    buffer["actions"].append(action)
    buffer["rewards"].append(rew)
    buffer["terminated"].append(terminated)
    buffer["truncated"].append(truncated)
    buffer["timestamps"].append(current_time)
    buffer["episode_ids"].append(episode_counter)

    if terminated or truncated:
        episode_counter += 1

    # Save chunk asynchronously
    if len(buffer["actions"]) >= CHUNK_SIZE:
        flush_buffer_to_queue(final=False)

    # End block
    if elapsed >= MAX_SECONDS:
        print("--- Block finished ---")
        flush_buffer_to_queue(final=True)
        raise KeyboardInterrupt("Block finished. Data queued for saving.")


# ----------------------------
# RUN BLOCK
# ----------------------------
if __name__ == "__main__":
    env = gym.make(GAME, render_mode="rgb_array", frameskip=FRAME_SKIP)

    try:
        play(env, keys_to_action=keys_to_action, callback=my_callback, fps=FPS)

    except KeyboardInterrupt as e:
        print(e)

    finally:
        # Save any remaining unsaved data
        flush_buffer_to_queue(final=True)

        # Wait until all queued chunks are fully written
        save_queue.join()

        # Stop saver thread cleanly
        save_queue.put(stop_saver)
        saver_thread.join()

        env.close()
        print("All data saved successfully.")