import gymnasium as gym
import ale_py
from gymnasium.utils.play import play
import numpy as np
import time
import os
import threading
import queue

from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from wrappers.custom_atari_wrapper import CustomAtariWrapper
from src.config import GAMES, GAME_TO_GYM_ID, SEEDS, get_path

# ----------------------------
# CONFIGURATION
# ----------------------------
GAME = GAMES[2]                            # "spaceinvaders"
GYM_ID = GAME_TO_GYM_ID[GAME]             # "SpaceInvadersNoFrameskip-v4"

SUBJECT_ID = "sub_training_set_pca"
BLOCK_INDEX = 1
BLOCK_DURATION_MINUTES = 15
FPS = 60
FRAME_SKIP = 1
SAVE_FOLDER = get_path("recordings_pca")   # data/human_plays/pca_training
CHUNK_SIZE = 5000

SEED = 200

os.makedirs(SAVE_FOLDER, exist_ok=True)

# ----------------------------
# CONTROLS
# ----------------------------
KEYS_TO_ACTION = {
    GAME_TO_GYM_ID["pong"]:          {(ord("w"),): 2, (ord("s"),): 3},
    GAME_TO_GYM_ID["pacman"]:        {(ord("w"),): 1, (ord("d"),): 2, (ord("a"),): 3, (ord("s"),): 4},
    GAME_TO_GYM_ID["spaceinvaders"]: {(ord("w"),): 1, (ord("d"),): 2, (ord("a"),): 3},
}

if GYM_ID not in KEYS_TO_ACTION:
    raise ValueError(f"No key mapping defined for {GYM_ID}")

keys_to_action = KEYS_TO_ACTION[GYM_ID]

# ----------------------------
# CREATE DQN ENV
# ----------------------------
def make_dqn_env():
    e = gym.make(GYM_ID)
    e = AtariWrapper(e)
    return e

dqn_env = DummyVecEnv([make_dqn_env])
dqn_env = VecFrameStack(dqn_env, n_stack=4)

# ----------------------------
# CREATE HUMAN-READABLE ENV
# ----------------------------
def make_human_readable_env():
    e = gym.make(GYM_ID)
    e = CustomAtariWrapper(e)
    return e

human_env = DummyVecEnv([make_human_readable_env])
human_env = VecFrameStack(human_env, n_stack=4)

# ----------------------------
# DATA BUFFER
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
stop_saver = object()

def saver_worker():
    while True:
        item = save_queue.get()
        if item is stop_saver:
            save_queue.task_done()
            break

        chunk_data, suffix = item

        filename = f"{SUBJECT_ID}_{GYM_ID.split('/')[-1]}_block{BLOCK_INDEX}_{suffix}.npz"
        filepath = os.path.join(SAVE_FOLDER, filename)

        print(f"[Saver] Saving {len(chunk_data['actions'])} steps to {filepath}...")

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
# SAVE FUNCTION
# ----------------------------
def flush_buffer_to_queue(final=False):
    global buffer, chunk_counter

    if len(buffer["actions"]) == 0:
        return

    suffix = "final" if final else f"chunk{chunk_counter:04d}"

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
    print(action)

    frame = env.render()

    print(frame.shape)
    buffer["frames"].append(frame.copy())
    buffer["actions"].append(action)
    buffer["rewards"].append(rew)
    buffer["terminated"].append(terminated)
    buffer["truncated"].append(truncated)
    buffer["timestamps"].append(current_time)
    buffer["episode_ids"].append(episode_counter)

    if terminated or truncated:
        episode_counter += 1
        dqn_env.reset()
        human_env.reset()

    if len(buffer["actions"]) >= CHUNK_SIZE:
        flush_buffer_to_queue(final=False)

    if elapsed >= MAX_SECONDS:
        print("--- Block finished ---")
        flush_buffer_to_queue(final=True)
        raise KeyboardInterrupt("Block finished. Data queued for saving.")

# ----------------------------
# RUN BLOCK
# ----------------------------
if __name__ == "__main__":
    env = gym.make(GYM_ID, render_mode="rgb_array", frameskip=FRAME_SKIP)

    obs, _ = env.reset(seed=SEED)

    dqn_env.seed(SEED)
    dqn_env.reset()

    human_env.seed(SEED)
    human_env.reset()

    try:
        play(env, keys_to_action=keys_to_action, callback=my_callback, fps=FPS)

    except KeyboardInterrupt as e:
        print(e)

    finally:
        flush_buffer_to_queue(final=True)

        save_queue.join()

        save_queue.put(stop_saver)
        saver_thread.join()

        env.close()
        print("All data saved successfully.")
