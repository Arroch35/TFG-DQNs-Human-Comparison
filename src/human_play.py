import gymnasium as gym
import ale_py
from gymnasium.utils.play import play
import numpy as np
import time
import os

#? De momento lo dejo así, pero estaría interesante una interfaz gráfica para cuando haga experimentos con humanos
#? Es decir, algo con lo que sea facil hacer clics y seleccionar el juego, que al acabar un bloque se pueda pulsar otra vez jugar cuando esten listos, etc
#? Y que se pueda identificar el sujeto, no con nombre, sino con numero (esto algo interno de la aplicación)

#! Falta hacer resize del screen

# ----------------------------
# CONFIGURATION
# ----------------------------
GAME = "ALE/MsPacman-v5" #"ALE/Pong-v5"  # change for the current block
SUBJECT_ID = "sub01"  # change per subject
BLOCK_INDEX = 1       # change if you want to track block number
BLOCK_DURATION_MINUTES = 2  # duration of this block
FPS = 60              # display FPS
FRAME_SKIP = 1        # environment frameskip
SAVE_FOLDER = "../data"

# Ensure save folder exists
os.makedirs(SAVE_FOLDER, exist_ok=True)

# ----------------------------
# DATA STORAGE
# ----------------------------
data = {
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
MAX_SECONDS = BLOCK_DURATION_MINUTES * 60

# ----------------------------
# CALLBACK FUNCTION
# ----------------------------
def my_callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    global start_time, episode_counter, data

    if start_time is None:
        start_time = time.time()
        print(f"Timer started! Play for {BLOCK_DURATION_MINUTES} minutes...")

    current_time = time.time()
    elapsed = current_time - start_time

    # Record everything
    data["frames"].append(obs_tp1.copy())
    data["actions"].append(action)
    data["rewards"].append(rew)
    data["terminated"].append(terminated)
    data["truncated"].append(truncated)
    data["timestamps"].append(current_time)
    data["episode_ids"].append(episode_counter)

    if terminated or truncated:
        episode_counter += 1

    if elapsed >= MAX_SECONDS:
        print(f"--- Block finished ({len(data['actions'])} frames) ---")
        filename = f"{SUBJECT_ID}_{GAME.split('/')[-1]}_block{BLOCK_INDEX}.npz"
        filepath = os.path.join(SAVE_FOLDER, filename)
        np.savez_compressed(
            filepath,
            frames=np.array(data["frames"], dtype=np.uint8),
            actions=np.array(data["actions"]),
            rewards=np.array(data["rewards"]),
            terminated=np.array(data["terminated"]),
            truncated=np.array(data["truncated"]),
            timestamps=np.array(data["timestamps"]),
            episode_ids=np.array(data["episode_ids"])
        )
        raise KeyboardInterrupt("Block finished. Data saved.")

# ----------------------------
# CONTROLS
# ----------------------------
keys_to_action = {
    (ord("w"),): 2,
    (ord("s"),): 3
}

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
        env.close()