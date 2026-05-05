import gymnasium as gym
import ale_py
import os
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor

# Import your custom modules
from models.custom_dqn import CustomCNN
from wrappers.environment_wrappers import (
    RestrictPongActions, 
    RestrictMsPacmanActions, 
    RestrictSpaceInvadorsActions
)

# --- CONFIGURATION SETTINGS ---
GAME_CONFIGS = {
    "MsPacman": {
        "env_id": "MsPacmanNoFrameskip-v4",
        "wrapper": RestrictMsPacmanActions,
        "model_path": "../models/MsPacmanNoFrameskip-v4/seed_27/best_model/best_model"
    },
    "Pong": {
        "env_id": "PongNoFrameskip-v4",
        "wrapper": RestrictPongActions,
        "model_path": "../models/ALE/Pong-v5/dqn_pong_seed_42_18000000_steps"
    },
    "SpaceInvaders": {
        "env_id": "SpaceInvadersNoFrameskip-v4",
        "wrapper": RestrictSpaceInvadorsActions,
        "model_path": "../models/SpaceInvadersNoFrameskip-v4/seed_42/best_model/best_model"
    }
}

# 1. PICK YOUR GAME HERE
CURRENT_GAME = "Pong" 
config = GAME_CONFIGS[CURRENT_GAME]

# --- SETUP FUNCTIONS ---

def make_env():
    # Use render_mode="rgb_array" for video recording
    env = gym.make(config["env_id"], render_mode="rgb_array") 
    env = AtariWrapper(env)
    env = config["wrapper"](env)
    env = Monitor(env)
    return env

# Create Vectorized Env
env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4)

# 2. VIDEO RECORDING SETUP
video_folder = f"videos/{CURRENT_GAME}/"
video_length = 5000  # Number of steps to record

env = VecVideoRecorder(
    env, 
    video_folder,
    record_video_trigger=lambda x: x == 0, # Record starting at step 0
    video_length=video_length,
    name_prefix=f"dqn-{CURRENT_GAME}"
)

# --- MODEL LOADING ---

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=512),
)

model = DQN(
    "CnnPolicy",
    env,
    policy_kwargs=policy_kwargs,
    buffer_size=1,
    learning_starts=0,
    verbose=1
)

print(f"Loading weights for {CURRENT_GAME}...")
model.set_parameters(config["model_path"], exact_match=True)

# --- EXECUTION ---

obs = env.reset()
print(f"Recording started. Saving to {video_folder}")

# The loop stays the same, but VecVideoRecorder handles saving in the background
for _ in range(video_length):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    if done:
        break

# Clean up
env.close()
print("Finished.")