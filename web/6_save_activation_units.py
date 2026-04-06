import sys
from pathlib import Path

# Add the src folder to Python’s search path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import numpy as np
import torch
import torch.nn as nn
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import gymnasium as gym
import cv2

from src.models.custom_dqn import CustomCNN
from src.wrappers.environment_wrappers import RestrictPongActions, RestrictMsPacmanActions, RestrictSpaceInvadorsActions

from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================
GAMES = ["pacman", "pong", "spaceinvaders"]
FRAMES_BASE_FOLDER = "../data/frame_arrays"
OUTPUT_FOLDER = "../data/DQN_activations"
MODEL_PATHS = {
    "MsPacmanNoFrameskip-v4": "../models/MsPacmanNoFrameskip-v4/seed_27/best_model/best_model",
    "PongNoFrameskip-v4": "../models/ALE/Pong-v5/dqn_pong_seed_42_18000000_steps",
    "SpaceInvadersNoFrameskip-v4": "../models/SpaceInvadorsNoFrameskip-v4/seed_42/best_model/best_model",
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GAME_TO_ID = {
    "pacman": "MsPacmanNoFrameskip-v4",
    "pong": "PongNoFrameskip-v4",
    "spaceinvaders": "SpaceInvadersNoFrameskip-v4"
}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# =========================================================
# ENVIRONMENT CREATION WITH GAME-SPECIFIC ACTIONS
# =========================================================
def make_env(game_name):
    env = gym.make(game_name)
    env = AtariWrapper(env)
    if game_name.startswith("Pong"):
        env = RestrictPongActions(env)
    elif game_name.startswith("MsPacman"):
        env = RestrictMsPacmanActions(env)
    elif game_name.startswith("SpaceInvaders"):
        env = RestrictSpaceInvadorsActions(env)
    else:
        raise ValueError(f"No restricted actions wrapper defined for {game_name}")
    env = Monitor(env)
    return env

# =========================================================
# PREPROCESS FRAMES
# =========================================================
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0

# =========================================================
# REGISTER HOOKS TO CAPTURE ACTIVATIONS
# =========================================================
def register_hooks(model):
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            flat = output.detach().cpu().numpy().reshape(output.shape[0], -1)
            if name not in activations:
                activations[name] = []
            activations[name].append(flat)
        return hook

    conv_counter = 1
    for layer in model.policy.q_net.features_extractor.cnn:
        if isinstance(layer, nn.Conv2d):
            layer.register_forward_hook(get_activation(f"conv{conv_counter}"))
            conv_counter += 1

    for idx, layer in enumerate(model.policy.q_net.features_extractor.fc):
        if isinstance(layer, nn.Linear):
            layer.register_forward_hook(get_activation(f"fc{idx}"))

    return activations

# =========================================================
# PROCESS ALL GAMES
# =========================================================
for game in GAMES:
    # 1. Folders use the short name (pacman, pong, spaceinvaders)
    frames_folder = os.path.join(FRAMES_BASE_FOLDER, game)
    
    # 2. Get the technical ID for gym.make()
    gym_id = GAME_TO_ID[game]
    print(gym_id)

    clip_files = [f for f in os.listdir(frames_folder) if f.endswith(".npy")]
    print(f"\nProcessing {len(clip_files)} clips for game: {game}")

    # Create environment & model
    env = make_env(gym_id)
    env = VecFrameStack(DummyVecEnv([lambda: env]), n_stack=4)

    policy_kwargs = dict(features_extractor_class=CustomCNN, features_extractor_kwargs=dict(features_dim=512))
    model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, buffer_size=1, learning_starts=0)
    model.set_parameters(MODEL_PATHS[gym_id], exact_match=True)
    model.policy.to(DEVICE)
    model.policy.eval()

    for clip_file in clip_files:
        clip_path = os.path.join(frames_folder, clip_file)
        clip_name = os.path.splitext(clip_file)[0]
        frames_array = np.load(clip_path)  # shape: (4, H, W, 3)

        # Preprocess and stack
        processed_frames = np.array([preprocess_frame(f) for f in frames_array])
        stack = processed_frames[np.newaxis, ...]  # shape (1, 4, H, W)
        stack_tensor = torch.tensor(stack, dtype=torch.float32).to(DEVICE)

        activations = register_hooks(model)

        # Forward pass
        with torch.no_grad():
            _ = model.policy.q_net.features_extractor(stack_tensor)

        # Save activations per layer
        final_activations = {}
        for layer_name, acts in activations.items():
            matrix = np.concatenate(acts, axis=0)
            key_name = f"{clip_name}_{layer_name}"
            final_activations[key_name] = matrix
            print(f"{key_name} shape: {matrix.shape}")

        save_file = os.path.join(OUTPUT_FOLDER, f"{clip_name}_activations.npz")
        np.savez_compressed(save_file, **final_activations)
        print(f"Saved activations: {save_file}")

print("\nAll clips processed with game-specific restricted actions.")

#TODO: Me sale un error de no se que mierdas