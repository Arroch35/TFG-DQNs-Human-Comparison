import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import numpy as np
import torch
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import gymnasium as gym
import cv2

from src.models.custom_dqn import CustomCNN
from src.wrappers.environment_wrappers import (
    RestrictPongActions,
    RestrictMsPacmanActions,
    RestrictSpaceInvadorsActions
)
from src.utils import dqn_preprocess_from_16_frames

# =========================================================
# CONFIGURATION
# =========================================================

GAMES = ["pong", "pacman", "spaceinvaders"]

seeds = ["seed_0", "seed_1","seed_2", "seed_3", "seed_42"]

FRAMES_BASE_FOLDER = "../data/test_16_arrays/pca_training"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GAME_TO_ID = {
    "pacman": "MsPacmanNoFrameskip-v4",
    "pong": "PongNoFrameskip-v4",
    "spaceinvaders": "SpaceInvadersNoFrameskip-v4",
}

# =========================================================
# ENVIRONMENT
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
        raise ValueError(f"No wrapper defined for {game_name}")

    env = Monitor(env)

    return env



# =========================================================
# MAIN
# =========================================================

for seed in seeds:
    OUTPUT_FOLDER = f"../data/dqn_state_action_qvalue/{seed}/pca_training_set"

    MODEL_PATHS = {
        "MsPacmanNoFrameskip-v4": f"../models/MsPacmanNoFrameskip-v4/{seed}/final_model",
        "PongNoFrameskip-v4": f"../models/PongNoFrameskip-v4/{seed}/final_model",
        "SpaceInvadersNoFrameskip-v4": f"../models/SpaceInvadersNoFrameskip-v4/{seed}/final_model",
    }

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for game in GAMES:

        gym_id = GAME_TO_ID[game]

        print(f"\nProcessing game: {game}")

        frames_folder = os.path.join(FRAMES_BASE_FOLDER, game)

        clip_files = sorted(
            [f for f in os.listdir(frames_folder) if f.endswith(".npy")]
        )

        env = make_env(gym_id)

        env = VecFrameStack(
            DummyVecEnv([lambda: env]),
            n_stack=4
        )

        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=512)
        )

        model = DQN(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            buffer_size=1,
            learning_starts=0
        )

        model.set_parameters(
            MODEL_PATHS[gym_id],
            exact_match=True
        )

        model.policy.to(DEVICE)
        model.policy.eval()

        game_output_folder = os.path.join(
            OUTPUT_FOLDER,
            game
        )

        os.makedirs(game_output_folder, exist_ok=True)

        # =====================================================
        # PROCESS CLIPS
        # =====================================================

        for clip_file in clip_files:

            clip_path = os.path.join(
                frames_folder,
                clip_file
            )

            clip_name = os.path.splitext(clip_file)[0]

            frames_array = np.load(clip_path)

            # -------------------------------------
            # DQN input
            # -------------------------------------

            stack = dqn_preprocess_from_16_frames(
                frames_array
            )

            state_vector = stack.flatten()

            # -------------------------------------
            # Save
            # -------------------------------------

            save_file = os.path.join(
                game_output_folder,
                f"{clip_name}.npz"
            )

            np.savez_compressed(
                save_file,
                state=state_vector
            )

        print(f"Finished {game}")

    print("\nDone.")