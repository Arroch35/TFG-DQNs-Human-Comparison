"""
1_build_pca_training_set.py
Extract flattened pixel states from PCA-training clips and save them as .npz
files used later by 2_train_pixel_pca_model.py.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import numpy as np
import torch
import ale_py
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

from src.config import GAMES, SEEDS, GAME_TO_GYM_ID, DEVICE, MODELS, get_path, ensure
from src.models.custom_dqn import CustomCNN
from src.wrappers.environment_wrappers import (
    RestrictPongActions,
    RestrictMsPacmanActions,
    RestrictSpaceInvadorsActions,
)
from src.utils import dqn_preprocess_from_16_frames

# =========================================================
# ENVIRONMENT
# =========================================================
def make_env(gym_id: str):
    env = gym.make(gym_id)
    env = AtariWrapper(env)
    if gym_id.startswith("Pong"):
        env = RestrictPongActions(env)
    elif gym_id.startswith("MsPacman"):
        env = RestrictMsPacmanActions(env)
    elif gym_id.startswith("SpaceInvaders"):
        env = RestrictSpaceInvadorsActions(env)
    else:
        raise ValueError(f"No wrapper defined for {gym_id}")
    env = Monitor(env)
    return env

# =========================================================
# MAIN
# =========================================================
for seed in SEEDS:
    print(f"\n{'='*60}\nSeed: {seed}\n{'='*60}")

    for game in GAMES:
        gym_id = GAME_TO_GYM_ID[game]
        print(f"\nProcessing game: {game}")

        frames_folder = get_path("arrays_pca_game", game=game)
        output_folder = ensure("states_pca_game", seed=seed, game=game)

        clip_files = sorted(f for f in os.listdir(frames_folder) if f.endswith(".npy"))

        env = VecFrameStack(DummyVecEnv([lambda: make_env(gym_id)]), n_stack=4)

        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=512),
        )
        model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, buffer_size=1, learning_starts=0)
        model.set_parameters(
            str(get_path("models_dqn", gym_id=gym_id, seed=seed)),
            exact_match=True,
        )
        model.policy.to(DEVICE)
        model.policy.eval()

        for clip_file in clip_files:
            clip_name  = Path(clip_file).stem
            frames_arr = np.load(frames_folder / clip_file)
            stack      = dqn_preprocess_from_16_frames(frames_arr)

            np.savez_compressed(
                output_folder / f"{clip_name}.npz",
                state=stack.flatten(),
            )

        print(f"Finished {game}")

print("\nDone.")
