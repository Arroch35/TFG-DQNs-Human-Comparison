import sys
from pathlib import Path
import joblib
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

# =========================================================
# CONFIGURATION
# =========================================================

GAMES = ["pong", "pacman", "spaceinvaders"]

seeds = ["seed_0", "seed_1","seed_2", "seed_3", "seed_42"]

FRAMES_BASE_FOLDER = "../data/test_16_arrays/buenos_25"

FILTER_CSV = None #! esto lo tengo mas abajo, cambiar a none cuando haga big rdm

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
# PREPROCESSING
# =========================================================

def dqn_preprocess_from_16_frames(frames_16):
    """
    Input:
        (16,H,W,3)

    Output:
        (4,84,84)
    """

    assert frames_16.shape[0] == 16

    selected_indices = [3, 7, 11, 15]

    processed_frames = []

    for t in selected_indices:

        pooled = np.maximum(frames_16[t], frames_16[t - 1])

        gray = cv2.cvtColor(pooled, cv2.COLOR_RGB2GRAY)

        resized = cv2.resize(
            gray,
            (84, 84),
            interpolation=cv2.INTER_AREA
        )

        processed_frames.append(resized)

    stack = np.stack(processed_frames, axis=0)

    stack = stack.astype(np.float32) / 255.0

    return stack



def load_pca_model(game, seed):

    path = os.path.join(
        "../models/pca_models/pixel_pca_models",
        game,
        seed,
        f"{game}_state_pca.pkl"
    )

    model = joblib.load(path)

    return model["pca"], model["scaler"]


# =========================================================
# MAIN
# =========================================================

for seed in seeds:

    print("\n" + "=" * 60)
    print(f"Processing seed: {seed}")
    print("=" * 60)

    OUTPUT_FOLDER = f"../data/dqn_state_action_qvalue/{seed}/selected_subset_15"

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

        all_clip_files = sorted(
            [f for f in os.listdir(frames_folder) if f.endswith(".npy")]
        )

        FILTER_CSV = f"../data/subset_selection/seed_42/{game}_best_subset_indices.csv"

        if FILTER_CSV and os.path.exists(FILTER_CSV):
            filter_df = pd.read_csv(FILTER_CSV)
            print(filter_df.columns)
            
            allowed_names = set(filter_df['clip_name'].astype(str).str.replace(".mp4", "", regex=False))

            clip_files = [
                f for f in all_clip_files 
                if f.replace(".npy", "") in allowed_names
            ]
        else:
            clip_files = all_clip_files

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

        pca, scaler = load_pca_model(game, seed)
        print(f"Loaded PCA for {game}")

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

            stack = dqn_preprocess_from_16_frames(frames_array)

            state_vector = stack.flatten()

            # =====================================================
            # PCA TRANSFORMATION
            # =====================================================

            state_vector_reshaped = state_vector.reshape(1, -1)

            if scaler is not None:
                state_vector_reshaped = scaler.transform(state_vector_reshaped)

            state_pca = pca.transform(state_vector_reshaped)

            state_pca = state_pca.squeeze(0)  # (100,)

            # -------------------------------------
            # Torch input
            # -------------------------------------

            stack_tensor = torch.tensor(
                stack[np.newaxis, ...],
                dtype=torch.float32,
                device=DEVICE
            )

            # -------------------------------------
            # Q-values
            # -------------------------------------

            with torch.no_grad():
                q_values = model.policy.q_net(stack_tensor)

            q_values = q_values.squeeze(0).cpu().numpy()

            # -------------------------------------
            # Action
            # -------------------------------------

            action = int(np.argmax(q_values))

            # -------------------------------------
            # State value
            # -------------------------------------

            value = float(np.max(q_values))

            # -------------------------------------
            # SAVE
            # -------------------------------------

            save_file = os.path.join(
                game_output_folder,
                f"{clip_name}.npz"
            )

            np.savez_compressed(
                save_file,
                state=state_vector,          # original pixels
                state_pca=state_pca,         # PCA 100D representation
                action=action,
                q_values=q_values,
                value=value
            )

        print(f"Finished {game}")

    print("\nDone.")