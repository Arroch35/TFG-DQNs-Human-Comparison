"""
3_extract_pixel_qvalue_states.py
For each clip in the selected-15 subset, extract:
  - flattened pixel state (+ PCA projection)
  - Q-values, greedy action, state value
and save as .npz files.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import numpy as np
import torch
import joblib
import pandas as pd
import ale_py
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

from src.config import (
    GAMES, SEEDS, GAME_TO_GYM_ID, DEVICE, REFERENCE_SEED,
    get_path, ensure,
)
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

        frames_folder  = get_path("arrays_pool25_game", game=game)
        output_folder  = ensure("states_subset15_game", seed=seed, game=game)

        # ── Clip filtering (always use REFERENCE_SEED subset) ──
        filter_csv = get_path("subsets_csv", seed=REFERENCE_SEED, game=game)
        all_clips  = sorted(f for f in os.listdir(frames_folder) if f.endswith(".npy"))

        if filter_csv.exists():
            filter_df    = pd.read_csv(filter_csv)
            allowed      = set(filter_df["clip_name"].astype(str).str.replace(".mp4", "", regex=False))
            clip_files   = [f for f in all_clips if f.replace(".npy", "") in allowed]
            print(f"Filtering: {len(clip_files)} of {len(all_clips)} clips kept.")
        else:
            clip_files = all_clips
            print(f"Warning: filter CSV not found at {filter_csv}. Using all clips.")

        # ── Model ──────────────────────────────────────────────
        env = VecFrameStack(DummyVecEnv([lambda: make_env(gym_id)]), n_stack=4)
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=512),
        )
        model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, buffer_size=1, learning_starts=0)
        model.set_parameters(str(get_path("models_dqn", gym_id=gym_id, seed=seed)), exact_match=True)
        model.policy.to(DEVICE)
        model.policy.eval()

        # ── PCA model ──────────────────────────────────────────
        pca_path = get_path("models_pca_pixel", game=game, seed=seed) / f"{game}_state_pca.pkl"
        pca_data = joblib.load(pca_path)
        pca, scaler = pca_data["pca"], pca_data["scaler"]
        print(f"Loaded PCA from {pca_path}")

        # ── Process clips ──────────────────────────────────────
        for clip_file in clip_files:
            clip_name  = Path(clip_file).stem
            frames_arr = np.load(frames_folder / clip_file)
            stack      = dqn_preprocess_from_16_frames(frames_arr)

            state_vector = stack.flatten()

            # PCA projection
            sv = state_vector.reshape(1, -1)
            if scaler is not None:
                sv = scaler.transform(sv)
            state_pca = pca.transform(sv).squeeze(0)

            # Q-values / action / value
            stack_tensor = torch.tensor(stack[np.newaxis, ...], dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                q_values = model.policy.q_net(stack_tensor).squeeze(0).cpu().numpy()

            action = int(np.argmax(q_values))
            value  = float(np.max(q_values))

            np.savez_compressed(
                output_folder / f"{clip_name}.npz",
                state=state_vector,
                state_pca=state_pca,
                action=action,
                q_values=q_values,
                value=value,
            )

        print(f"Finished {game}")

print("\nDone.")
