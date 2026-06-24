import os
import numpy as np
import torch
import torch.nn as nn
import ale_py
import gymnasium as gym
import cv2

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

from src.config import GAMES, SEEDS, GAME_TO_GYM_ID, get_path, ensure, DEVICE
from src.models.custom_dqn import CustomCNN
from src.utils import dqn_preprocess_from_16_frames
from src.wrappers.environment_wrappers import (
    RestrictPongActions, RestrictMsPacmanActions, RestrictSpaceInvadorsActions,
)

# =========================================================
# CONFIGURATION
# =========================================================

EVAL_SEEDS = [s for s in SEEDS if s != "seed_42"]

# =========================================================
# ENVIRONMENT FACTORY
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
        raise ValueError(f"No restricted actions wrapper defined for {gym_id}")
    env = Monitor(env)
    return env

# =========================================================
# HOOK REGISTRATION
# =========================================================
def register_hooks(model):
    activations = {}

    def get_activation(name):
        def hook(module, input, output):
            flat = output.detach().cpu().numpy().reshape(output.shape[0], -1)
            activations.setdefault(name, []).append(flat)
        return hook

    conv_counter = 1
    for layer in model.policy.q_net.features_extractor.cnn:
        if isinstance(layer, nn.ReLU):
            layer.register_forward_hook(get_activation(f"conv{conv_counter}"))
            conv_counter += 1

    for layer in model.policy.q_net.features_extractor.fc:
        if isinstance(layer, nn.ReLU):
            layer.register_forward_hook(get_activation("fc"))

    return activations

# =========================================================
# MAIN LOOP
# =========================================================
for seed in EVAL_SEEDS:
    # Paths for this seed (all resolved from config)
    output_folder = ensure("activations_pool25_seed", seed=seed)

    for game in GAMES:
        gym_id = GAME_TO_GYM_ID[game]

        frames_folder = get_path("arrays_pool25_game", game=game)
        model_path    = get_path("models_dqn", gym_id=gym_id, seed=seed)

        clip_files = [f for f in os.listdir(frames_folder) if f.endswith(".npy")]
        print(f"\n[{seed}] {game}: {len(clip_files)} clips  |  model: {model_path}")

        # Build env + load model
        raw_env = make_env(gym_id)
        env     = VecFrameStack(DummyVecEnv([lambda: raw_env]), n_stack=4)

        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=512),
        )
        model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs,
                    buffer_size=1, learning_starts=0)
        model.set_parameters(str(model_path), exact_match=True)
        model.policy.to(DEVICE)
        model.policy.eval()

        activations = register_hooks(model)

        for clip_file in clip_files:
            # Reset accumulated activations for this clip
            for key in activations:
                activations[key] = []

            frames_array = np.load(frames_folder / clip_file)   # (16, H, W, 3)
            clip_name    = os.path.splitext(clip_file)[0]

            stack        = dqn_preprocess_from_16_frames(frames_array)
            stack        = stack[np.newaxis, ...]                # (1, 4, 84, 84)
            stack_tensor = torch.tensor(stack, dtype=torch.float32).to(DEVICE)

            with torch.no_grad():
                _ = model.policy.q_net.features_extractor(stack_tensor)

            final_activations = {
                f"{clip_name}_{layer_name}": np.concatenate(acts, axis=0)
                for layer_name, acts in activations.items()
            }

            save_file = output_folder / f"{clip_name}_activations.npz"
            np.savez_compressed(save_file, **final_activations)

    print(f"\n[{seed}] All clips processed.")

print("\nDone — all seeds and games processed.")
