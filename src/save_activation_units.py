"""
DQN Activation Extraction Script for RSA
-----------------------------------------

This script:
1. Loads a trained DQN model.
2. Loads human gameplay frames.
3. Preprocesses frames exactly like AtariWrapper.
4. Downsamples frames by 20 (as in the paper).
5. Creates sliding 4-frame stacks (t-3, t-2, t-1, t).
6. Passes each stack through the DQN.
7. Extracts and flattens activations from each layer.
8. Saves layer-wise activation matrices ready for RSA.

Output:
Each saved array has shape:
    (num_selected_frames, num_units)

Saved naming format:
    subject_game_block_layer
"""

import os
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import ale_py
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import cv2

# =========================================================
# CONFIGURATION
# =========================================================
SUBJECT_ID = "sub01"
GAME = "Pong"
BLOCK = "block2"

FILE = "../data/sub01_Pong-v5_block1.npz"
MODEL_PATH = "../models/ALE/Pong-v5/dqn_pong_seed_42_18000000_steps"
SAVE_FOLDER = "../data"

DOWNSAMPLE = 20   # Paper uses downsampling by 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# ACTION RESTRICTION WRAPPER (same as training)
# =========================================================
class RestrictPongActions(gym.ActionWrapper):
    """
    Restrict actions to [NOOP, UP, DOWN]
    Must match training exactly.
    """
    def __init__(self, env):
        super().__init__(env)
        self.allowed_actions = np.array([0, 2, 3])
        self.action_space = gym.spaces.Discrete(len(self.allowed_actions))

    def action(self, action):
        return int(self.allowed_actions[action])


# =========================================================
# CUSTOM CNN (same architecture as training)
# =========================================================
class CustomCNN(BaseFeaturesExtractor):
    """
    CNN used during DQN training.
    Must match training architecture exactly.
    """
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )

        # Compute flattened size dynamically
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


# =========================================================
# CREATE ENVIRONMENT (to build correct observation space)
# =========================================================
def make_env():
    env = gym.make("PongNoFrameskip-v4")
    env = AtariWrapper(env)
    env = RestrictPongActions(env)
    env = Monitor(env)
    return env

env = make_env()
env = VecFrameStack(DummyVecEnv([lambda: env]), n_stack=4)


# =========================================================
# LOAD TRAINED DQN
# =========================================================
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
)

model.set_parameters(MODEL_PATH, exact_match=True)
model.policy.to(DEVICE)
model.policy.eval()


# =========================================================
# REGISTER FORWARD HOOKS TO CAPTURE ACTIVATIONS
# =========================================================
activations = {}

def get_activation(name):
    """
    Stores flattened activation per frame.
    Output shape per frame:
        (1, num_units)
    """
    def hook(model, input, output):
        flat = output.detach().cpu().numpy().reshape(output.shape[0], -1)
        if name not in activations:
            activations[name] = []
        activations[name].append(flat)
    return hook


# Register hooks for each convolution layer
conv_counter = 1  # start numbering from 1
for layer in model.policy.q_net.features_extractor.cnn:
    if isinstance(layer, nn.Conv2d):
        layer.register_forward_hook(get_activation(f"conv{conv_counter}"))
        conv_counter += 1

# Register hook for fully connected layer
for idx, layer in enumerate(model.policy.q_net.features_extractor.fc):
    if isinstance(layer, nn.Linear):
        layer.register_forward_hook(get_activation(f"fc{idx}"))


# =========================================================
# LOAD HUMAN GAMEPLAY FRAMES
# =========================================================
data = np.load(FILE)
raw_frames = data["frames"]  # shape (N, 210, 160, 3)


# =========================================================
# PREPROCESS FRAMES (MATCH AtariWrapper)
# =========================================================
def preprocess_frame(frame):
    """
    Matches AtariWrapper:
    1. Convert RGB -> grayscale
    2. Resize to 84x84
    3. Normalize to [0,1]
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    processed = resized.astype(np.float32) / 255.0
    return processed

processed_frames = np.array([preprocess_frame(f) for f in raw_frames])


# =========================================================
# SELECT FRAMES (DOWNSAMPLE BY 20)
# =========================================================
# We start at index 3 because DQN needs t-3, t-2, t-1, t
selected_indices = np.arange(3, len(processed_frames), DOWNSAMPLE)


# =========================================================
# PASS FRAMES THROUGH DQN (SLIDING STACK)
# =========================================================
with torch.no_grad():
    for t in selected_indices:
        stack = processed_frames[t-3:t+1]   # sliding stack
        stack_tensor = torch.tensor(stack, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        _ = model.policy.q_net.features_extractor(stack_tensor)


# =========================================================
# CONCATENATE ACTIVATIONS INTO MATRICES
# =========================================================
final_activations = {}

for layer_name, acts in activations.items():
    # acts is a list of (1, num_units)
    matrix = np.concatenate(acts, axis=0)
    key_name = f"{SUBJECT_ID}_{GAME}_{BLOCK}_{layer_name}"
    final_activations[key_name] = matrix
    print(f"{key_name} shape: {matrix.shape}")


# =========================================================
# SAVE ACTIVATIONS FOR RSA
# =========================================================
save_path = os.path.join(
    SAVE_FOLDER,
    f"{SUBJECT_ID}_{GAME}_{BLOCK}_DQN_activations.npz"
)

np.savez_compressed(save_path, **final_activations)

print(f"\nSaved RSA-ready activations to:\n{save_path}")