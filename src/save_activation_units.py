import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor

# ----------------------------
# CONFIGURATION
# ----------------------------
HUMAN_DATA_FILE = "../data/sub01_Pong-v5_block1.npz"  # human gameplay file
SAVE_FOLDER = "../data/activations"
FRAME_SKIP = 1         # as in gameplay recording
STACK_SIZE = 4
FPS_INTERVAL = 5       # save activations every 5 frames (tunable)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Metadata
SUBJECT_ID = "sub01"
GAME = "Pong"
BLOCK_INDEX = 1

os.makedirs(SAVE_FOLDER, exist_ok=True)

# ----------------------------
# LOAD HUMAN DATA
# ----------------------------
human_data = np.load(HUMAN_DATA_FILE)
frames = human_data["frames"]  # shape: (num_frames, H, W, C)
frames = frames.transpose(0, 3, 1, 2)  # convert to (N, C, H, W) for PyTorch

# ----------------------------
# DQN SETUP
# ----------------------------
class RestrictPongActions(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.allowed_actions = np.array([0, 2, 3])
        self.action_space = gym.spaces.Discrete(len(self.allowed_actions))

    def action(self, action):
        return int(self.allowed_actions[action])

class CustomCNN(BaseFeaturesExtractor):
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

# Load model
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=512),
)

env_dummy = gym.make("ALE/Pong-v5")
env_dummy = AtariWrapper(env_dummy)
env_dummy = RestrictPongActions(env_dummy)
env_dummy = Monitor(env_dummy)
env_dummy.reset(seed=42)

model = DQN(
    "CnnPolicy",
    env_dummy,
    policy_kwargs=policy_kwargs,
    buffer_size=1,         # not used for playing
    learning_starts=0
)
model.set_parameters("../../kaggle_outputs/dqn_pong_final", exact_match=True)
model.policy.to(DEVICE)
model.policy.eval()

# ----------------------------
# REGISTER HOOKS TO COLLECT ACTIVATIONS
# ----------------------------
activations = {}
def save_activation(name):
    def hook(model, input, output):
        activations[name].append(output.detach().cpu().numpy())
    return hook

# Collect all named layers
for name, module in model.policy.features_extractor.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.ReLU):
        activations[name] = []
        module.register_forward_hook(save_activation(name))

# ----------------------------
# HELPER: STACK FRAMES
# ----------------------------
def get_stacked_frame(frames_list, idx, stack_size=4):
    start_idx = max(0, idx - stack_size + 1)
    stacked = frames_list[start_idx:idx+1]
    # pad if not enough frames
    if stacked.shape[0] < stack_size:
        pad = np.repeat(stacked[0:1], stack_size - stacked.shape[0], axis=0)
        stacked = np.concatenate([pad, stacked], axis=0)
    return stacked

# ----------------------------
# RUN HUMAN FRAMES THROUGH DQN
# ----------------------------
num_frames = frames.shape[0]
for idx in range(0, num_frames, FPS_INTERVAL):
    stacked_input = get_stacked_frame(frames, idx, STACK_SIZE)
    stacked_input = torch.tensor(stacked_input, dtype=torch.float32, device=DEVICE).unsqueeze(0) / 255.0
    with torch.no_grad():
        _ = model.policy.features_extractor(stacked_input)

# ----------------------------
# SAVE ACTIVATIONS
# ----------------------------
for layer_name, acts in activations.items():
    acts = np.array(acts)
    filename = f"{SUBJECT_ID}_{GAME}_block{BLOCK_INDEX}_{layer_name}.npz"
    filepath = os.path.join(SAVE_FOLDER, filename)
    np.savez_compressed(filepath, activations=acts)

print(f"Saved activations for {num_frames} frames in {SAVE_FOLDER}")