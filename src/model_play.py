import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
import stable_baselines3
print(stable_baselines3.__version__)

# Same action restriction wrapper
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


# Create evaluation environment (IMPORTANT: must match training)
def make_eval_env():
    env = gym.make("ALE/Pong-v5", render_mode="human")
    env = AtariWrapper(env)          # preprocessing + frameskip
    env = RestrictPongActions(env)   # restrict actions
    env = Monitor(env)                # important for WandB reward logging
    env.reset(seed=42)
    return env

env = DummyVecEnv([make_eval_env])
env = VecFrameStack(env, 4)

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=512),
)

model = DQN(
    "CnnPolicy",
    env,
    policy_kwargs=policy_kwargs,
    buffer_size=1,          # 👈 tiny buffer
    learning_starts=0,      # 👈 irrelevant for playing
)

# 2️⃣ Load ONLY the parameters (weights)
model.set_parameters(
    "../../kaggle_outputs/best_model",
    exact_match=True,
)

obs = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward[0]
