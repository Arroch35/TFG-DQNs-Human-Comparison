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

from models.custom_dqn import CustomCNN
from wrappers.environment_wrappers import RestrictPongActions, RestrictMsPacmanActions, RestrictSpaceInvadorsActions

# Create evaluation environment (IMPORTANT: must match training)
def make_eval_env():
    env = gym.make("MsPacmanNoFrameskip-v4", render_mode="human") #PongNoFrameskip-v4 SpaceInvadorsNoFrameskip-v4 MsPacmanNoFrameskip
    env = AtariWrapper(env)          # preprocessing + frameskip
    env = RestrictMsPacmanActions(env) # RestrictPongActions(env)   # restrict actions
    env = Monitor(env)                # important for WandB reward logging
    env.reset()
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
    "../models/MsPacmanNoFrameskip-v4/seed_27/best_model/best_model", #"../models/ALE/Pong-v5/dqn_pong_seed_42_18000000_steps",
    exact_match=True,
)

obs = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward[0]
