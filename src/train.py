import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import numpy as np
import random
import wandb
import os

from gymnasium import ActionWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.atari_wrappers import AtariWrapper
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from models.custom_dqn import CustomCNN
from wrappers.environment_wrappers import RestrictPongActions

wandb_api="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"

os.environ["WANDB_API_KEY"] = wandb_api
wandb.login()

os.makedirs("./checkpoints/", exist_ok=True)


# =========================================================
# CONFIGURATION
# =========================================================
GAME = "ALE/Pong-v5"
TOTAL_TIMESTEPS = 25_000_000
CHECKPOINT_FREQ = 500_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


# =========================================================
# ENVIRONMENT SETUP
# =========================================================
def make_env():
    env = gym.make(GAME)
    env = AtariWrapper(env)          # preprocessing + frameskip
    env = RestrictPongActions(env)   # restrict actions
    env = Monitor(env)                # important for WandB reward logging
    return env

env = make_env()

# Stack 4 frames as input
env = VecFrameStack(DummyVecEnv([lambda: env]), n_stack=4)

# Separate evaluation environment (no training noise)
eval_env = make_env()
eval_env = VecFrameStack(DummyVecEnv([lambda: eval_env]), n_stack=4)

# Stop when mean reward reaches threshold
stop_callback = StopTrainingOnRewardThreshold(
    reward_threshold=20,  # stop when mean reward >= 20
    verbose=1
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/"+GAME+"/best_model/",
    log_path="./models/"+GAME+"/eval_logs/",
    eval_freq=100_000,
    deterministic=True,
    render=False,
    callback_on_new_best=stop_callback,
)

# =========================================================
# WANDB INIT
# =========================================================
run = wandb.init(
    project="dqn-pong-single-env",
    config={
        "total_timesteps": TOTAL_TIMESTEPS,
        "checkpoint_freq": CHECKPOINT_FREQ,
        "architecture": "NatureCNN",
    },
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
)

# =========================================================
# CHECKPOINT CALLBACK
# =========================================================
checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path="./models/"+GAME+"/checkpoints/",
    name_prefix="dqn_pong"
)

# =========================================================
# DQN MODEL
# =========================================================
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=512),
)

model = DQN(
    "CnnPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=50_000,
    batch_size=32,
    gamma=0.99,
    train_freq=4,
    target_update_interval=10_000,
    exploration_fraction=0.1,      # 10% of training with epsilon decay
    exploration_final_eps=0.01,
    tensorboard_log="./tensorboard/",
    policy_kwargs=policy_kwargs,
    verbose=1,
    device=DEVICE,
)

# =========================================================
# TRAIN
# =========================================================
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[
        checkpoint_callback,
        eval_callback,
        WandbCallback(verbose=2)
    ]
)

# =========================================================
# SAVE FINAL MODEL
# =========================================================
model.save("dqn_pong_final")
run.finish()

