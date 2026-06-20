import gymnasium as gym
import ale_py
import torch
import wandb
import os
import json
from datetime import datetime

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

from src.models.custom_dqn import CustomCNN
from src.wrappers.environment_wrappers import RestrictPongActions
import argparse

# =========================================================
# CONFIGURATION
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

GAME = "PongNoFrameskip-v4"
TOTAL_TIMESTEPS = 25_000_000
CHECKPOINT_FREQ = 500_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_ENVS = 4  # single environment

print(DEVICE)

# =========================================================
# ENVIRONMENT FUNCTION
# =========================================================
def make_env(seed_offset=0):
    def _init():
        env = gym.make(GAME)
        env = AtariWrapper(env)
        env = RestrictPongActions(env)
        env = Monitor(env)
        env.reset(seed=args.seed + seed_offset)
        return env
    return _init

# =========================================================
# TRAINING FUNCTION
# =========================================================
def train():
    seed = args.seed
    wandb.login()

    # ------------------------------
    # Training Environment (Single)
    # ------------------------------
    env = SubprocVecEnv(
        [make_env(i) for i in range(N_ENVS)]
    )
    env = VecFrameStack(env, n_stack=4)

    # ------------------------------
    # Evaluation Environment
    # ------------------------------
    eval_env = DummyVecEnv([make_env(1000)])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    print("Num envs:", env.num_envs)

    # ------------------------------
    # Config dictionary (for manifest + wandb)
    # ------------------------------
    config = {
        "seed": seed,
        "game": GAME,
        "total_timesteps": TOTAL_TIMESTEPS,
        "checkpoint_freq": CHECKPOINT_FREQ,
        "n_envs": N_ENVS,
        "batch_size": 32,
        "buffer_size": 100_000,
        "learning_starts": 50_000,
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "train_freq": 4,
        "gradient_steps": 4,  # important fix
        "target_update_interval": 10_000,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.1,
        "frame_stack": 4,
        "device": DEVICE,
        "architecture": "CustomCNN",
    }

    # ------------------------------
    # WandB
    # ------------------------------
    run = wandb.init(
        project="dqn-pong-cluster",
        name=f"seed_{seed}",
        config=config,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True,
    )

    # ------------------------------
    # Paths
    # ------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = f"./models/{GAME}/seed_{seed}/{timestamp}/"
    os.makedirs(base_path, exist_ok=True)

    # ------------------------------
    # Save manifest
    # ------------------------------
    manifest_path = os.path.join(base_path, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Manifest saved at: {manifest_path}")

    # ------------------------------
    # Callbacks
    # ------------------------------
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=21,
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=base_path + "best_model/",
        log_path=base_path + "eval_logs/",
        eval_freq=100_000,
        n_eval_episodes=20,  # more stable eval
        deterministic=False,
        render=False,
        callback_on_new_best=stop_callback,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=base_path + "checkpoints/",
        name_prefix=f"dqn_pong_seed_{seed}"
    )

    # ------------------------------
    # Model
    # ------------------------------
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )

    model = DQN(
        "CnnPolicy",
        env,
        seed=seed,
        batch_size=config["batch_size"],
        buffer_size=config["buffer_size"],
        learning_starts=config["learning_starts"],
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        train_freq=config["train_freq"],
        gradient_steps=config["gradient_steps"],
        target_update_interval=config["target_update_interval"],
        exploration_fraction=config["exploration_fraction"],
        exploration_final_eps=config["exploration_final_eps"],
        tensorboard_log="./tensorboard/",
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=DEVICE,
    )

    # ------------------------------
    # Train
    # ------------------------------
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback, WandbCallback(verbose=2)]
    )

    model.save(base_path + "final_model")
    run.finish()


if __name__ == '__main__':
    train()