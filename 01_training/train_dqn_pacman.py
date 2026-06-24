import gymnasium as gym
import ale_py
import torch
import wandb
import os
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

from src.models.custom_dqn import CustomCNN
from src.wrappers.environment_wrappers import RestrictMsPacmanActions
from src.config import DEVICE
import argparse

# =========================================================
# CONFIGURATION
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

GAME =  "MsPacmanNoFrameskip-v4" # "PongNoFrameskip-v4"
TOTAL_TIMESTEPS = 25_000_000
CHECKPOINT_FREQ = 500_000
N_ENVS = 4  # Number of parallel environments

# =========================================================
# ENVIRONMENT FUNCTION
# =========================================================
def make_env(seed_offset=0):
    def _init():
        env = gym.make(GAME)
        env = AtariWrapper(env)
        env = RestrictMsPacmanActions(env)
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
    # Parallel Training Environment
    # ------------------------------
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    env = VecFrameStack(env, n_stack=4)

    # ------------------------------
    # Evaluation Environment
    # ------------------------------
    eval_env = SubprocVecEnv([make_env(1000)])  # separate seed for eval
    eval_env = VecFrameStack(eval_env, n_stack=4)
    
    print("Train obs space:", env.observation_space)
    print("Eval obs space:", eval_env.observation_space)
    print("Train action space:", env.action_space)
    print("Eval action space:", eval_env.action_space)
    print("Train env type:", type(env))
    print("Eval env type:", type(eval_env))

    print("Num envs:", env.num_envs)
    
    import time

    obs = env.reset()
    start = time.time()
    
    for _ in range(100):
        actions = [env.action_space.sample() for _ in range(env.num_envs)]
        obs, rewards, dones, infos = env.step(actions)
    
    end = time.time()
    print("Steps/sec:", 100 * env.num_envs / (end - start))

    # ------------------------------
    # Callbacks & WandB
    # ------------------------------
    run = wandb.init(
        project="dqn-pacman-cluster",
        name=f"seed_{seed}",
        config={
            "seed": seed,
            "total_timesteps": TOTAL_TIMESTEPS,
            "checkpoint_freq": CHECKPOINT_FREQ,
            "architecture": "NatureCNN",
            "n_envs": N_ENVS,
            "batch_size": 32,
            "gradient_steps": 4
        },
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True,
    )

    base_path = f"./models/{GAME}/seed_{seed}/"
    os.makedirs(base_path, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=base_path + "best_model/",
        log_path=base_path + "eval_logs/",
        eval_freq=100_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=base_path + "checkpoints/",
        name_prefix=f"dqn_pacman_seed_{seed}"
    )

    # ------------------------------
    # DQN MODEL
    # ------------------------------
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )

    model = DQN(
        "CnnPolicy",
        env,
        seed=seed,
        batch_size=32,          # increased batch size
        buffer_size=100_000,
        learning_starts=50_000,
        learning_rate=1e-4,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,        # increased gradient steps
        target_update_interval=10_000,
        exploration_fraction=0.25, #0.4
        exploration_final_eps=0.1,
        tensorboard_log="./tensorboard/",
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=DEVICE,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback, WandbCallback(verbose=2)]
    )

    model.save(base_path + "final_model")
    run.finish()

if __name__ == '__main__':
    train()