import gymnasium as gym
import ale_py
import os
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor

from models.custom_dqn import CustomCNN
from wrappers.environment_wrappers import (
    RestrictPongActions,
    RestrictMsPacmanActions,
    RestrictSpaceInvadorsActions
)

# --- CONFIGURATION SETTINGS ---
GAME_CONFIGS = {
    "MsPacman": {
        "env_id": "MsPacmanNoFrameskip-v4",
        "wrapper": RestrictMsPacmanActions,
        "model_root": "../models/MsPacmanNoFrameskip-v4"
    },
    "Pong": {
        "env_id": "PongNoFrameskip-v4",
        "wrapper": RestrictPongActions,
        "model_root": "../models/PongNoFrameskip-v4"
    },
    "SpaceInvaders": {
        "env_id": "SpaceInvadersNoFrameskip-v4",
        "wrapper": RestrictSpaceInvadorsActions,
        "model_root": "../models/SpaceInvadersNoFrameskip-v4"
    }
}

# --- SETUP FUNCTIONS ---
def make_env(env_id, wrapper_cls):
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env = AtariWrapper(env)
        env = wrapper_cls(env)
        env = Monitor(env)
        return env
    return _init


# --------------------------------------------------
# MAIN LOOP (ITERATION ADDED HERE)
# --------------------------------------------------

for CURRENT_GAME, config in GAME_CONFIGS.items():

    model_root = config["model_root"]

    # find all seeds
    seed_dirs = sorted([
        d for d in os.listdir(model_root)
        if os.path.isdir(os.path.join(model_root, d))
    ])

    for seed in seed_dirs:

        model_path = os.path.join(model_root, seed, "final_model")

        print("=" * 60)
        print(f"Game: {CURRENT_GAME}")
        print(f"Seed: {seed}")
        print("=" * 60)

        # Create env exactly like your original code
        env = DummyVecEnv([make_env(config["env_id"], config["wrapper"])])
        env = VecFrameStack(env, n_stack=4)

        # Video setup
        video_folder = f"../data/videos/{CURRENT_GAME}/{seed}/"
        os.makedirs(video_folder, exist_ok=True)

        video_length = 5000

        env = VecVideoRecorder(
            env,
            video_folder,
            record_video_trigger=lambda x: x == 0,
            video_length=video_length,
            name_prefix=f"dqn-{CURRENT_GAME}-{seed}"
        )

        # Model (UNCHANGED logic)
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
            verbose=0
        )

        print(f"Loading weights from {model_path}")
        model.set_parameters(model_path, exact_match=True)

        # Execution (UNCHANGED logic)
        env.seed(1000)
        obs = env.reset()
        print(f"Recording started. Saving to {video_folder}")

        for _ in range(video_length):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            if done:
                break

        env.close()

        print(f"Finished {CURRENT_GAME} - {seed}")

print("\nAll videos generated.")