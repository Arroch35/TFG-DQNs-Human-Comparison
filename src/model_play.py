import gymnasium as gym
import ale_py
import os
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor

# IMPORTANT: RecordVideo comes from gymnasium (NOT SB3 vec env)
from gymnasium.wrappers import RecordVideo

# Import your custom modules
from models.custom_dqn import CustomCNN
from wrappers.environment_wrappers import (
    RestrictPongActions,
    RestrictMsPacmanActions,
    RestrictSpaceInvadorsActions,
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

GAME_CONFIGS = {
    "MsPacman": {
        "env_id": "MsPacmanNoFrameskip-v4",
        "wrapper": RestrictMsPacmanActions,
        "model_root": "../models/MsPacmanNoFrameskip-v4",
    },
    "Pong": {
        "env_id": "PongNoFrameskip-v4",
        "wrapper": RestrictPongActions,
        "model_root": "../models/ALE/Pong-v5",
    },
    "SpaceInvaders": {
        "env_id": "SpaceInvadersNoFrameskip-v4",
        "wrapper": RestrictSpaceInvadorsActions,
        "model_root": "../models/SpaceInvadersNoFrameskip-v4",
    },
}

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

for game_name, config in GAME_CONFIGS.items():

    model_root = config["model_root"]

    seed_dirs = sorted(
        [
            d
            for d in os.listdir(model_root)
            if os.path.isdir(os.path.join(model_root, d))
            and d.startswith("seed_")
        ]
    )

    for seed in seed_dirs:

        model_path = os.path.join(model_root, seed, "final_model")

        seed_value = int(seed.split("_")[1])

        print("=" * 60)
        print(f"Game: {game_name}")
        print(f"Seed: {seed}")
        print("=" * 60)

        def make_env():
            env = gym.make(
                config["env_id"],
                render_mode="rgb_array",
            )
            env = AtariWrapper(env)
            env = config["wrapper"](env)
            env = Monitor(env)
            return env

        # -----------------------------
        # Video folder
        # -----------------------------
        video_folder = os.path.join("../data/videos", game_name, seed)
        os.makedirs(video_folder, exist_ok=True)

        # -----------------------------
        # Record full episodes
        # -----------------------------
        env = gym.make(
            config["env_id"],
            render_mode="rgb_array",
        )
        env = AtariWrapper(env)
        env = config["wrapper"](env)
        env = Monitor(env)

        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda episode_id: True,  # record EVERY episode
            name_prefix=f"dqn-{game_name}-{seed}",
        )

        # -----------------------------
        # Load model (no env needed for SB3 inference)
        # -----------------------------
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=512),
        )

        model = DQN(
            "CnnPolicy",
            env=None,  # IMPORTANT: not needed for inference
            policy_kwargs=policy_kwargs,
            buffer_size=1,
            learning_starts=0,
            verbose=0,
        )

        print(f"Loading weights from {model_path}")
        model.set_parameters(model_path, exact_match=True)

        # -----------------------------
        # Run episode
        # -----------------------------
        obs, info = env.reset(seed=1000)

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        env.close()

        print(f"Finished recording {game_name} - {seed}")

print("\nAll videos have been generated.")