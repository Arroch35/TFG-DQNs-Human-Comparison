import os
import cv2
import gymnasium as gym
import ale_py

from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from wrappers.custom_atari_wrapper import CustomAtariWrapper

# =========================================================
# CONFIG
# =========================================================
GAME = "MsPacmanNoFrameskip-v4"
OUTPUT_DIR = "./debug_5_stacks/human_readable"
os.makedirs(OUTPUT_DIR, exist_ok=True)

UPSCALE = 336

# When to capture stacks (different game moments)
CAPTURE_STEPS = [200, 400, 600, 800, 1000]

# =========================================================
# ENV
# =========================================================
def make_env():
    def _init():
        env = gym.make(GAME)
        env = CustomAtariWrapper(env)
        return env
    return _init

env = DummyVecEnv([make_env()])
env = VecFrameStack(env, n_stack=4)

obs = env.reset()

print("Starting rollout...")

# =========================================================
# MAIN LOOP
# =========================================================
current_step = 0
capture_idx = 0

while capture_idx < len(CAPTURE_STEPS):
    action = [env.action_space.sample()]
    obs, _, dones, _ = env.step(action)
    current_step += 1

    if dones[0]:
        obs = env.reset()

    # Check if we should capture a stack here
    if current_step == CAPTURE_STEPS[capture_idx]:
        print(f"Capturing stack {capture_idx+1} at step {current_step}")

        stack = obs[0]  # (84,84,4)

        # =========================================================
        # SAVE VIDEO
        # =========================================================
        video_path = os.path.join(OUTPUT_DIR, f"MsPacman_stack_{capture_idx+1}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 2, (160, 210))

        for i in range(4):
            frame = stack[:, :, i*3:(i+1)*3]  # (H,W,3)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frame = cv2.resize(frame, (160, 210), interpolation=cv2.INTER_NEAREST)

            print(stack.shape)
            print(stack[:, :, :3].mean(), stack[:, :, 3:6].mean())


            writer.write(frame)

        writer.release()
        print(f"Saved: {video_path}")

        capture_idx += 1

env.close()

print("Done!")