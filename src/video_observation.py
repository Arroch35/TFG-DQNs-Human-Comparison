import os
import cv2
import gymnasium as gym
import ale_py

from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# =========================================================
# CONFIG
# =========================================================
GAME = "SpaceInvadersNoFrameskip-v4"
OUTPUT_DIR = "./debug_5_stacks"
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
        env = AtariWrapper(env)
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
        video_path = os.path.join(OUTPUT_DIR, f"SpaceInvaders_stack_{capture_idx+1}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 2, (UPSCALE, UPSCALE))

        for i in range(4):
            frame = stack[:, :, i]

            frame = cv2.resize(frame, (UPSCALE, UPSCALE), interpolation=cv2.INTER_NEAREST)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            cv2.putText(
                frame,
                f"Frame {i+1}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            writer.write(frame)

        writer.release()
        print(f"Saved: {video_path}")

        capture_idx += 1

env.close()

print("Done!")