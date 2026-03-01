import numpy as np
import cv2

# ----------------------------
# 1. Load Data
# ----------------------------
data = np.load("human_2min_data.npz")

frames = data["frames"]  # shape: (N, H, W, 3)
print(f"Loaded {len(frames)} frames.")
print(f"Frame shape: {frames[0].shape}")

# ----------------------------
# 2. Video Settings
# ----------------------------
output_file = "human_2min_video.mp4"
fps = 60  # same as play() fps
height, width, channels = frames[0].shape

# Define codec and create VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# ----------------------------
# 3. Write Frames
# ----------------------------
for frame in frames:
    # OpenCV expects BGR, but Gym gives RGB
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video_writer.write(frame_bgr)

video_writer.release()

print(f"Video saved as {output_file}")