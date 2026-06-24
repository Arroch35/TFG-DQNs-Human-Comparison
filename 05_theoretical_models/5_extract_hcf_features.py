"""
5_extract_hcf_features.py
Extract hand-crafted features (ball position/velocity, paddle positions)
from Pong frames and save as JSON for RDM computation.
"""
import json
import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from src.config import REFERENCE_SEED, get_path, ensure

# =========================================================
# CONFIG
# =========================================================

ARRAYS_FOLDER = get_path("arrays_bigset_game", game="pong")

# Use the reference-seed subset filter (set to None to process all clips)
FILTER_CSV = get_path("subsets_csv", seed=REFERENCE_SEED, game="pong")  # or None

# Output JSON
HCF_JSON = get_path("jsons_hcf_features")   # data/jsons/big_rdm_equal_size/pong_hcf_features.json
HCF_JSON.parent.mkdir(parents=True, exist_ok=True)

# =========================================================
# HELPERS
# =========================================================
def to_84x84(frame):
    if frame.ndim == 2:
        return cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    elif frame.ndim == 3:
        return cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    else:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")


def extract_objects(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    # Case A: normal situation
    if num > 3:
        paddles, ball = [], None
        h, w = mask.shape

        for i in range(1, num):
            x, y, ww, hh, area = stats[i]
            cx, cy = centroids[i]

            if ww > 2 and hh > 2:
                if cx < w * 0.3:
                    paddles.append(("left", cx, cy))
                elif cx > w * 0.7:
                    paddles.append(("right", cx, cy))
            else:
                ball = (cx, cy)

        return paddles, ball

    # Case B: ambiguous / collision
    h, w = mask.shape
    paddles, paddle_mask = [], np.zeros_like(mask)

    for y in range(h - 6):
        for x in range(w - 3):
            if np.count_nonzero(mask[y:y+6, x:x+3]) == 18:
                cx, cy = x + 1, y + 3
                if cx < w * 0.3 and not any(p[0] == "left" for p in paddles):
                    paddles.append(("left", np.float64(cx), np.float64(cy)))
                    paddle_mask[y:y+6, x:x+3] = 1
                elif cx > w * 0.7 and not any(p[0] == "right" for p in paddles):
                    paddles.append(("right", np.float64(cx), np.float64(cy)))
                    paddle_mask[y:y+6, x:x+3] = 1

    remaining = mask.copy()
    remaining[paddle_mask == 1] = 0

    num2, _, stats2, centroids2 = cv2.connectedComponentsWithStats(remaining)
    ball, best_area = None, float("inf")
    for i in range(1, num2):
        area = stats2[i, cv2.CC_STAT_AREA]
        if 1 <= area < best_area:
            best_area = area
            ball = tuple(centroids2[i])

    return paddles, ball

# =========================================================
# CLIP SELECTION
# =========================================================
if FILTER_CSV is not None and FILTER_CSV.exists():
    filter_df     = pd.read_csv(FILTER_CSV)
    allowed_names = list(filter_df["clip_name"].astype(str).str.replace(".mp4", ".npy", regex=False))
else:
    allowed_names = [f for f in os.listdir(ARRAYS_FOLDER) if f.endswith(".npy")]

# =========================================================
# FEATURE EXTRACTION
# =========================================================
results = {}

for file in allowed_names:
    if not file.endswith(".npy"):
        continue

    frames_array = np.load(ARRAYS_FOLDER / file)
    print(f"Processing: {file}  shape: {frames_array.shape}")

    frame_t   = to_84x84(frames_array[-1])[14:76, 0:84]
    frame_tm1 = to_84x84(frames_array[-2])[14:76, 0:84]

    paddles_t,   ball_t   = extract_objects(frame_t)
    paddles_tm1, ball_tm1 = extract_objects(frame_tm1)

    ball_x  = ball_t[0]  if ball_t  is not None else None
    ball_y  = ball_t[1]  if ball_t  is not None else None
    ball_vx = (ball_t[0] - ball_tm1[0]) if (ball_t and ball_tm1) else None
    ball_vy = (ball_t[1] - ball_tm1[1]) if (ball_t and ball_tm1) else None

    left_y, right_y = None, None
    for side, cx, cy in paddles_t:
        if side == "left":
            left_y = cy
        elif side == "right":
            right_y = cy

    results[file] = np.array([
        ball_x  if ball_x  is not None else np.nan,
        ball_y  if ball_y  is not None else np.nan,
        ball_vx if ball_vx is not None else np.nan,
        ball_vy if ball_vy is not None else np.nan,
        left_y  if left_y  is not None else np.nan,
        right_y if right_y is not None else np.nan,
    ])
    print(file, results[file])

# =========================================================
# IMPUTE MISSING VALUES WITH FEATURE MEANS
# =========================================================
all_vectors  = np.array(list(results.values()))
feature_means = np.nan_to_num(np.nanmean(all_vectors, axis=0), nan=0.0)

results_json = {}
for k, v in results.items():
    nan_mask  = np.isnan(v)
    v[nan_mask] = feature_means[nan_mask]
    results_json[k] = v.tolist()

# =========================================================
# SAVE
# =========================================================
with open(HCF_JSON, "w") as f:
    json.dump(results_json, f, indent=4)

print(f"Saved HCF features → {HCF_JSON}")
