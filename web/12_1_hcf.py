import json
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

def to_84x84(frame):
    """
    Resizes a single frame to 84x84.
    
    frame: (H, W, C) or (H, W)
    returns: (84, 84, C) or (84, 84)
    """

    if frame.ndim == 2:
        # grayscale
        resized = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    elif frame.ndim == 3:
        # RGB or BGR
        resized = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    else:
        raise ValueError("Unexpected frame shape: " + str(frame.shape))


def extract_objects(frame):

    # --------------------------------------------------
    # 1. GRAYSCALE + THRESHOLD
    # --------------------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    _, mask = cv2.threshold(
        gray, 90, 255, cv2.THRESH_BINARY)

    # plt.imshow(gray, cmap="gray")
    # plt.axis("off")
    # #plt.title(f"{file} frame {-1}")
    # plt.show()

    # plt.imshow(mask, cmap="gray")
    # plt.axis("off")
    # #plt.title(f"{file} frame {-1}")
    # plt.show()


    # --------------------------------------------------
    # 2. CONNECTED COMPONENTS (initial structure)
    # --------------------------------------------------
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    print("Connected components found:", num)

    # 3. CASE A: normal situation (enough structure)
    if num > 3:

        paddles = []
        ball = None

        h, w = mask.shape

        for i in range(1, num):

            x, y, ww, hh, area = stats[i]
            cx, cy = centroids[i]

            # --------------------------------------------------
            # PADDLE DETECTION (ROBUST TO CROPPING)
            # --------------------------------------------------
            is_narrow = ww > 2
            has_structure = hh > 2

            if is_narrow and has_structure:

                if cx < w * 0.3:
                    paddles.append(("left", cx, cy))

                elif cx > w * 0.7:
                    paddles.append(("right", cx, cy))

            # --------------------------------------------------
            # BALL DETECTION (ONLY IF CLEAN SMALL OBJECT)
            # --------------------------------------------------
            else:
                ball = (cx, cy)

        return paddles, ball


    # 4. CASE B: ambiguous (collision / merge)
    else:

        h, w = mask.shape

        paddles = []
        paddle_mask = np.zeros_like(mask)

        # --------------------------------------------------
        # SLIDING WINDOW (3x6 paddle detector)
        # --------------------------------------------------
        for y in range(h - 6):
            for x in range(w - 3):

                patch = mask[y:y+6, x:x+3]

                filled = np.count_nonzero(patch)

                if filled == 18:  

                    cx = x + 1
                    cy = y + 3

                    # LEFT PADDLE
                    if cx < w * 0.3:
                        if not any(p[0] == "left" for p in paddles):
                            paddles.append(("left", np.float64(cx), np.float64(cy)))
                        paddle_mask[y:y+6, x:x+3] = 1

                    # RIGHT PADDLE
                    elif cx > w * 0.7:
                        if not any(p[0] == "right" for p in paddles):
                            paddles.append(("right", np.float64(cx), np.float64(cy)))
                        paddle_mask[y:y+6, x:x+3] = 1

        # --------------------------------------------------
        # REMOVE ONLY DETECTED PADDLE PIXELS
        # --------------------------------------------------
        remaining = mask.copy()
        remaining[paddle_mask == 1] = 0


        # --------------------------------------------------
        # BALL = SMALLEST REMAINING COMPONENT
        # --------------------------------------------------
        num2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(remaining)
        
        ball = None
        best_area = float("inf")

        for i in range(1, num2):

            area = stats2[i, cv2.CC_STAT_AREA]

            if area < 1:
                continue

            if area < best_area:
                best_area = area
                ball = tuple(centroids2[i])

        return paddles, ball
    

def draw_frame(frame, paddles, ball):

    vis = frame.copy()

    # draw paddles
    for p in paddles:
        print("Paddle:", p)
        side, cx, cy = p
        cx, cy = int(cx), int(cy)

        cv2.circle(vis, (cx, cy), 1, (255, 0, 0), -1)

    # draw ball
    if ball is not None:
        cx, cy = int(ball[0]), int(ball[1])
        cv2.circle(vis, (cx, cy), 1, (0, 0, 255), -1)

    return vis


folder = "../data/test_16_arrays/big_rdm_equal_size/pong/"

FILTER_CSV = None # f"../data/subset_selection/seed_42/pong_best_subset_indices.csv" #028800 044700
if FILTER_CSV is not None:
    filter_df = pd.read_csv(FILTER_CSV)
    print(filter_df.columns)

    allowed_names = list(filter_df['clip_name'].astype(str).str.replace(".mp4", ".npy", regex=False))
else:
    allowed_names = [f for f in os.listdir(folder) if f.endswith(".npy")]


results = {}

for file in allowed_names:

    if not file.endswith(".npy"):
        continue

    frames_file = os.path.join(folder, file)
    frames_array = np.load(frames_file)

    print("Processing:", file, "Shape:", frames_array.shape)

    # ----------------------------------
    # LAST FRAME
    # ----------------------------------
    frame_t = to_84x84(frames_array[-1])
    cropped_t = frame_t[14:76, 0:84]

    paddles_t, ball_t = extract_objects(cropped_t)

    # if ball_t is None:
    #     print("Ball not detected in frame t for file:", file)
    #     vis=draw_frame(cropped_t, paddles_t, ball_t)
    #     plt.imshow(vis)
    #     plt.show()

    # ----------------------------------
    # PENULTIMATE FRAME
    # ----------------------------------
    frame_tm1 = to_84x84(frames_array[-2])
    cropped_tm1 = frame_tm1[14:76, 0:84]

    paddles_tm1, ball_tm1 = extract_objects(cropped_tm1)

    # if ball_t is None:
    #     vis=draw_frame(cropped_tm1, paddles_tm1, ball_tm1)
    #     plt.imshow(vis)
    #     plt.show()


    # ----------------------------------
    # BALL POSITION
    # ----------------------------------
    print("Ball t:", ball_t)
    if ball_t is not None:
        ball_x = ball_t[0]
        ball_y = ball_t[1]
    else:
        ball_x = None
        ball_y = None

    # ----------------------------------
    # BALL VELOCITY
    # ----------------------------------
    if ball_t is not None and ball_tm1 is not None:
        ball_vx = ball_t[0] - ball_tm1[0]
        ball_vy = ball_t[1] - ball_tm1[1]
    else:
        ball_vx = None
        ball_vy = None

    # ----------------------------------
    # PADDLES
    # ----------------------------------
    left_y = None
    right_y = None

    for side, cx, cy in paddles_t:

        if side == "left":
            left_y = cy

        elif side == "right":
            right_y = cy

    # ----------------------------------
    # STORE 6-D VECTOR
    # ----------------------------------

    results[file] = np.array([
        ball_x if ball_x is not None else np.nan,
        ball_y if ball_y is not None else np.nan,
        ball_vx if ball_vx is not None else np.nan,
        ball_vy if ball_vy is not None else np.nan,
        left_y if left_y is not None else np.nan,
        right_y if right_y is not None else np.nan
    ])

    print(file, results[file])

# --- STEP 2: Compute feature means from the populated dictionary ---
# Stack all arrays into one big 2D matrix (ignoring the file keys for a moment)
all_vectors = np.array(list(results.values()))

# Calculate the mean of each column (feature) while ignoring the NaNs
# This gives you an array of 6 means: [mean_x, mean_y, mean_vx, mean_vy, mean_left, mean_right]
feature_means = np.nanmean(all_vectors, axis=0)

# Handle the edge case where a feature is entirely missing across ALL files
# (If a column is all NaN, nanmean outputs NaN; we fallback to 0.0)
feature_means = np.nan_to_num(feature_means, nan=0.0)


# --- STEP 3: Clean the vectors and save to JSON ---
results_json = {}
for k, v in results.items():
    # Find which indices inside this specific vector are NaN
    is_nan_mask = np.isnan(v)
    
    # Replace only the NaN positions with their corresponding feature mean
    v[is_nan_mask] = feature_means[is_nan_mask]
    
    # Now that it's 100% clean numbers, convert to a list for JSON
    results_json[k] = v.tolist()


# convert numpy arrays to lists
# results_json = {
#     k: v.tolist()
#     for k, v in results.items()
# }

os.makedirs("../data/jsons/big_rdm_equal_size/", exist_ok=True)

with open("../data/jsons/big_rdm_equal_size/pong_hcf_features.json", "w") as f:
    json.dump(results_json, f, indent=4)



#? El script me funciona para sacar la posicion de los 3 objetos tanto en el último frame como en el penúltimo.
#TODO: También tendré que comparalo con las RDMs humanas, que no se me olvide
#TODO: HACER tambien esto con las RDM grande, que las correlaciones no salen como esperaba