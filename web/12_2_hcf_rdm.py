import json
import os
import numpy as np
import rsatoolbox
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm
import matplotlib.pyplot as plt

# --------------------------------------------------
# LOAD FEATURES
# --------------------------------------------------

json_file = "../data/jsons/big_rdm_equal_size/pong_hcf_features.json"
save_folder="../data/test_16_rdms/big_rdm_equal_size/pong/hcf/"
os.makedirs(save_folder, exist_ok=True)

with open(json_file, "r") as f:
    features = json.load(f)

# --------------------------------------------------
# BUILD DATA MATRIX
# --------------------------------------------------

clip_names = sorted(features.keys())

X = np.array([
    [
        features[name][0],  # ball_x
        features[name][1],  # ball_y
        features[name][2],  # ball_vx
        features[name][3],  # ball_vy
        features[name][4],  # left paddle
        features[name][5],  # right paddle
    ]
    for name in clip_names
])

print("Data matrix shape:", X.shape)

# --------------------------------------------------
# RSATOOLBOX DATASET
# --------------------------------------------------

dataset = Dataset(
    measurements=X,
    obs_descriptors={
        "clip": clip_names
    }
)

# --------------------------------------------------
# COMPUTE RDM (EUCLIDEAN DISTANCE)
# --------------------------------------------------

rdm = calc_rdm(
    dataset,
    method="euclidean"
)

# --------------------------------------------------
# SAVE RDM
# --------------------------------------------------

rdm_matrix = rdm.get_matrices()[0]


np.save(save_folder + "pong_hcf_rdm.npy", rdm_matrix)

print("RDM shape:", rdm_matrix.shape)

# --------------------------------------------------
# PLOT
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(rdm_matrix, interpolation="nearest")

ax.set_xticks(range(len(clip_names)))
ax.set_yticks(range(len(clip_names)))

ax.set_xticklabels(range(len(clip_names)))

ax.set_yticklabels(range(len(clip_names)))

plt.colorbar(im, ax=ax, label="Euclidean distance")

plt.tight_layout()

plt.savefig(
    save_folder + "pong_hcf_rdm.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: pong_hcf_rdm.png")