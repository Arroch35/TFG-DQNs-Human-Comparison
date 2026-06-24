"""
6_compute_hcf_rdm.py
Compute Euclidean RDM from Pong hand-crafted features JSON
produced by 5_extract_hcf_features.py.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm

from src.config import get_path, ensure

# =========================================================
# PATHS (both already defined in config.py)
# =========================================================
HCF_JSON    = get_path("jsons_hcf_features")            
SAVE_FOLDER = get_path("rdms_hcf").parent              
SAVE_FOLDER.mkdir(parents=True, exist_ok=True)

RDM_SAVE    = get_path("rdms_hcf")                   
PNG_SAVE    = SAVE_FOLDER / "pong_hcf_rdm.png"

# =========================================================
# LOAD & BUILD DATA MATRIX
# =========================================================
with open(HCF_JSON, "r") as f:
    features = json.load(f)

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

# =========================================================
# COMPUTE RDM
# =========================================================
dataset = Dataset(
    measurements=X,
    obs_descriptors={"clip": clip_names},
)

rdm        = calc_rdm(dataset, method="euclidean")
rdm_matrix = rdm.get_matrices()[0]

print("RDM shape:", rdm_matrix.shape)

# =========================================================
# SAVE
# =========================================================
np.save(RDM_SAVE, rdm_matrix)
print(f"Saved RDM → {RDM_SAVE}")

# =========================================================
# PLOT
# =========================================================
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(rdm_matrix, interpolation="nearest")

ticks = range(len(clip_names))
ax.set_xticks(ticks); ax.set_xticklabels(ticks)
ax.set_yticks(ticks); ax.set_yticklabels(ticks)

plt.colorbar(im, ax=ax, label="Euclidean distance")
plt.tight_layout()
plt.savefig(PNG_SAVE, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved plot → {PNG_SAVE}")
