# src/utils.py
import re
import numpy as np
import cv2
from scipy.stats import spearmanr

# ── DQN preprocessing ──────────────────────────────────────
def dqn_preprocess_from_16_frames(frames_16, human=False):
    assert frames_16.shape[0] == 16
    processed = []
    for t in [3, 7, 11, 15]:
        pooled  = np.maximum(frames_16[t], frames_16[t - 1])
        if not human:
            gray    = cv2.cvtColor(pooled, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        processed.append(resized)
    return np.stack(processed, axis=0).astype(np.float32) / 255.0

# ── Activation key parsing ──────────────────────────────────
def extract_layer_name(key):
    match = re.search(r"(conv\d+|fc)$", key)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract layer name from key: {key}")

# ── RDM utilities ───────────────────────────────────────────
def upper_tri(rdm):
    return rdm[np.triu_indices_from(rdm, k=1)]

def rdm_spearman(rdm1, rdm2):
    return spearmanr(upper_tri(rdm1), upper_tri(rdm2)).correlation

def embedding_to_rdm(X):
    diff = X[:, None, :] - X[None, :, :]
    return np.linalg.norm(diff, axis=-1)

# ── Triplet utilities ───────────────────────────────────────
def build_triplets_from_rdm(rdm):
    from itertools import combinations
    triplets = []
    for i, j, k in combinations(range(len(rdm)), 3):
        dij, dik, djk = rdm[i,j], rdm[i,k], rdm[j,k]
        if dij <= dik and dij <= djk:
            triplets += [(i,j,k), (j,i,k)]
        elif dik <= dij and dik <= djk:
            triplets += [(i,k,j), (k,i,j)]
        else:
            triplets += [(j,k,i), (k,j,i)]
    return np.array(triplets, dtype=np.int32)

def remove_symmetric_triplets(triplets):
    seen, unique = set(), []
    for i, j, k in triplets:
        key = (min(i,j), max(i,j), k)
        if key not in seen:
            seen.add(key)
            unique.append((i, j, k))
    return np.array(unique, dtype=np.int32)

def add_symmetric_triplets(triplets):
    existing = set(tuple(t) for t in triplets)
    result = list(triplets)
    for i, j, k in triplets:
        sym = (j, i, k)
        if sym not in existing:
            existing.add(sym)
            result.append(sym)
    return np.array(result, dtype=np.int32)

# ── Human response utilities ────────────────────────────────
def majority_vote_with_ties(group):
    counts    = group["odd_clip_idx"].value_counts()
    max_count = counts.iloc[0]
    return counts[counts == max_count].index.tolist()