import numpy as np
import os
import pandas as pd
import shutil
from itertools import combinations
from scipy.stats import spearmanr
from tqdm import tqdm
import cy_tste

# =========================================================
# CONFIG
# =========================================================
GAMES = ["pacman", "pong", "spaceinvaders"]

METHOD = "correlation"

SEED = "seed_42"

# Full RDMs computed from buenos_25 activations
BASE_RDM_FOLDER = f"../data/test_16_rdms/selected_subset_15/{SEED}"

# Original clip maps
MAP_FOLDER = "../data/maps/buenos_25"

# Real video clips
CLIP_BASE_FOLDER = "../data/test_16_clips"

# SA-selected subsets
SA_FOLDER = f"../data/subset_selection/{SEED}"

# Output
SAVE_FOLDER = f"../data/triplet_visualization_subset/selected_15/{SEED}"

DIM = 2
N_REPEATS = 100
MAX_ITER = 1000

# Keep top X% most informative triplets
PERCENT = 0.2

os.makedirs(SAVE_FOLDER, exist_ok=True)

# =========================================================
# BUILD TRIPLETS
# =========================================================
def build_triplets_from_rdm(rdm):

    triplets = []

    for i, j, k in combinations(range(len(rdm)), 3):

        dij = rdm[i, j]
        dik = rdm[i, k]
        djk = rdm[j, k]

        if dij <= dik and dij <= djk:

            triplets.append((i, j, k))
            triplets.append((j, i, k))

        elif dik <= dij and dik <= djk:

            triplets.append((i, k, j))
            triplets.append((k, i, j))

        else:

            triplets.append((j, k, i))
            triplets.append((k, j, i))

    return np.array(triplets, dtype=np.int32)

# =========================================================
# REMOVE SYMMETRIC DUPLICATES
# =========================================================
def remove_symmetric_triplets(triplets):

    seen = set()
    unique = []

    for i, j, k in triplets:

        key = (min(i, j), max(i, j), k)

        if key not in seen:

            seen.add(key)
            unique.append((i, j, k))

    return np.array(unique, dtype=np.int32)

# =========================================================
# EMBEDDING → RDM
# =========================================================
def embedding_to_rdm(X):

    diff = X[:, None, :] - X[None, :, :]

    return np.linalg.norm(diff, axis=-1)

# =========================================================
# COMPARE RDMS
# =========================================================
def compare_rdms(rdm1, rdm2):

    idx = np.triu_indices_from(rdm1, k=1)

    return spearmanr(
        rdm1[idx],
        rdm2[idx]
    ).correlation

# =========================================================
# MAIN
# =========================================================
for game in GAMES:

    print("\n" + "=" * 60)
    print(f"GAME: {game}")
    print("=" * 60)

    # =====================================================
    # LOAD SELECTED SUBSET CSV
    # =====================================================
    subset_csv = os.path.join(
        SA_FOLDER,
        f"{game}_best_subset_indices.csv"
    )

    if not os.path.exists(subset_csv):

        print(f"Missing subset CSV: {subset_csv}")
        continue

    subset_df = pd.read_csv(subset_csv)

    # Original indices inside buenos_25 RDM
    SELECTED_INDICES = subset_df["clip_index"].tolist()

    print(f"Loaded subset with {len(SELECTED_INDICES)} clips")

    # =====================================================
    # LOAD FULL RDM
    # =====================================================
    rdm_path = os.path.join(
        BASE_RDM_FOLDER,
        game,
        f"{game}_fc_correlation_RDM.npy"
    )

    if not os.path.exists(rdm_path):

        print(f"Missing RDM: {rdm_path}")
        continue

    rdm = np.load(rdm_path)

    # # =====================================================
    # # EXTRACT SUBSET RDM
    # # =====================================================
    # rdm = rdm_full[np.ix_(
    #     SELECTED_INDICES,
    #     SELECTED_INDICES
    # )]

    # print(f"Subset RDM shape: {rdm.shape}")

    # mapping:
    # local index in subset -> original buenos_25 index
    new_to_orig = {
        i: orig
        for i, orig in enumerate(SELECTED_INDICES)
    }

    # =====================================================
    # BUILD TRIPLETS
    # =====================================================
    all_triplets = build_triplets_from_rdm(rdm)

    print(f"Total triplets: {len(all_triplets)}")

    # =====================================================
    # SCORE TRIPLETS
    # =====================================================
    scores = np.zeros(len(all_triplets))

    for idx, (i, j, k) in enumerate(all_triplets):

        dij = rdm[i, j]
        dik = rdm[i, k]

        scores[idx] = (
            (dik - dij)
            / (dik + dij + 1e-8)
        )

    # =====================================================
    # KEEP TOP %
    # =====================================================
    n_triplets = int(len(all_triplets) * PERCENT)

    top_idxs = np.argsort(scores)[-n_triplets:]

    selected_triplets = all_triplets[top_idxs]

    # Remove symmetric duplicates
    selected_triplets = remove_symmetric_triplets(
        selected_triplets
    )

    print(
        f"After removing duplicates: "
        f"{len(selected_triplets)} triplets"
    )

    # =====================================================
    # BUILD INDEX → CLIP MAP
    # =====================================================
    index_to_clip = dict(
        zip(
            subset_df["clip_index"],
            subset_df["clip_name"]
        )
    )

    # =====================================================
    # RUN t-STE
    # =====================================================
    best_score = -np.inf

    for repeat in tqdm(range(N_REPEATS)):

        X = cy_tste.tste(
            selected_triplets,
            no_dims=DIM,
            max_iter=MAX_ITER,
            verbose=False,
            use_log=True
        )

        rdm_rec = embedding_to_rdm(X)

        score = compare_rdms(
            rdm,
            rdm_rec
        )

        if score > best_score:

            best_score = score

    print(f"Best score: {best_score:.4f}")

    # =====================================================
    # SAVE RESULTS
    # =====================================================
    game_out = os.path.join(
        SAVE_FOLDER,
        game
    )

    os.makedirs(game_out, exist_ok=True)

    with open(
        os.path.join(game_out, "best_score.txt"),
        "w"
    ) as f:

        f.write(f"{best_score:.6f}")

    # =====================================================
    # COPY ALL USED CLIPS
    # =====================================================
    clips_out = os.path.join(
        game_out,
        "clips"
    )

    os.makedirs(clips_out, exist_ok=True)

    used_clips = set()

    # =====================================================
    # SAVE TRIPLETS
    # =====================================================
    for idx, (i, j, k) in enumerate(
        tqdm(selected_triplets)
    ):

        triplet_folder = os.path.join(
            game_out,
            f"triplet_{idx:06d}"
        )

        os.makedirs(
            triplet_folder,
            exist_ok=True
        )

        # convert subset-local idx
        # back to original buenos_25 idx
        orig_i = new_to_orig[i]
        orig_j = new_to_orig[j]
        orig_k = new_to_orig[k]

        clip_i = index_to_clip[orig_i]
        clip_j = index_to_clip[orig_j]
        clip_k = index_to_clip[orig_k]

        used_clips.add(clip_i)
        used_clips.add(clip_j)
        used_clips.add(clip_k)

        src_i = os.path.join(
            CLIP_BASE_FOLDER,
            game,
            "buenos_25/human_dqn_visualitzation",
            clip_i
        )

        src_j = os.path.join(
            CLIP_BASE_FOLDER,
            game,
            "buenos_25/human_dqn_visualitzation",
            clip_j
        )

        src_k = os.path.join(
            CLIP_BASE_FOLDER,
            game,
            "buenos_25/human_dqn_visualitzation",
            clip_k
        )

        dst_i = os.path.join(
            triplet_folder,
            f"clip1_{clip_i}"
        )

        dst_j = os.path.join(
            triplet_folder,
            f"clip2_{clip_j}"
        )

        dst_k = os.path.join(
            triplet_folder,
            f"odd_{clip_k}"
        )

        try:

            shutil.copy(src_i, dst_i)
            shutil.copy(src_j, dst_j)
            shutil.copy(src_k, dst_k)

        except Exception as e:

            print(f"Error copying triplet {idx}: {e}")

    # =====================================================
    # COPY UNIQUE CLIPS
    # =====================================================
    for clip_name in tqdm(used_clips):

        src = os.path.join(
            CLIP_BASE_FOLDER,
            game,
            "buenos_25/human_dqn_visualitzation",
            clip_name
        )

        dst = os.path.join(
            clips_out,
            clip_name
        )

        try:

            if not os.path.exists(dst):

                shutil.copy(src, dst)

        except Exception as e:

            print(f"Error copying clip {clip_name}: {e}")

print("\nDONE.")