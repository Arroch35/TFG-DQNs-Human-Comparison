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

# Real video clips
CLIP_BASE_FOLDER = "../data/test_16_clips"

# SA-selected subsets
SA_FOLDER = f"../data/subset_selection/{SEED}"

# Output
SAVE_FOLDER = (
    f"../data/triplet_visualization_subset/"
    f"selected_15/{SEED}/filtered_all_difficulties"
)

DIM = 2
N_REPEATS = 100
MAX_ITER = 1000

# =========================================================
# DIFFICULTY CONFIG
# =========================================================

# easy = top 20%
EASY_PERCENT = 0.2

# hard = bottom 20%
HARD_PERCENT = 0.2

# medium = middle 40-60%
MEDIUM_LOW = 0.4
MEDIUM_HIGH = 0.6

# =========================================================
# STRUCTURE FILTER
# =========================================================

# discard bottom X% least-structured triplets
STRUCTURE_PERCENTILE = 20

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

        # smallest distance defines similar pair
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
# ADD SYMMETRIC TRIPLETS BACK
# =========================================================
def add_symmetric_triplets(triplets):

    existing = set(tuple(t) for t in triplets)

    result = list(triplets)

    for i, j, k in triplets:

        symmetric_counterpart = (j, i, k)

        if symmetric_counterpart not in existing:

            existing.add(symmetric_counterpart)

            result.append(symmetric_counterpart)

    return np.array(result, dtype=np.int32)

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
# SAVE TRIPLETS
# =========================================================
def save_triplets(
    triplets_array,
    difficulty_name,
    game_out,
    new_to_orig,
    index_to_clip,
    used_clips,
    game
):

    print(f"\nSaving {difficulty_name} triplets...")

    diff_folder = os.path.join(
        game_out,
        difficulty_name
    )

    os.makedirs(diff_folder, exist_ok=True)

    for idx, (i, j, k) in enumerate(
        tqdm(triplets_array)
    ):

        triplet_folder = os.path.join(
            diff_folder,
            f"triplet_{idx:06d}"
        )

        os.makedirs(
            triplet_folder,
            exist_ok=True
        )

        # local subset idx -> original idx
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

            print(
                f"Error copying "
                f"{difficulty_name} triplet {idx}: {e}"
            )


# =========================================================
# SAVE TRIPLET CSV
# =========================================================
def save_triplet_csv(
    triplets_array,
    csv_path
):

    rows = []

    for i, j, k in triplets_array:

        rows.append({
            "similar1": int(i),
            "similar2": int(j),
            "odd": int(k)
        })

    df = pd.DataFrame(rows)

    df.to_csv(
        csv_path,
        index=False
    )

# =========================================================
# SAVE CLIP INDEX MAP
# =========================================================
def save_clip_index_map(
    subset_df,
    csv_path
):

    mapping_df = (
        subset_df[
            ["clip_index", "clip_name"]
        ]
        .sort_values("clip_index")
        .reset_index(drop=True)
    )

    mapping_df.columns = [
        "index",
        "clip_name"
    ]

    mapping_df.to_csv(
        csv_path,
        index=False
    )

# =========================================================
# MAIN
# =========================================================
for game in GAMES:

    print("\n" + "=" * 60)
    print(f"GAME: {game}")
    print("=" * 60)

    # =====================================================
    # LOAD SUBSET CSV
    # =====================================================
    subset_csv = os.path.join(
        SA_FOLDER,
        f"{game}_best_subset_indices.csv"
    )

    if not os.path.exists(subset_csv):

        print(f"Missing subset CSV: {subset_csv}")
        continue

    subset_df = pd.read_csv(subset_csv)

    SELECTED_INDICES = subset_df[
        "clip_index"
    ].tolist()

    print(
        f"Loaded subset with "
        f"{len(SELECTED_INDICES)} clips"
    )

    # =====================================================
    # LOAD RDM
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

    print(f"RDM shape: {rdm.shape}")

    # =====================================================
    # LOCAL -> ORIGINAL INDEX
    # =====================================================
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
    # REMOVE SYMMETRIC DUPLICATES
    # =====================================================
    all_triplets = remove_symmetric_triplets(
        all_triplets
    )

    print(
        f"Unique triplets: "
        f"{len(all_triplets)}"
    )

    # =====================================================
    # STRUCTURE + DIFFICULTY SCORES
    # =====================================================
    scores = np.zeros(len(all_triplets))

    structure_scores = np.zeros(
        len(all_triplets)
    )

    for idx, (i, j, k) in enumerate(all_triplets):

        # similar pair
        dij = rdm[i, j]

        # odd distances
        dik = rdm[i, k]
        djk = rdm[j, k]

        # -------------------------------------------------
        # STRUCTURE SCORE
        # -------------------------------------------------
        d1, d2, d3 = sorted([
            dij,
            dik,
            djk
        ])

        structure_scores[idx] = (
            (d3 - d1)
            / (d3 + 1e-8)
        )

        # -------------------------------------------------
        # DIFFICULTY SCORE
        # -------------------------------------------------
        d_odd = (dik + djk) / 2.0

        scores[idx] = (
            (d_odd - dij)
            / (d_odd + dij + 1e-8)
        )

    # =====================================================
    # REMOVE LOW-STRUCTURE TRIPLETS
    # =====================================================
    structure_threshold = np.percentile(
        structure_scores,
        STRUCTURE_PERCENTILE
    )

    keep_mask = (
        structure_scores > structure_threshold
    )

    all_triplets = all_triplets[keep_mask]
    scores = scores[keep_mask]
    structure_scores = structure_scores[keep_mask]

    print(
        f"\nStructure threshold "
        f"({STRUCTURE_PERCENTILE}%): "
        f"{structure_threshold:.4f}"
    )

    print(
        f"Remaining structured triplets: "
        f"{len(all_triplets)}"
    )

    print(
        f"Structure score range: "
        f"{structure_scores.min():.4f} "
        f"to "
        f"{structure_scores.max():.4f}"
    )

    # =====================================================
    # SORT TRIPLETS BY DIFFICULTY
    # =====================================================
    n_total = len(all_triplets)

    sorted_idxs = np.argsort(scores)

    # -----------------------------------------------------
    # HARD
    # -----------------------------------------------------
    hard_end = int(n_total * HARD_PERCENT)

    hard_triplets = all_triplets[
        sorted_idxs[:hard_end]
    ]

    print(
        f"\nHard triplets: "
        f"{min(scores[sorted_idxs[:hard_end]]):.4f} "
        f"to "
        f"{max(scores[sorted_idxs[:hard_end]]):.4f}"
    )

    # -----------------------------------------------------
    # MEDIUM
    # -----------------------------------------------------
    medium_start = int(n_total * MEDIUM_LOW)

    medium_end = int(n_total * MEDIUM_HIGH)

    medium_triplets = all_triplets[
        sorted_idxs[medium_start:medium_end]
    ]

    print(
        f"Medium triplets: "
        f"{min(scores[sorted_idxs[medium_start:medium_end]]):.4f} "
        f"to "
        f"{max(scores[sorted_idxs[medium_start:medium_end]]):.4f}"
    )

    # -----------------------------------------------------
    # EASY
    # -----------------------------------------------------
    easy_start = int(
        n_total * (1.0 - EASY_PERCENT)
    )

    easy_triplets = all_triplets[
        sorted_idxs[easy_start:]
    ]

    print(
        f"Easy triplets: "
        f"{min(scores[sorted_idxs[easy_start:]]):.4f} "
        f"to "
        f"{max(scores[sorted_idxs[easy_start:]]):.4f}"
    )

    print(
        f"\nHard triplets:   "
        f"{len(hard_triplets)}"
    )

    print(
        f"Medium triplets: "
        f"{len(medium_triplets)}"
    )

    print(
        f"Easy triplets:   "
        f"{len(easy_triplets)}"
    )

    # =====================================================
    # INDEX -> CLIP MAP
    # =====================================================
    index_to_clip = dict(
        zip(
            subset_df["clip_index"],
            subset_df["clip_name"]
        )
    )

    # =====================================================
    # t-STE RECONSTRUCTION
    # =====================================================
    print("\nRunning t-STE reconstruction...")

    best_score = -np.inf

    reconstruction_triplets = np.concatenate([
        hard_triplets,
        medium_triplets,
        easy_triplets
    ])

    reconstruction_triplets = (
        add_symmetric_triplets(
            reconstruction_triplets
        )
    )

    for repeat in tqdm(range(N_REPEATS)):

        X = cy_tste.tste(
            reconstruction_triplets,
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

    print(
        f"Best score: "
        f"{best_score:.4f}"
    )

    # =====================================================
    # SAVE RESULTS
    # =====================================================
    game_out = os.path.join(
        SAVE_FOLDER,
        game
    )

    os.makedirs(game_out, exist_ok=True)


    # =====================================================
    # SAVE CLIP INDEX MAP
    # =====================================================
    clip_map_csv = os.path.join(
        game_out,
        f"{game}_clip_index_map.csv"
    )

    save_clip_index_map(
        subset_df,
        clip_map_csv
    )

    with open(
        os.path.join(game_out, "best_score.txt"),
        "w"
    ) as f:

        f.write(f"{best_score:.6f}")

    # =====================================================
    # SAVE TRIPLETS
    # =====================================================
    used_clips = set()

    difficulty_sets = {
        "easy": easy_triplets,
        "medium": medium_triplets,
        "hard": hard_triplets
    }

    for difficulty_name, triplets_array in (
        difficulty_sets.items()
    ):

        # =================================================
        # SAVE TRIPLET CSV
        # =================================================
        triplet_csv_path = os.path.join(
            game_out,
            f"{difficulty_name}_triplets.csv"
        )

        save_triplet_csv(
            triplets_array,
            triplet_csv_path
        )

        save_triplets(
            triplets_array=triplets_array,
            difficulty_name=difficulty_name,
            game_out=game_out,
            new_to_orig=new_to_orig,
            index_to_clip=index_to_clip,
            used_clips=used_clips,
            game=game
        )

    # =====================================================
    # COPY UNIQUE CLIPS
    # =====================================================
    print("\nCopying unique clips...")

    clips_out = os.path.join(
        game_out,
        "clips"
    )

    os.makedirs(clips_out, exist_ok=True)

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

            print(
                f"Error copying clip "
                f"{clip_name}: {e}"
            )

print("\nDONE.")