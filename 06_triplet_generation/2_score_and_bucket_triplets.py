"""
1_score_and_bucket_triplets.py
Score all triplets from the SA-selected subset, bucket them by difficulty,
copy clips into per-triplet folders, and run a t-STE reconstruction check.
"""
import os
import shutil
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from src.config import (
    GAMES, REFERENCE_SEED, REPR, TSTE,
    get_path, ensure,
)
from src.utils import (
    embedding_to_rdm, build_triplets_from_rdm,
    remove_symmetric_triplets, add_symmetric_triplets,
)
import cy_tste

# =========================================================
# CONFIG
# =========================================================
SEED   = REFERENCE_SEED             # "seed_42"
METHOD = REPR["rdm_method"]         # "correlation"

DIM      = TSTE["dim"]              # 2
N_REPEAT = TSTE["n_repeats"]        # 100
MAX_ITER = TSTE["max_iter"]         # 1000

EASY_PERCENT         = TSTE["easy_percent"]       # 0.2
HARD_PERCENT         = TSTE["hard_percent"]       # 0.2
MEDIUM_LOW           = TSTE["medium_low"]         # 0.4
MEDIUM_HIGH          = TSTE["medium_high"]        # 0.6
STRUCTURE_PERCENTILE = TSTE["structure_pctile"]   # 20

SAVE_FOLDER = get_path("triplet_viz",seed=SEED)

# =========================================================
# HELPERS
# =========================================================
def compare_rdms(rdm1, rdm2):
    idx = np.triu_indices_from(rdm1, k=1)
    return spearmanr(rdm1[idx], rdm2[idx]).correlation


def save_triplets(triplets_array, difficulty_name, game_out, new_to_orig,
                  index_to_clip, used_clips, game):
    print(f"\nSaving {difficulty_name} triplets...")
    diff_folder = game_out / difficulty_name
    diff_folder.mkdir(parents=True, exist_ok=True)

    for idx, (i, j, k) in enumerate(tqdm(triplets_array)):
        triplet_folder = diff_folder / f"triplet_{idx:06d}"
        triplet_folder.mkdir(parents=True, exist_ok=True)

        orig_i, orig_j, orig_k = new_to_orig[i], new_to_orig[j], new_to_orig[k]
        clip_i = index_to_clip[orig_i]
        clip_j = index_to_clip[orig_j]
        clip_k = index_to_clip[orig_k]

        used_clips.update([clip_i, clip_j, clip_k])

        # Source clips come from the buenos_25 visualisation folder
        clips_base = get_path("clips_subset15_game", game=game)
        for clip_name, prefix in [(clip_i, "clip1"), (clip_j, "clip2"), (clip_k, "odd")]:
            src = clips_base / clip_name
            dst = triplet_folder / f"{prefix}_{clip_name}"
            try:
                shutil.copy(src, dst)
            except Exception as e:
                print(f"Error copying {difficulty_name} triplet {idx} ({clip_name}): {e}")


def save_triplet_csv(triplets_array, csv_path):
    pd.DataFrame(
        [{"similar1": int(i), "similar2": int(j), "odd": int(k)} for i, j, k in triplets_array]
    ).to_csv(csv_path, index=False)


def save_clip_index_map(subset_df, csv_path):
    (
        subset_df[["clip_index", "clip_name"]]
        .sort_values("clip_index")
        .reset_index(drop=True)
        .rename(columns={"clip_index": "index"})
        .to_csv(csv_path, index=False)
    )


# =========================================================
# MAIN
# =========================================================
for game in GAMES:
    print(f"\n{'='*60}\nGAME: {game}\n{'='*60}")

    # ── Subset CSV ────────────────────────────────────────
    subset_csv = get_path("subsets_csv", seed=SEED, game=game)
    if not subset_csv.exists():
        print(f"Missing subset CSV: {subset_csv}"); continue

    subset_df       = pd.read_csv(subset_csv)
    selected_idxs   = subset_df["clip_index"].tolist()
    new_to_orig     = {i: orig for i, orig in enumerate(selected_idxs)}
    index_to_clip   = dict(zip(subset_df["clip_index"], subset_df["clip_name"]))
    print(f"Loaded subset with {len(selected_idxs)} clips")

    # ── RDM ───────────────────────────────────────────────
    rdm_path = get_path("rdms_subset15", seed=SEED, game=game) / f"{game}_fc_{METHOD}_RDM.npy"
    if not rdm_path.exists():
        print(f"Missing RDM: {rdm_path}"); continue

    rdm = np.load(rdm_path)
    print(f"RDM shape: {rdm.shape}")

    # ── Build + filter triplets ───────────────────────────
    all_triplets     = remove_symmetric_triplets(build_triplets_from_rdm(rdm))
    print(f"Unique triplets: {len(all_triplets)}")

    scores           = np.zeros(len(all_triplets))
    structure_scores = np.zeros(len(all_triplets))

    for idx, (i, j, k) in enumerate(all_triplets):
        dij = rdm[i, j]
        dik, djk = rdm[i, k], rdm[j, k]
        d1, d2, d3 = sorted([dij, dik, djk])
        structure_scores[idx] = (d3 - d1) / (d3 + 1e-8)
        d_odd = (dik + djk) / 2.0
        scores[idx] = (d_odd - dij) / (d_odd + dij + 1e-8)

    struct_threshold = np.percentile(structure_scores, STRUCTURE_PERCENTILE)
    keep_mask        = structure_scores > struct_threshold
    all_triplets     = all_triplets[keep_mask]
    scores           = scores[keep_mask]
    structure_scores = structure_scores[keep_mask]

    print(f"Structure threshold ({STRUCTURE_PERCENTILE}%): {struct_threshold:.4f}")
    print(f"Remaining structured triplets: {len(all_triplets)}")

    # ── Bucket by difficulty ──────────────────────────────
    n_total     = len(all_triplets)
    sorted_idxs = np.argsort(scores)

    hard_end     = int(n_total * HARD_PERCENT)
    medium_start = int(n_total * MEDIUM_LOW)
    medium_end   = int(n_total * MEDIUM_HIGH)
    easy_start   = int(n_total * (1.0 - EASY_PERCENT))

    hard_triplets   = all_triplets[sorted_idxs[:hard_end]]
    medium_triplets = all_triplets[sorted_idxs[medium_start:medium_end]]
    easy_triplets   = all_triplets[sorted_idxs[easy_start:]]

    print(f"\nHard: {len(hard_triplets)}  Medium: {len(medium_triplets)}  Easy: {len(easy_triplets)}")

    # ── t-STE reconstruction check ────────────────────────
    print("\nRunning t-STE reconstruction...")
    reconstruction_triplets = add_symmetric_triplets(
        np.concatenate([hard_triplets, medium_triplets, easy_triplets])
    )
    best_score = -np.inf
    for _ in tqdm(range(N_REPEAT)):
        X       = cy_tste.tste(reconstruction_triplets, no_dims=DIM, max_iter=MAX_ITER,
                               verbose=False, use_log=True)
        score   = compare_rdms(rdm, embedding_to_rdm(X))
        if score > best_score:
            best_score = score
    print(f"Best score: {best_score:.4f}")

    # ── Output paths ──────────────────────────────────────
    game_out = SAVE_FOLDER / game
    game_out.mkdir(parents=True, exist_ok=True)

    save_clip_index_map(subset_df, game_out / f"{game}_clip_index_map.csv")
    (game_out / "best_score.txt").write_text(f"{best_score:.6f}")

    # ── Save per-difficulty triplets ──────────────────────
    used_clips = set()
    for diff_name, triplets_array in [("easy", easy_triplets), ("medium", medium_triplets), ("hard", hard_triplets)]:
        save_triplet_csv(triplets_array, game_out / f"{diff_name}_triplets.csv")
        save_triplets(triplets_array, diff_name, game_out, new_to_orig, index_to_clip, used_clips, game)

    # ── Copy unique clips ─────────────────────────────────
    print("\nCopying unique clips...")
    clips_out = game_out / "clips"
    clips_out.mkdir(parents=True, exist_ok=True)
    clips_src = get_path("clips_subset15_game", game=game)

    for clip_name in tqdm(used_clips):
        dst = clips_out / clip_name
        if not dst.exists():
            try:
                shutil.copy(clips_src / clip_name, dst)
            except Exception as e:
                print(f"Error copying {clip_name}: {e}")

print("\nDONE.")
