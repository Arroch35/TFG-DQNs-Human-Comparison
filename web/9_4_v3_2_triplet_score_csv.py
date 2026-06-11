import numpy as np
import os
import pandas as pd
from itertools import combinations

# =========================================================
# CONFIG  — keep identical to 9_4_v3
# =========================================================
GAMES = ["pacman", "pong", "spaceinvaders"]

SEED = "seed_42"

BASE_RDM_FOLDER = f"../data/test_16_rdms/selected_subset_15/{SEED}"
SA_FOLDER       = f"../data/subset_selection/{SEED}"

OUTPUT_DIR = "../data/triplets_results/triplet_scores"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Difficulty percentile cuts — identical to 9_4
EASY_PERCENT         = 0.2
HARD_PERCENT         = 0.2
MEDIUM_LOW           = 0.4
MEDIUM_HIGH          = 0.6
STRUCTURE_PERCENTILE = 20

# =========================================================
# HELPERS  — identical logic to 9_4
# =========================================================
def build_triplets_from_rdm(rdm):
    triplets = []
    for i, j, k in combinations(range(len(rdm)), 3):
        dij, dik, djk = rdm[i,j], rdm[i,k], rdm[j,k]
        if dij <= dik and dij <= djk:
            triplets.append((i, j, k)); triplets.append((j, i, k))
        elif dik <= dij and dik <= djk:
            triplets.append((i, k, j)); triplets.append((k, i, j))
        else:
            triplets.append((j, k, i)); triplets.append((k, j, i))
    return np.array(triplets, dtype=np.int32)


def remove_symmetric_triplets(triplets):
    seen, unique = set(), []
    for i, j, k in triplets:
        key = (min(i,j), max(i,j), k)
        if key not in seen:
            seen.add(key)
            unique.append((i, j, k))
    return np.array(unique, dtype=np.int32)


def compute_scores(triplets, rdm):
    n = len(triplets)
    difficulty_scores = np.zeros(n)
    structure_scores  = np.zeros(n)
    d_similar_arr     = np.zeros(n)
    d_odd_avg_arr     = np.zeros(n)

    for idx, (i, j, k) in enumerate(triplets):
        dij = rdm[i, j]          # similar-pair distance
        dik = rdm[i, k]
        djk = rdm[j, k]
        d_odd = (dik + djk) / 2.0

        # difficulty: ~0 = impossible (odd≈similar), ~1 = trivial (odd≫similar)
        difficulty_scores[idx] = (d_odd - dij) / (d_odd + dij + 1e-8)

        # structure: how spread apart the three distances are
        d1, d2, d3 = sorted([dij, dik, djk])
        structure_scores[idx]  = (d3 - d1) / (d3 + 1e-8)

        d_similar_arr[idx] = dij
        d_odd_avg_arr[idx] = d_odd

    return difficulty_scores, structure_scores, d_similar_arr, d_odd_avg_arr


# =========================================================
# MAIN LOOP
# =========================================================
all_rows = []

for game in GAMES:

    print("\n" + "=" * 60)
    print(f"GAME: {game}")
    print("=" * 60)

    # --- Load subset CSV ---
    subset_csv = os.path.join(SA_FOLDER, f"{game}_best_subset_indices.csv")
    if not os.path.exists(subset_csv):
        print(f"  Missing subset CSV: {subset_csv}, skipping.")
        continue
    subset_df      = pd.read_csv(subset_csv)
    selected_idxs  = subset_df["clip_index"].tolist()
    new_to_orig    = {i: orig for i, orig in enumerate(selected_idxs)}
    index_to_clip  = dict(zip(subset_df["clip_index"], subset_df["clip_name"]))

    # --- Load RDM ---
    rdm_path = os.path.join(BASE_RDM_FOLDER, game, f"{game}_fc_correlation_RDM.npy")
    if not os.path.exists(rdm_path):
        print(f"  Missing RDM: {rdm_path}, skipping.")
        continue
    rdm = np.load(rdm_path)
    print(f"  RDM shape: {rdm.shape}")

    # --- Build triplets + scores (exact same pipeline as 9_4) ---
    triplets = build_triplets_from_rdm(rdm)
    triplets  = remove_symmetric_triplets(triplets)

    diff_scores, struct_scores, d_sim_arr, d_odd_arr = compute_scores(triplets, rdm)

    # Apply structure filter
    struct_threshold = np.percentile(struct_scores, STRUCTURE_PERCENTILE)
    keep          = struct_scores > struct_threshold
    triplets      = triplets[keep]
    diff_scores   = diff_scores[keep]
    struct_scores = struct_scores[keep]
    d_sim_arr     = d_sim_arr[keep]
    d_odd_arr     = d_odd_arr[keep]

    print(f"  Triplets after structure filter: {len(triplets)}")

    # Assign difficulty buckets (same percentile cuts as 9_4)
    n_total      = len(triplets)
    sorted_idx   = np.argsort(diff_scores)

    hard_end     = int(n_total * HARD_PERCENT)
    medium_start = int(n_total * MEDIUM_LOW)
    medium_end   = int(n_total * MEDIUM_HIGH)
    easy_start   = int(n_total * (1.0 - EASY_PERCENT))

    difficulty_labels = np.full(n_total, "outside_buckets", dtype=object)
    difficulty_labels[sorted_idx[:hard_end]]               = "hard_triplets"
    difficulty_labels[sorted_idx[medium_start:medium_end]] = "medium_triplets"
    difficulty_labels[sorted_idx[easy_start:]]             = "easy_triplets"

    # --- Build rows ---
    for pos, (li, lj, lk) in enumerate(triplets):
        oi = new_to_orig[li]   # original clip indices
        oj = new_to_orig[lj]
        ok = new_to_orig[lk]   # ok is the odd clip

        all_rows.append({
            "game":              game,
            "difficulty":        difficulty_labels[pos],
            # Clip indices in original (full-set) space — same as 5_x scripts use
            "similar_1_idx":     oi,
            "similar_2_idx":     oj,
            "odd_idx":           ok,
            # Human-readable clip names
            "similar_1_clip":    index_to_clip.get(oi, ""),
            "similar_2_clip":    index_to_clip.get(oj, ""),
            "odd_clip":          index_to_clip.get(ok, ""),
            # Scores
            "difficulty_score":  diff_scores[pos],   # (d_odd - d_sim)/(d_odd + d_sim)
            "structure_score":   struct_scores[pos],  # (d_max - d_min)/d_max
            "d_similar":         d_sim_arr[pos],      # raw RDM distance of the similar pair
            "d_odd_avg":         d_odd_arr[pos],      # mean RDM distance odd→each similar clip
        })

    counts = {
        "hard_triplets":    (difficulty_labels == "hard_triplets").sum(),
        "medium_triplets":  (difficulty_labels == "medium_triplets").sum(),
        "easy_triplets":    (difficulty_labels == "easy_triplets").sum(),
        "outside_buckets":  (difficulty_labels == "outside_buckets").sum(),
    }
    for label, n in counts.items():
        print(f"  {label}: {n}")

# =========================================================
# SAVE
# =========================================================
out_df = pd.DataFrame(all_rows)

if out_df.empty:
    print("\nNo rows produced — check RDM and subset CSV paths.")
else:
    # One CSV with everything
    out_csv = os.path.join(OUTPUT_DIR, "triplet_scores_all_games.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(out_df)} rows → {out_csv}")

    # One CSV per game
    for game in out_df["game"].unique():
        g_df  = out_df[out_df["game"] == game]
        g_csv = os.path.join(OUTPUT_DIR, f"triplet_scores_{game}.csv")
        g_df.to_csv(g_csv, index=False)
        print(f"  {game}: {len(g_df)} triplets → {g_csv}")

    # Summary
    print("\n--- difficulty_score range per game × difficulty ---")
    summary = (
        out_df[out_df["difficulty"] != "outside_buckets"]
        .groupby(["game", "difficulty"])["difficulty_score"]
        .agg(count="count", min="min", mean="mean", max="max")
        .round(4)
    )
    print(summary.to_string())

print("\nDONE.")
