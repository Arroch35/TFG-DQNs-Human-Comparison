"""
2_export_triplet_scores_csv.py
Re-score all triplets from the SA-selected subset and export one CSV per game
plus an aggregated CSV, with difficulty buckets matching script 1.
"""
import numpy as np
import pandas as pd

from src.config import (
    GAMES, REFERENCE_SEED, REPR, TSTE,
    get_path, ensure,
)
from src.utils import build_triplets_from_rdm, remove_symmetric_triplets

# =========================================================
# CONFIG
# =========================================================
SEED   = REFERENCE_SEED             # "seed_42"
METHOD = REPR["rdm_method"]         # "correlation"

EASY_PERCENT         = TSTE["easy_percent"]       # 0.2
HARD_PERCENT         = TSTE["hard_percent"]       # 0.2
MEDIUM_LOW           = TSTE["medium_low"]         # 0.4
MEDIUM_HIGH          = TSTE["medium_high"]        # 0.6
STRUCTURE_PERCENTILE = TSTE["structure_pctile"]   # 20

OUTPUT_DIR = ensure("triplets_scores_dir")

# =========================================================
# SCORING
# =========================================================
def compute_scores(triplets, rdm):
    n = len(triplets)
    difficulty_scores = np.zeros(n)
    structure_scores  = np.zeros(n)
    d_similar_arr     = np.zeros(n)
    d_odd_avg_arr     = np.zeros(n)

    for idx, (i, j, k) in enumerate(triplets):
        dij   = rdm[i, j]
        dik, djk = rdm[i, k], rdm[j, k]
        d_odd = (dik + djk) / 2.0

        difficulty_scores[idx] = (d_odd - dij) / (d_odd + dij + 1e-8)
        d1, d2, d3 = sorted([dij, dik, djk])
        structure_scores[idx]  = (d3 - d1) / (d3 + 1e-8)
        d_similar_arr[idx]     = dij
        d_odd_avg_arr[idx]     = d_odd

    return difficulty_scores, structure_scores, d_similar_arr, d_odd_avg_arr

# =========================================================
# MAIN LOOP
# =========================================================
all_rows = []

for game in GAMES:
    print(f"\n{'='*60}\nGAME: {game}\n{'='*60}")

    # ── Subset CSV ────────────────────────────────────────
    subset_csv = get_path("subsets_csv", seed=SEED, game=game)
    if not subset_csv.exists():
        print(f"  Missing subset CSV: {subset_csv}, skipping."); continue

    subset_df      = pd.read_csv(subset_csv)
    selected_idxs  = subset_df["clip_index"].tolist()
    new_to_orig    = {i: orig for i, orig in enumerate(selected_idxs)}
    index_to_clip  = dict(zip(subset_df["clip_index"], subset_df["clip_name"]))

    # ── RDM ───────────────────────────────────────────────
    rdm_path = get_path("rdms_subset15", seed=SEED, game=game) / f"{game}_fc_{METHOD}_RDM.npy"
    if not rdm_path.exists():
        print(f"  Missing RDM: {rdm_path}, skipping."); continue

    rdm = np.load(rdm_path)
    print(f"  RDM shape: {rdm.shape}")

    # ── Build + score triplets ────────────────────────────
    triplets = remove_symmetric_triplets(build_triplets_from_rdm(rdm))

    diff_scores, struct_scores, d_sim_arr, d_odd_arr = compute_scores(triplets, rdm)

    # Structure filter
    struct_threshold = np.percentile(struct_scores, STRUCTURE_PERCENTILE)
    keep          = struct_scores > struct_threshold
    triplets      = triplets[keep]
    diff_scores   = diff_scores[keep]
    struct_scores = struct_scores[keep]
    d_sim_arr     = d_sim_arr[keep]
    d_odd_arr     = d_odd_arr[keep]

    print(f"  Triplets after structure filter: {len(triplets)}")

    # Difficulty buckets
    n_total       = len(triplets)
    sorted_idx    = np.argsort(diff_scores)

    hard_end      = int(n_total * HARD_PERCENT)
    medium_start  = int(n_total * MEDIUM_LOW)
    medium_end    = int(n_total * MEDIUM_HIGH)
    easy_start    = int(n_total * (1.0 - EASY_PERCENT))

    difficulty_labels = np.full(n_total, "outside_buckets", dtype=object)
    difficulty_labels[sorted_idx[:hard_end]]               = "hard_triplets"
    difficulty_labels[sorted_idx[medium_start:medium_end]] = "medium_triplets"
    difficulty_labels[sorted_idx[easy_start:]]             = "easy_triplets"

    # Build rows
    for pos, (li, lj, lk) in enumerate(triplets):
        oi, oj, ok = new_to_orig[li], new_to_orig[lj], new_to_orig[lk]
        all_rows.append({
            "game":             game,
            "difficulty":       difficulty_labels[pos],
            "similar_1_idx":    oi,
            "similar_2_idx":    oj,
            "odd_idx":          ok,
            "similar_1_clip":   index_to_clip.get(oi, ""),
            "similar_2_clip":   index_to_clip.get(oj, ""),
            "odd_clip":         index_to_clip.get(ok, ""),
            "difficulty_score": diff_scores[pos],
            "structure_score":  struct_scores[pos],
            "d_similar":        d_sim_arr[pos],
            "d_odd_avg":        d_odd_arr[pos],
        })

    for label in ["hard_triplets", "medium_triplets", "easy_triplets", "outside_buckets"]:
        print(f"  {label}: {(difficulty_labels == label).sum()}")

# =========================================================
# SAVE
# =========================================================
out_df = pd.DataFrame(all_rows)

if out_df.empty:
    print("\nNo rows produced — check RDM and subset CSV paths.")
else:
    # All-games CSV
    all_csv = OUTPUT_DIR / "triplet_scores_all_games.csv"
    out_df.to_csv(all_csv, index=False)
    print(f"\nSaved {len(out_df)} rows → {all_csv}")

    # Per-game CSVs — path already in config as "triplet_score_csv"
    for game in out_df["game"].unique():
        g_csv = get_path("triplets_scores_csv", game=game)
        out_df[out_df["game"] == game].to_csv(g_csv, index=False)
        print(f"  {game}: {len(out_df[out_df['game'] == game])} triplets → {g_csv}")

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
