import numpy as np
import os
import pandas as pd
from itertools import combinations

# =========================================================
# CONFIG
# =========================================================
GAMES = ["pacman", "pong", "spaceinvaders"]

SEEDS = ["seed_42", "seed_0", "seed_1", "seed_2", "seed_3"]  # seed_42 must be first

BASE_RDM_FOLDER = "../data/test_16_rdms/selected_subset_15"

# Clip index maps saved by 9_4 — one per game, same for all seeds
CLIP_MAP_FOLDER = f"../data/maps/selected_15/"

OUTPUT_DIR = "../data/triplets_results/triplet_scores"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Difficulty percentile cuts — identical to 9_4
EASY_PERCENT         = 0.2
HARD_PERCENT         = 0.2
MEDIUM_LOW           = 0.4
MEDIUM_HIGH          = 0.6
STRUCTURE_PERCENTILE = 20

# =========================================================
# HELPERS
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
        dij = rdm[i, j]
        dik, djk = rdm[i, k], rdm[j, k]
        d_odd = (dik + djk) / 2.0
        difficulty_scores[idx] = (d_odd - dij) / (d_odd + dij + 1e-8)
        d1, d2, d3 = sorted([dij, dik, djk])
        structure_scores[idx]  = (d3 - d1) / (d3 + 1e-8)
        d_similar_arr[idx] = dij
        d_odd_avg_arr[idx] = d_odd
    return difficulty_scores, structure_scores, d_similar_arr, d_odd_avg_arr


def get_seed_answer(s1, s2, odd, rdm, orig_to_local):
    """Given a triplet in original indices, return which clip the seed picks as odd."""
    li, lj, lk = orig_to_local[s1], orig_to_local[s2], orig_to_local[odd]
    dij = rdm[li, lj]
    dik = rdm[li, lk]
    djk = rdm[lj, lk]
    # The odd clip is the one with the largest mean distance to the other two
    mean_i = (dij + dik) / 2.0   # mean dist of clip i to the other two
    mean_j = (dij + djk) / 2.0
    mean_k = (dik + djk) / 2.0
    pick_local = [li, lj, lk][np.argmax([mean_i, mean_j, mean_k])]
    # Convert back to original index
    local_to_orig = {v: k for k, v in orig_to_local.items()}
    return local_to_orig[pick_local]


# =========================================================
# MAIN LOOP
# =========================================================
for game in GAMES:

    print("\n" + "=" * 60)
    print(f"GAME: {game}")
    print("=" * 60)

    # --- Load clip index map saved by 9_4 (same for all seeds) ---
    clip_map_csv = os.path.join(CLIP_MAP_FOLDER, f"{game}_clip_map.csv")
    if not os.path.exists(clip_map_csv):
        print(f"  Missing clip index map: {clip_map_csv}, skipping.")
        continue
    clip_map_df   = pd.read_csv(clip_map_csv)
    # columns: "index" (original clip index), "clip_name"
    new_to_orig   = dict(enumerate(clip_map_df["clip_index"].tolist()))
    orig_to_new   = {orig: local for local, orig in new_to_orig.items()}
    index_to_clip = dict(zip(clip_map_df["clip_index"], clip_map_df["clip_name"]))

    master_df = None  # will be built on seed_42

    for seed in SEEDS:

        print(f"\n  --- {seed} ---")

        rdm_path = os.path.join(BASE_RDM_FOLDER, seed, game, f"{game}_fc_correlation_RDM.npy")
        if not os.path.exists(rdm_path):
            print(f"    Missing RDM: {rdm_path}, skipping.")
            continue
        rdm = np.load(rdm_path)

        # -----------------------------------------------
        # SEED_42: build master CSV with all 455 triplets
        # -----------------------------------------------
        if seed == "seed_42":

            triplets = build_triplets_from_rdm(rdm)
            triplets = remove_symmetric_triplets(triplets)
            print(f"    Total unique triplets: {len(triplets)}")

            diff_scores, struct_scores, d_sim_arr, d_odd_arr = compute_scores(triplets, rdm)

            # Structure filter — discard bottom 20% BEFORE difficulty binning
            struct_threshold = np.percentile(struct_scores, STRUCTURE_PERCENTILE)
            above_filter = struct_scores > struct_threshold

            # Work on filtered indices only
            filtered_idx = np.where(above_filter)[0]
            filtered_diff = diff_scores[filtered_idx]
            sorted_rel = np.argsort(filtered_diff)   # relative order within filtered set

            n_filtered   = len(filtered_idx)
            hard_end     = int(n_filtered * HARD_PERCENT)
            medium_start = int(n_filtered * MEDIUM_LOW)
            medium_end   = int(n_filtered * MEDIUM_HIGH)
            easy_start   = int(n_filtered * (1.0 - EASY_PERCENT))

            difficulty_labels = np.full(len(triplets), "outside_buckets", dtype=object)
            # low-structure triplets stay "outside_buckets"; filtered ones get binned
            difficulty_labels[filtered_idx[sorted_rel[:hard_end]]]               = "hard_triplets"
            difficulty_labels[filtered_idx[sorted_rel[medium_start:medium_end]]] = "medium_triplets"
            difficulty_labels[filtered_idx[sorted_rel[easy_start:]]]             = "easy_triplets"
            # triplets that failed the structure filter get their own label
            difficulty_labels[~above_filter] = "outside_buckets"

            rows = []
            for pos, (li, lj, lk) in enumerate(triplets):
                oi, oj, ok = new_to_orig[li], new_to_orig[lj], new_to_orig[lk]
                rows.append({
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
                    "above_structure_filter": struct_scores[pos] > struct_threshold,
                    "seed_42_answer":   ok,  # by definition always correct
                })

            master_df = pd.DataFrame(rows)
            print(f"    Master CSV built: {len(master_df)} triplets")
            for label in ["hard_triplets", "medium_triplets", "easy_triplets", "outside_buckets", "lower_bounded"]:
                print(f"      {label}: {(difficulty_labels == label).sum()}")

        # -----------------------------------------------
        # OTHER SEEDS: add their answer column to master
        # -----------------------------------------------
        else:
            if master_df is None:
                print("    master_df not built yet (seed_42 missing), skipping.")
                continue

            answers = []
            for _, row in master_df.iterrows():
                ans = get_seed_answer(
                    int(row["similar_1_idx"]),
                    int(row["similar_2_idx"]),
                    int(row["odd_idx"]),
                    rdm,
                    orig_to_new
                )
                answers.append(ans)

            col = f"{seed}_answer"
            master_df[col] = answers
            agrees = (master_df[col] == master_df["odd_idx"]).mean()
            print(f"    Agreement with seed_42: {agrees:.3f}")

    # -----------------------------------------------
    # SAVE
    # -----------------------------------------------
    if master_df is not None:
        # Add a summary column: how many seeds agree with seed_42
        answer_cols = [c for c in master_df.columns if c.endswith("_answer") and c != "seed_42_answer"]
        if answer_cols:
            master_df["n_seeds_agree"] = sum(
                (master_df[col] == master_df["odd_idx"]).astype(int)
                for col in answer_cols
            )
            master_df["frac_seeds_agree"] = master_df["n_seeds_agree"] / len(answer_cols)

        out_csv = os.path.join(OUTPUT_DIR, f"triplet_scores_{game}.csv")
        master_df.to_csv(out_csv, index=False)
        print(f"\n  Saved {len(master_df)} triplets → {out_csv}")

        # Quick summary per difficulty
        print("\n  Agreement with seed_42 per difficulty:")
        if answer_cols:
            for diff in ["hard_triplets", "medium_triplets", "easy_triplets"]:
                sub = master_df[master_df["difficulty"] == diff]
                if sub.empty:
                    continue
                mean_agree = sub["frac_seeds_agree"].mean()
                print(f"    {diff}: {mean_agree:.3f} (n={len(sub)})")

print("\nDONE.")
