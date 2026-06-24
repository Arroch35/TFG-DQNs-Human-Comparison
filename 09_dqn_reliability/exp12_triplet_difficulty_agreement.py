"""
exp12_triplet_difficulty_agreement.py
For each game, build the master triplet table from seed_42's RDM,
then compute per-triplet agreement across the other 4 seeds.
"""
import numpy as np
import pandas as pd

from src.config import GAMES, SEEDS, REFERENCE_SEED, TSTE, REPR, get_path, ensure
from src.utils import build_triplets_from_rdm, remove_symmetric_triplets

# =========================================================
# CONFIG
# NOTE: seed_42 must be processed first (defines the master table).
# Config's SEEDS list starts with seed_0; we reorder locally.
# =========================================================
SEED_42    = REFERENCE_SEED                    # "seed_42"
ALL_SEEDS  = [SEED_42] + [s for s in SEEDS if s != SEED_42]

RDM_METHOD           = REPR["rdm_method"]      # "correlation"
EASY_PERCENT         = TSTE["easy_percent"]    # 0.2
HARD_PERCENT         = TSTE["hard_percent"]    # 0.2
MEDIUM_LOW           = TSTE["medium_low"]      # 0.4
MEDIUM_HIGH          = TSTE["medium_high"]     # 0.6
STRUCTURE_PERCENTILE = TSTE["structure_pctile"] # 20

OUTPUT_DIR = ensure("triplets_scores_dir") 

# =========================================================
# HELPERS
# =========================================================
def compute_scores(triplets, rdm):
    n = len(triplets)
    diff_scores = struct_scores = d_sim = d_odd = (np.zeros(n),) * 4
    diff_scores, struct_scores, d_sim, d_odd = [np.zeros(n) for _ in range(4)]
    for idx, (i, j, k) in enumerate(triplets):
        dij = rdm[i, j]
        dik, djk = rdm[i, k], rdm[j, k]
        d_odd_val = (dik + djk) / 2.0
        diff_scores[idx]  = (d_odd_val - dij) / (d_odd_val + dij + 1e-8)
        d1, d2, d3        = sorted([dij, dik, djk])
        struct_scores[idx] = (d3 - d1) / (d3 + 1e-8)
        d_sim[idx]         = dij
        d_odd[idx]         = d_odd_val
    return diff_scores, struct_scores, d_sim, d_odd


def get_seed_answer(s1, s2, odd, rdm, orig_to_local):
    li, lj, lk = orig_to_local[s1], orig_to_local[s2], orig_to_local[odd]
    mean_i = (rdm[li, lj] + rdm[li, lk]) / 2.0
    mean_j = (rdm[li, lj] + rdm[lj, lk]) / 2.0
    mean_k = (rdm[li, lk] + rdm[lj, lk]) / 2.0
    pick_local = [li, lj, lk][np.argmax([mean_i, mean_j, mean_k])]
    local_to_orig = {v: k for k, v in orig_to_local.items()}
    return local_to_orig[pick_local]


# =========================================================
# MAIN
# =========================================================
summary_rows = []

for game in GAMES:
    print(f"\n{'='*60}\nGAME: {game}\n{'='*60}")

    clip_map_path = get_path("maps_subset15_game", game=game)
    if not clip_map_path.exists():
        print(f"  Missing clip map: {clip_map_path}, skipping."); continue

    clip_map_df   = pd.read_csv(clip_map_path)
    new_to_orig   = dict(enumerate(clip_map_df["clip_index"].tolist()))
    orig_to_new   = {orig: local for local, orig in new_to_orig.items()}
    index_to_clip = dict(zip(clip_map_df["clip_index"], clip_map_df["clip_name"]))

    master_df = None

    for seed in ALL_SEEDS:
        print(f"\n  --- {seed} ---")

        rdm_path = get_path("rdms_subset15", seed=seed, game=game) / f"{game}_fc_{RDM_METHOD}_RDM.npy"
        if not rdm_path.exists():
            print(f"    Missing RDM: {rdm_path}, skipping."); continue

        rdm = np.load(rdm_path)

        if seed == SEED_42:
            triplets = remove_symmetric_triplets(build_triplets_from_rdm(rdm))
            print(f"    Unique triplets: {len(triplets)}")

            diff_scores, struct_scores, d_sim_arr, d_odd_arr = compute_scores(triplets, rdm)
            struct_threshold = np.percentile(struct_scores, STRUCTURE_PERCENTILE)
            above_filter     = struct_scores > struct_threshold
            filtered_idx     = np.where(above_filter)[0]
            sorted_rel       = np.argsort(diff_scores[filtered_idx])
            n_filtered       = len(filtered_idx)

            hard_end     = int(n_filtered * HARD_PERCENT)
            medium_start = int(n_filtered * MEDIUM_LOW)
            medium_end   = int(n_filtered * MEDIUM_HIGH)
            easy_start   = int(n_filtered * (1.0 - EASY_PERCENT))

            difficulty_labels = np.full(len(triplets), "outside_buckets", dtype=object)
            difficulty_labels[filtered_idx[sorted_rel[:hard_end]]]               = "hard_triplets"
            difficulty_labels[filtered_idx[sorted_rel[medium_start:medium_end]]] = "medium_triplets"
            difficulty_labels[filtered_idx[sorted_rel[easy_start:]]]             = "easy_triplets"

            rows = []
            for pos, (li, lj, lk) in enumerate(triplets):
                oi, oj, ok = new_to_orig[li], new_to_orig[lj], new_to_orig[lk]
                rows.append({
                    "game": game, "difficulty": difficulty_labels[pos],
                    "similar_1_idx": oi, "similar_2_idx": oj, "odd_idx": ok,
                    "similar_1_clip": index_to_clip.get(oi, ""),
                    "similar_2_clip": index_to_clip.get(oj, ""),
                    "odd_clip":       index_to_clip.get(ok, ""),
                    "difficulty_score": diff_scores[pos],
                    "structure_score":  struct_scores[pos],
                    "d_similar":        d_sim_arr[pos],
                    "d_odd_avg":        d_odd_arr[pos],
                    "above_structure_filter": struct_scores[pos] > struct_threshold,
                    "seed_42_answer":   ok,
                })

            master_df = pd.DataFrame(rows)
            print(f"    Master table: {len(master_df)} triplets")
            for label in ["hard_triplets", "medium_triplets", "easy_triplets", "outside_buckets"]:
                print(f"      {label}: {(difficulty_labels == label).sum()}")

        else:
            if master_df is None:
                print("    master_df not built (seed_42 missing), skipping."); continue

            answers = [
                get_seed_answer(int(r["similar_1_idx"]), int(r["similar_2_idx"]),
                                int(r["odd_idx"]), rdm, orig_to_new)
                for _, r in master_df.iterrows()
            ]
            col = f"{seed}_answer"
            master_df[col] = answers
            agrees = (master_df[col] == master_df["odd_idx"]).mean()
            print(f"    Agreement with seed_42: {agrees:.3f}")

    # ── Save + summary ────────────────────────────────────
    if master_df is not None:
        answer_cols = [c for c in master_df.columns if c.endswith("_answer") and c != "seed_42_answer"]
        if answer_cols:
            master_df["n_seeds_agree"]    = sum((master_df[c] == master_df["odd_idx"]).astype(int) for c in answer_cols)
            master_df["frac_seeds_agree"] = master_df["n_seeds_agree"] / len(answer_cols)

        out_csv = get_path("triplets_scores_csv", game=game)
        master_df.to_csv(out_csv, index=False)
        print(f"\n  Saved {len(master_df)} triplets → {out_csv}")

        if answer_cols:
            print("\n  Agreement with seed_42 per difficulty:")
            for diff in ["hard_triplets", "medium_triplets", "easy_triplets", "outside_buckets"]:
                sub = master_df[master_df["difficulty"] == diff]
                if sub.empty: continue
                mean_a, std_a = sub["frac_seeds_agree"].mean(), sub["frac_seeds_agree"].std()
                s_min,  s_max = sub["difficulty_score"].min(),  sub["difficulty_score"].max()
                print(f"    {diff}: {mean_a:.3f} ± {std_a:.3f}  "
                      f"(score [{s_min:.3f}, {s_max:.3f}], n={len(sub)})")
                summary_rows.append({"game": game, "difficulty": diff, "n_triplets": len(sub),
                                     "mean_agree": round(mean_a, 4), "std_agree": round(std_a, 4),
                                     "score_min": round(s_min, 4), "score_max": round(s_max, 4)})

# =========================================================
# SAVE SUMMARY
# =========================================================
summary_csv = OUTPUT_DIR / "triplet_agreement_summary.csv"
pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
print(f"\nSummary saved → {summary_csv}")
print("\nDONE.")
