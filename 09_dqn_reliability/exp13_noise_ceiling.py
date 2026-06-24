"""
exp13_noise_ceiling.py
For each non-42 seed, use the sparse human triplets (majority-voted with
that seed's answers) to reconstruct an RDM via t-STE, then correlate
against seed_42's RDM. The mean across seeds is the noise ceiling.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import cy_tste

from src.config import GAMES, SEEDS, REFERENCE_SEED, TSTE, REPR, get_path, ensure
from src.utils import embedding_to_rdm, add_symmetric_triplets

# =========================================================
# CONFIG
# NOTE: seed_42 defines the reference RDM; must be processed first.
# =========================================================
SEED_42   = REFERENCE_SEED
ALL_SEEDS = [SEED_42] + [s for s in SEEDS if s != SEED_42]

RDM_METHOD = REPR["rdm_method"]    # "correlation"
DIM        = TSTE["dim"]           # 2
N_REPEATS  = TSTE["n_repeats"]     # 100
MAX_ITER   = TSTE["max_iter"]      # 1000

OUTPUT_DIR = ensure("results_noise_ceiling")   # data/triplets_results/noise_ceiling

# =========================================================
# HELPERS
# =========================================================
def rdm_correlation(rdm1, rdm2):
    idx = np.triu_indices_from(rdm1, k=1)
    return spearmanr(rdm1[idx], rdm2[idx]).correlation


def run_tste(triplets_local, rdm_42_sub):
    """Run t-STE N_REPEATS times; return the RDM from the best-correlating run."""
    sym = add_symmetric_triplets(triplets_local)
    best_X, best_corr = None, -np.inf
    for _ in range(N_REPEATS):
        X    = cy_tste.tste(sym, no_dims=DIM, max_iter=MAX_ITER, verbose=False, use_log=True)
        corr = rdm_correlation(rdm_42_sub, embedding_to_rdm(X))
        if corr > best_corr:
            best_corr, best_X = corr, X
    return embedding_to_rdm(best_X)


# =========================================================
# MAIN
# =========================================================
summary_rows = []

for game in GAMES:
    print(f"\n{'='*60}\nGAME: {game}\n{'='*60}")

    # ── Clip map ──────────────────────────────────────────
    clip_map_path = get_path("maps_selected15_game", game=game)
    if not clip_map_path.exists():
        print(f"  Missing clip map: {clip_map_path}, skipping."); continue

    clip_map_df   = pd.read_csv(clip_map_path)
    orig_to_local = {orig: local for local, orig in enumerate(clip_map_df["clip_index"].tolist())}
    n_clips       = len(clip_map_df)

    # ── seed_42 RDM (subselected to the 15 experiment clips) ──
    rdm42_path = get_path("rdms_selected15", seed=SEED_42, game=game) / f"{game}_fc_{RDM_METHOD}_RDM.npy"
    if not rdm42_path.exists():
        print(f"  Missing seed_42 RDM: {rdm42_path}, skipping."); continue

    rdm_42_full = np.load(rdm42_path)
    local_idx   = list(range(n_clips))
    rdm_42_sub  = rdm_42_full[np.ix_(local_idx, local_idx)]

    # ── Master triplet CSV (output of exp12) ──────────────
    master_csv = get_path("triplets_scores_csv", game=game)
    if not master_csv.exists():
        print(f"  Missing master CSV: {master_csv}, skipping."); continue
    master_df = pd.read_csv(master_csv)

    # ── Sparse human responses ────────────────────────────
    sparse_csv = get_path("experiment_cleaned") / f"{game}_triplets_indexed_with_difficulty.csv"
    if not sparse_csv.exists():
        print(f"  Missing sparse responses: {sparse_csv}, skipping."); continue
    sparse_df = pd.read_csv(sparse_csv)

    # ── Unique sparse triplets + majority vote ────────────
    sparse_df["triplet_fs"] = sparse_df.apply(
        lambda r: frozenset({int(r["similar_clip_1_idx"]), int(r["similar_clip_2_idx"]), int(r["odd_clip_idx"])}),
        axis=1,
    )
    unique_fs      = sparse_df["triplet_fs"].unique()
    majority_votes = {
        fs: grp["odd_clip_idx"].value_counts().index[0]
        for fs, grp in sparse_df.groupby("triplet_fs")
    }
    print(f"  Unique sparse triplets: {len(unique_fs)}")

    master_df["triplet_fs"] = master_df.apply(
        lambda r: frozenset({int(r["similar_1_idx"]), int(r["similar_2_idx"]), int(r["odd_idx"])}),
        axis=1,
    )
    sparse_master = master_df[master_df["triplet_fs"].isin(unique_fs)].copy()
    print(f"  Sparse triplets in master: {len(sparse_master)}")

    seed_answer_cols = [c for c in master_df.columns if c.endswith("_answer") and c != "seed_42_answer"]
    other_seeds      = [c.replace("_answer", "") for c in seed_answer_cols]

    if not other_seeds:
        print("  No other seed answer columns found — skipping."); continue

    # ── Per-seed t-STE reconstruction ────────────────────
    seed_correlations = []

    for seed, ans_col in zip(other_seeds, seed_answer_cols):
        print(f"\n  Processing {seed}...")

        triplets_local, skipped = [], 0
        for _, row in sparse_master.iterrows():
            s1, s2, odd    = int(row["similar_1_idx"]), int(row["similar_2_idx"]), int(row["odd_idx"])
            seed_odd       = int(row[ans_col])
            similar_clips  = [c for c in [s1, s2, odd] if c != seed_odd]
            if len(similar_clips) != 2:
                skipped += 1; continue
            l_s1  = orig_to_local.get(similar_clips[0])
            l_s2  = orig_to_local.get(similar_clips[1])
            l_odd = orig_to_local.get(seed_odd)
            if None in (l_s1, l_s2, l_odd):
                skipped += 1; continue
            triplets_local.append((l_s1, l_s2, l_odd))

        if skipped:
            print(f"    Skipped {skipped} triplets (index not found)")
        if len(triplets_local) < 10:
            print(f"    Too few triplets ({len(triplets_local)}), skipping."); continue

        triplets_arr = np.array(triplets_local, dtype=np.int32)
        print(f"    Running t-STE on {len(triplets_arr)} triplets...")

        tste_rdm = run_tste(triplets_arr, rdm_42_sub)
        corr     = rdm_correlation(rdm_42_sub, tste_rdm)
        print(f"    Correlation with seed_42: {corr:.4f}")

        seed_correlations.append(corr)
        summary_rows.append({"game": game, "seed": seed, "n_triplets": len(triplets_arr), "correlation": corr})

    if seed_correlations:
        nc_mean = np.mean(seed_correlations)
        nc_std  = np.std(seed_correlations)
        print(f"\n  === Noise ceiling ({game}): {nc_mean:.4f} ± {nc_std:.4f} "
              f"(mean ± std over {len(seed_correlations)} seeds) ===")
        summary_rows.append({"game": game, "seed": "NOISE_CEILING",
                              "n_triplets": len(sparse_master), "correlation": nc_mean, "std": nc_std})

# =========================================================
# SAVE
# =========================================================
summary_df = pd.DataFrame(summary_rows).round(4)
out_csv    = OUTPUT_DIR / "dqn_noise_ceiling.csv"
summary_df.to_csv(out_csv, index=False)
print(f"\nSaved summary → {out_csv}")

nc = summary_df[summary_df["seed"] == "NOISE_CEILING"]
print("\n--- Final noise ceilings ---")
print(nc[["game", "correlation"]].to_string(index=False))
print("\nDONE.")
