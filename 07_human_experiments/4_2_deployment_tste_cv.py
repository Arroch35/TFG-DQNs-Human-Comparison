"""
4_2_deployment_tste_cv.py
t-STE Cross-Validation — Deployment Version.
15 clips (indices 0–14), ~41 participants, ~6 triplets/participant/game.
Runs LOPO-CV (primary) + triplet-split CV (secondary).
"""
import os
import numpy as np
import pandas as pd
import cy_tste
from scipy.spatial.distance import cdist

from src.config import GAMES, TSTE, get_path, ensure

# =========================================================
# CONFIG
# =========================================================
DIMS_TO_TRY  = [2, 3, 5, 8, 10, 15, 20, 50]
N_CLIPS      = TSTE["n_clips"]       # 15

PARTICIPANT_COL = "participant_id"
TRIPLET_COLS    = ["reference", "near", "far"]

MAX_ITER = TSTE["max_iter"]          # 1000
USE_LOG  = True

MIN_TRIPLETS_PER_PARTICIPANT = 6

# Suggested addition to config.py PATHS:
#   "tste_cv_results": DATA / "triplets_results" / "final_experiment" / "cleaned_results" / "tste_cv_results",
BASE_INPUT_DIR  = get_path("experiment_cleaned")
BASE_OUTPUT_DIR = ensure("experiment_cleaned").parent / "cleaned_results" / "tste_cv_results"

# Simpler: derive directly
from src.config import DATA
BASE_INPUT_DIR  = DATA / "triplets_results" / "final_experiment" / "cleaned_results"
BASE_OUTPUT_DIR = BASE_INPUT_DIR / "tste_cv_results"
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# HELPERS
# =========================================================
def compute_triplet_accuracy(X, triplets):
    D = cdist(X, X, metric="euclidean")
    return np.mean(D[triplets[:, 0], triplets[:, 1]] < D[triplets[:, 0], triplets[:, 2]])


def run_lopo_cv(df, game_name, dims_to_try):
    triplets_per_p          = df.groupby(PARTICIPANT_COL).size()
    all_participants         = sorted(df[PARTICIPANT_COL].unique())
    eligible_test            = sorted(triplets_per_p[triplets_per_p >= MIN_TRIPLETS_PER_PARTICIPANT].index)
    excluded                 = sorted(set(all_participants) - set(eligible_test))

    print(f"\n  Total participants:  {len(all_participants)}")
    print(f"  Eligible test folds: {len(eligible_test)} (>= {MIN_TRIPLETS_PER_PARTICIPANT} triplets)")
    if excluded:
        print(f"  Excluded as test:    {len(excluded)} (still in training)")
        for p in excluded:
            print(f"    {p}: {triplets_per_p[p]} triplet(s)")
    print(f"  Total triplets:      {len(df)}")
    print(f"  Triplets/participant — min={triplets_per_p.min()}, "
          f"median={triplets_per_p.median():.0f}, max={triplets_per_p.max()}")

    fold_results = []

    for dim in dims_to_try:
        print(f"\n  --- dim={dim} ---")
        for fold_idx, test_p in enumerate(eligible_test, 1):
            train_df = df[df[PARTICIPANT_COL] != test_p]
            test_df  = df[df[PARTICIPANT_COL] == test_p]

            train_triplets = np.ascontiguousarray(train_df[TRIPLET_COLS].values, dtype=np.int32)
            test_triplets  = np.ascontiguousarray(test_df[TRIPLET_COLS].values,  dtype=np.int32)

            X         = cy_tste.tste(train_triplets, no_dims=dim, max_iter=MAX_ITER, verbose=False, use_log=USE_LOG)
            train_acc = compute_triplet_accuracy(X, train_triplets)
            test_acc  = compute_triplet_accuracy(X, test_triplets)

            fold_results.append({"game": game_name, "dimension": dim, "fold": fold_idx,
                                  "test_participant": test_p, "n_train_triplets": len(train_triplets),
                                  "n_test_triplets": len(test_triplets),
                                  "train_accuracy": train_acc, "test_accuracy": test_acc})

            if fold_idx % 10 == 0 or fold_idx == len(eligible_test):
                print(f"    Fold {fold_idx}/{len(eligible_test)} done")

    fold_df    = pd.DataFrame(fold_results)
    summary_df = (fold_df.groupby(["game", "dimension"], as_index=False)
                  .agg(mean_train_accuracy=("train_accuracy", "mean"), std_train_accuracy=("train_accuracy", "std"),
                       mean_test_accuracy=("test_accuracy", "mean"),  std_test_accuracy=("test_accuracy", "std"),
                       n_folds=("fold", "count")))
    return fold_df, summary_df


def run_triplet_split_cv(df, game_name, dims_to_try,
                         train_percents=None, n_repeats=20):
    if train_percents is None:
        train_percents = [0.2, 0.4, 0.6, 0.8]

    triplets_all = np.ascontiguousarray(df[TRIPLET_COLS].values, dtype=np.int32)
    n_total      = len(triplets_all)
    print(f"\n  Total triplets for split CV: {n_total}")
    results = []

    for dim in dims_to_try:
        print(f"\n  --- dim={dim} ---")
        for p in train_percents:
            n_train = int(n_total * p)
            for repeat in range(n_repeats):
                np.random.seed(42 + repeat)
                idx   = np.random.permutation(n_total)
                train_triplets = triplets_all[idx[:n_train]]
                test_triplets  = triplets_all[idx[n_train:]]

                X         = cy_tste.tste(train_triplets, no_dims=dim, max_iter=MAX_ITER, verbose=False, use_log=USE_LOG)
                train_acc = compute_triplet_accuracy(X, train_triplets)
                test_acc  = compute_triplet_accuracy(X, test_triplets)

                results.append({"game": game_name, "dimension": dim, "train_percent": p, "repeat": repeat,
                                 "n_train": len(train_triplets), "n_test": len(test_triplets),
                                 "train_accuracy": train_acc, "test_accuracy": test_acc})

    results_df = pd.DataFrame(results)
    summary_df = (results_df.groupby(["game", "dimension", "train_percent"], as_index=False)
                  .agg(mean_train_accuracy=("train_accuracy", "mean"), std_train_accuracy=("train_accuracy", "std"),
                       mean_test_accuracy=("test_accuracy", "mean"),  std_test_accuracy=("test_accuracy", "std")))
    return results_df, summary_df


def print_summary(summary_df, label):
    print(f"\n  ── {label} ──")
    best = (summary_df.sort_values("mean_test_accuracy", ascending=False)
            .groupby("game", as_index=False).first()
            [["game", "dimension", "mean_train_accuracy", "std_train_accuracy",
              "mean_test_accuracy", "std_test_accuracy"]])
    print(best.to_string(index=False))


# =========================================================
# MAIN
# =========================================================
all_lopo_folds    = []
all_lopo_summary  = []
all_split_folds   = []
all_split_summary = []

for game_name in GAMES:
    input_file = BASE_INPUT_DIR / f"{game_name}_triplets_constraints.csv"

    if not input_file.exists():
        print(f"WARNING: {input_file} not found. Skipping {game_name}.")
        continue

    print(f"\n{'='*60}\nGAME: {game_name}\n{'='*60}")
    df = pd.read_csv(input_file)

    required = [PARTICIPANT_COL] + TRIPLET_COLS
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {game_name}: {missing}")

    df[TRIPLET_COLS] = df[TRIPLET_COLS].astype(int)
    triplets = np.ascontiguousarray(df[TRIPLET_COLS].values, dtype=np.int32)

    print(f"  Triplets:   {triplets.shape}  |  Participants: {df[PARTICIPANT_COL].nunique()}")
    print(f"  Clip range: {np.min(triplets)} – {np.max(triplets)}")
    print(f"  Possible C({N_CLIPS},3) = {N_CLIPS*(N_CLIPS-1)*(N_CLIPS-2)//6}")

    assert triplets.shape[1] == 3
    assert np.min(triplets) >= 0
    assert np.max(triplets) < N_CLIPS, f"Expected indices 0–{N_CLIPS-1}, got max={np.max(triplets)}"

    game_output_dir = BASE_OUTPUT_DIR / game_name
    game_output_dir.mkdir(parents=True, exist_ok=True)

    # ── LOPO-CV ───────────────────────────────────────────
    print("\n[LOPO-CV]")
    lopo_fold_df, lopo_summary_df = run_lopo_cv(df, game_name, DIMS_TO_TRY)
    print_summary(lopo_summary_df, "LOPO-CV best dim")
    lopo_fold_df.to_csv(   game_output_dir / f"{game_name}_lopo_fold_results.csv", index=False)
    lopo_summary_df.to_csv(game_output_dir / f"{game_name}_lopo_summary.csv",      index=False)
    all_lopo_folds.append(lopo_fold_df.round(4))
    all_lopo_summary.append(lopo_summary_df.round(4))

    # ── Triplet-split CV ──────────────────────────────────
    print("\n[Triplet-split CV]")
    split_fold_df, split_summary_df = run_triplet_split_cv(
        df, game_name, DIMS_TO_TRY,
        train_percents=[0.2, 0.4, 0.6, 0.8], n_repeats=20,
    )
    print_summary(split_summary_df, "Split-CV best dim")
    split_fold_df.to_csv(   game_output_dir / f"{game_name}_split_fold_results.csv", index=False)
    split_summary_df.to_csv(game_output_dir / f"{game_name}_split_summary.csv",      index=False)
    all_split_folds.append(split_fold_df.round(4))
    all_split_summary.append(split_summary_df.round(4))

# ── Combined outputs ──────────────────────────────────────
if all_lopo_folds:
    pd.concat(all_lopo_folds, ignore_index=True).to_csv(BASE_OUTPUT_DIR / "all_games_lopo_fold_results.csv", index=False)
if all_lopo_summary:
    combined = pd.concat(all_lopo_summary, ignore_index=True)
    combined.to_csv(BASE_OUTPUT_DIR / "all_games_lopo_summary.csv", index=False)
    print(f"\n{'='*60}\nFINAL LOPO SUMMARY\n{'='*60}")
    print(combined.sort_values(["game", "mean_test_accuracy"], ascending=[True, False]).to_string(index=False))
if all_split_folds:
    pd.concat(all_split_folds, ignore_index=True).to_csv(BASE_OUTPUT_DIR / "all_games_split_fold_results.csv", index=False)
if all_split_summary:
    pd.concat(all_split_summary, ignore_index=True).to_csv(BASE_OUTPUT_DIR / "all_games_split_summary.csv", index=False)

print("\nDone.")
