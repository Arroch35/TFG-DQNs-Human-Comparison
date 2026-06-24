"""
4_1_pilot_tste_cv.py
t-STE cross-validation — pilot version.
10 clips per game. Runs triplet-split CV (LOPO commented out as in original).
"""
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from src.config import GAMES, get_path, ensure
import cy_tste

# =========================================================
# CONFIG
# =========================================================
DIMS_TO_TRY = [2, 3, 5, 8, 10]

PARTICIPANT_COL = "participant_id"
TRIPLET_COLS    = ["reference", "near", "far"]

MAX_ITER = 1000
USE_LOG  = True

# Suggested addition to config.py PATHS:
#   "exp_own_data":      DATA / "triplets_results" / "own_data" / "cleaned_results",
#   "tste_cv_pilot":     DATA / "triplets_results" / "own_data" / "cleaned_results" / "tste_cv_results",
from src.config import DATA
BASE_INPUT_DIR  = get_path("experiment_pilot")
BASE_OUTPUT_DIR = ensure("results_cv_pilot")

# =========================================================
# HELPERS
# =========================================================
def compute_triplet_accuracy(X, triplets):
    D = cdist(X, X, metric="euclidean")
    return np.mean(D[triplets[:, 0], triplets[:, 1]] < D[triplets[:, 0], triplets[:, 2]])


def run_lopo_cv(df, game_name, dims_to_try):
    participants = sorted(df[PARTICIPANT_COL].unique())
    print(f"\n===== GAME: {game_name} =====\nParticipants: {participants}")
    fold_results = []

    for dim in dims_to_try:
        print(f"\n--- dim={dim} ---")
        for fold_idx, test_p in enumerate(participants, 1):
            train_df, test_df = df[df[PARTICIPANT_COL] != test_p], df[df[PARTICIPANT_COL] == test_p]
            train_triplets = np.ascontiguousarray(train_df[TRIPLET_COLS].values, dtype=np.int32)
            test_triplets  = np.ascontiguousarray(test_df[TRIPLET_COLS].values,  dtype=np.int32)

            X          = cy_tste.tste(train_triplets, no_dims=dim, max_iter=MAX_ITER, verbose=False, use_log=USE_LOG)
            train_acc  = compute_triplet_accuracy(X, train_triplets)
            test_acc   = compute_triplet_accuracy(X, test_triplets)
            print(f"Fold {fold_idx}/{len(participants)} | Train: {train_acc:.3f} | Test: {test_acc:.3f}")

            fold_results.append({"game": game_name, "dimension": dim, "fold": fold_idx,
                                  "test_participant": test_p, "n_train_triplets": len(train_triplets),
                                  "n_test_triplets": len(test_triplets),
                                  "train_accuracy": train_acc, "test_accuracy": test_acc})

    fold_df = pd.DataFrame(fold_results)
    summary_df = (fold_df.groupby(["game", "dimension"], as_index=False)
                  .agg(mean_train_accuracy=("train_accuracy", "mean"), std_train_accuracy=("train_accuracy", "std"),
                       mean_test_accuracy=("test_accuracy", "mean"),  std_test_accuracy=("test_accuracy", "std"),
                       n_folds=("fold", "count")))
    return fold_df, summary_df


def run_triplet_split_cv(df, game_name, dims_to_try,
                         train_percents=None, n_repeats=10):
    if train_percents is None:
        train_percents = [0.1, 0.2, 0.4, 0.6, 0.8]

    print(f"\n===== GAME: {game_name} =====")
    triplets_all = np.ascontiguousarray(df[TRIPLET_COLS].values, dtype=np.int32)
    n_total      = len(triplets_all)
    print(f"Total triplets: {n_total}")
    results = []

    for dim in dims_to_try:
        print(f"\n--- dim={dim} ---")
        for p in train_percents:
            n_train = int(n_total * p)
            for repeat in range(n_repeats):
                np.random.seed(42 + repeat)
                idx    = np.random.permutation(n_total)
                train_triplets = triplets_all[idx[:n_train]]
                test_triplets  = triplets_all[idx[n_train:]]

                X         = cy_tste.tste(train_triplets, no_dims=dim, max_iter=MAX_ITER, verbose=False, use_log=USE_LOG)
                train_acc = compute_triplet_accuracy(X, train_triplets)
                test_acc  = compute_triplet_accuracy(X, test_triplets)
                print(f"[dim={dim} | p={p} | rep={repeat}] Train: {train_acc:.3f} | Test: {test_acc:.3f}")

                results.append({"game": game_name, "dimension": dim, "train_percent": p, "repeat": repeat,
                                 "n_train": len(train_triplets), "n_test": len(test_triplets),
                                 "train_accuracy": train_acc, "test_accuracy": test_acc})

    results_df = pd.DataFrame(results)
    summary_df = (results_df.groupby(["game", "dimension", "train_percent"], as_index=False)
                  .agg(mean_train_accuracy=("train_accuracy", "mean"), std_train_accuracy=("train_accuracy", "std"),
                       mean_test_accuracy=("test_accuracy", "mean"),  std_test_accuracy=("test_accuracy", "std")))
    return results_df, summary_df


# =========================================================
# MAIN
# =========================================================
all_fold_results    = []
all_summary_results = []

for game_name in GAMES:
    input_file = BASE_INPUT_DIR / f"{game_name}_tste_constraints.csv"

    if not input_file.exists():
        print(f"WARNING: {input_file} not found. Skipping.")
        continue

    print(f"\nLoading: {input_file}")
    df = pd.read_csv(input_file)

    required = [PARTICIPANT_COL] + TRIPLET_COLS
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {game_name}: {missing}")

    df[TRIPLET_COLS] = df[TRIPLET_COLS].astype(int)
    triplets = np.ascontiguousarray(df[TRIPLET_COLS].values, dtype=np.int32)
    print(f"Triplets shape: {triplets.shape}")
    print(f"Unique: {len(np.unique(triplets, axis=0))} / {len(triplets)}")

    # Pilot sanity check: 10 clips (0–9)
    assert triplets.shape[1] == 3
    assert np.min(triplets) >= 0
    assert np.max(triplets) < 10, "Expected clip indices 0–9 for pilot"

    fold_df, summary_df = run_triplet_split_cv(
        df, game_name, DIMS_TO_TRY,
        train_percents=[0.1, 0.2, 0.4, 0.6, 0.8],
        n_repeats=20,
    )

    game_output_dir = BASE_OUTPUT_DIR / game_name
    game_output_dir.mkdir(parents=True, exist_ok=True)

    fold_df.to_csv(   game_output_dir / f"{game_name}_lopo_fold_results.csv", index=False)
    summary_df.to_csv(game_output_dir / f"{game_name}_lopo_summary.csv",      index=False)

    all_fold_results.append(fold_df)
    all_summary_results.append(summary_df)

# ── Combined outputs ──────────────────────────────────────
if all_fold_results:
    pd.concat(all_fold_results,    ignore_index=True).round(4).to_csv(BASE_OUTPUT_DIR / "all_games_lopo_fold_results.csv", index=False)
if all_summary_results:
    combined = pd.concat(all_summary_results, ignore_index=True).round(4)
    combined.to_csv(BASE_OUTPUT_DIR / "all_games_lopo_summary.csv", index=False)
    print("\n===== FINAL SUMMARY =====")
    print(combined.sort_values(["game", "mean_test_accuracy"], ascending=[True, False]).to_string())
