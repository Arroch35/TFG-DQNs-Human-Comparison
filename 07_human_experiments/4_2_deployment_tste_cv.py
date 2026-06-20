"""
t-STE Cross-Validation — Deployment Version
============================================
15 clips, variable participants (target ~41), variable triplets per
participant per game (target 6, but participants may have dropped out).

Runs Leave-One-Participant-Out (LOPO) CV as primary method,
plus triplet-split CV for comparison.

Key differences from pilot:
  - Clip indices 0–14 (15 clips)
  - LOPO is now statistically meaningful with enough participants
  - Participants with fewer than MIN_TRIPLETS_PER_PARTICIPANT are
    excluded from LOPO to avoid uninformative folds (but their
    triplets are still used in the training set of other folds
    and in triplet-split CV)
"""

import os
import numpy as np
import pandas as pd
import cy_tste
from scipy.spatial.distance import cdist

# =========================
# CONFIGURATION
# =========================
GAMES        = ["pong", "pacman", "spaceinvaders"]
DIMS_TO_TRY  = [2, 3, 5, 8, 10, 15, 20, 50]
N_CLIPS      = 15          # deployment uses 15 clips (indices 0–14)

BASE_INPUT_DIR  = "../data/triplets_results/final_experiment/cleaned_results"
BASE_OUTPUT_DIR = "../data/triplets_results/final_experiment/cleaned_results/tste_cv_results/"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

PARTICIPANT_COL = "participant_id"
TRIPLET_COLS    = ["reference", "near", "far"]

MAX_ITER = 1000
USE_LOG  = True

# Participants with fewer than this many triplets for a given game
# are excluded as LOPO test folds (too noisy to evaluate on).
# Their triplets are still used in all other folds' training sets
# and in the triplet-split CV.
MIN_TRIPLETS_PER_PARTICIPANT = 6

# =========================
# HELPERS
# =========================
def compute_triplet_accuracy(X, triplets):
    """Fraction of triplets satisfied by embedding X."""
    D = cdist(X, X, metric="euclidean")
    return np.mean(
        D[triplets[:, 0], triplets[:, 1]] < D[triplets[:, 0], triplets[:, 2]]
    )


def run_lopo_cv(df, game_name, dims_to_try):
    """
    Leave-One-Participant-Out CV.

    All participants contribute their triplets to every training fold.
    Only participants with >= MIN_TRIPLETS_PER_PARTICIPANT are used as
    test folds, to avoid uninformative accuracy estimates.
    """
    # Count triplets per participant
    triplets_per_p = df.groupby(PARTICIPANT_COL).size()

    all_participants      = sorted(df[PARTICIPANT_COL].unique())
    eligible_test_participants = sorted(
        triplets_per_p[triplets_per_p >= MIN_TRIPLETS_PER_PARTICIPANT].index
    )
    excluded = sorted(set(all_participants) - set(eligible_test_participants))

    print(f"\n  Total participants:    {len(all_participants)}")
    print(f"  Eligible test folds:   {len(eligible_test_participants)} "
          f"(>= {MIN_TRIPLETS_PER_PARTICIPANT} triplets)")
    if excluded:
        print(f"  Excluded as test fold: {len(excluded)} participants "
              f"(still used in training)")
        for p in excluded:
            print(f"    {p}: {triplets_per_p[p]} triplet(s)")
    print(f"  Total triplets:        {len(df)}")
    print(f"  Triplets/participant — "
          f"min={triplets_per_p.min()}, "
          f"median={triplets_per_p.median():.0f}, "
          f"max={triplets_per_p.max()}")

    fold_results = []

    for dim in dims_to_try:
        print(f"\n  --- dim={dim} ---")

        for fold_idx, test_p in enumerate(eligible_test_participants, start=1):
            # Training: ALL participants except the test one
            train_df = df[df[PARTICIPANT_COL] != test_p]
            test_df  = df[df[PARTICIPANT_COL] == test_p]

            train_triplets = np.ascontiguousarray(
                train_df[TRIPLET_COLS].values, dtype=np.int32
            )
            test_triplets = np.ascontiguousarray(
                test_df[TRIPLET_COLS].values, dtype=np.int32
            )

            # Fit on training participants
            X = cy_tste.tste(
                train_triplets,
                no_dims=dim,
                max_iter=MAX_ITER,
                verbose=False,
                use_log=USE_LOG,
            )

            train_acc = compute_triplet_accuracy(X, train_triplets)
            test_acc  = compute_triplet_accuracy(X, test_triplets)

            fold_results.append({
                "game":             game_name,
                "dimension":        dim,
                "fold":             fold_idx,
                "test_participant": test_p,
                "n_train_triplets": len(train_triplets),
                "n_test_triplets":  len(test_triplets),
                "train_accuracy":   train_acc,
                "test_accuracy":    test_acc,
            })

            if fold_idx % 10 == 0 or fold_idx == len(eligible_test_participants):
                print(f"    Fold {fold_idx}/{len(eligible_test_participants)} done")

    fold_df = pd.DataFrame(fold_results)

    summary_df = (
        fold_df
        .groupby(["game", "dimension"], as_index=False)
        .agg(
            mean_train_accuracy=("train_accuracy", "mean"),
            std_train_accuracy=("train_accuracy", "std"),
            mean_test_accuracy=("test_accuracy", "mean"),
            std_test_accuracy=("test_accuracy", "std"),
            n_folds=("fold", "count"),
        )
    )

    return fold_df, summary_df


def run_triplet_split_cv(df, game_name, dims_to_try,
                         train_percents=[0.2, 0.4, 0.6, 0.8],
                         n_repeats=20):
    """
    Random triplet-split CV across all participants pooled.
    Complementary to LOPO — tests reconstruction stability
    as a function of data quantity.
    """
    triplets_all = np.ascontiguousarray(
        df[TRIPLET_COLS].values, dtype=np.int32
    )
    n_total = len(triplets_all)
    print(f"\n  Total triplets for split CV: {n_total}")

    results = []

    for dim in dims_to_try:
        print(f"\n  --- dim={dim} ---")

        for p in train_percents:
            n_train = int(n_total * p)

            for repeat in range(n_repeats):
                np.random.seed(42 + repeat)
                idx = np.random.permutation(n_total)

                train_triplets = triplets_all[idx[:n_train]]
                test_triplets  = triplets_all[idx[n_train:]]

                X = cy_tste.tste(
                    train_triplets,
                    no_dims=dim,
                    max_iter=MAX_ITER,
                    verbose=False,
                    use_log=USE_LOG,
                )

                train_acc = compute_triplet_accuracy(X, train_triplets)
                test_acc  = compute_triplet_accuracy(X, test_triplets)

                results.append({
                    "game":            game_name,
                    "dimension":       dim,
                    "train_percent":   p,
                    "repeat":          repeat,
                    "n_train":         len(train_triplets),
                    "n_test":          len(test_triplets),
                    "train_accuracy":  train_acc,
                    "test_accuracy":   test_acc,
                })

    results_df = pd.DataFrame(results)

    summary_df = (
        results_df
        .groupby(["game", "dimension", "train_percent"], as_index=False)
        .agg(
            mean_train_accuracy=("train_accuracy", "mean"),
            std_train_accuracy=("train_accuracy", "std"),
            mean_test_accuracy=("test_accuracy", "mean"),
            std_test_accuracy=("test_accuracy", "std"),
        )
    )

    return results_df, summary_df


def print_summary(summary_df, method_name):
    print(f"\n  ── {method_name} summary ──")
    best = (
        summary_df
        .sort_values("mean_test_accuracy", ascending=False)
        .groupby("game", as_index=False)
        .first()
    )[["game", "dimension", "mean_train_accuracy",
       "std_train_accuracy", "mean_test_accuracy", "std_test_accuracy"]]
    print(best.to_string(index=False))


# =========================
# MAIN LOOP
# =========================
all_lopo_folds    = []
all_lopo_summary  = []
all_split_folds   = []
all_split_summary = []

for game_name in GAMES:
    input_file = os.path.join(
        BASE_INPUT_DIR, f"{game_name}_triplets_constraints.csv"
    )

    if not os.path.exists(input_file):
        print(f"WARNING: {input_file} not found. Skipping {game_name}.")
        continue

    print(f"\n{'='*60}")
    print(f"GAME: {game_name}")
    print(f"{'='*60}")

    df = pd.read_csv(input_file)

    # Validate columns
    required_cols = [PARTICIPANT_COL] + TRIPLET_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {game_name}: {missing}")

    df[TRIPLET_COLS] = df[TRIPLET_COLS].astype(int)

    triplets = np.ascontiguousarray(df[TRIPLET_COLS].values, dtype=np.int32)
    print(f"  Triplets shape:      {triplets.shape}")
    print(f"  Participants:        {df[PARTICIPANT_COL].nunique()}")
    print(f"  Clip indices range:  {np.min(triplets)} – {np.max(triplets)}")

    # Sanity checks for deployment (15 clips)
    assert triplets.shape[1] == 3,          "Triplets must have 3 columns"
    assert np.min(triplets) >= 0,           "Triplet indices must be >= 0"
    assert np.max(triplets) < N_CLIPS, (
        f"Expected clip indices 0–{N_CLIPS-1}, "
        f"got max={np.max(triplets)}"
    )

    n_unique = len(np.unique(triplets, axis=0))
    print(f"  Unique triplets:     {n_unique} / {len(triplets)}")
    print(f"  Possible triplets:   C({N_CLIPS},3) = "
          f"{N_CLIPS*(N_CLIPS-1)*(N_CLIPS-2)//6}")

    game_output_dir = os.path.join(BASE_OUTPUT_DIR, game_name)
    os.makedirs(game_output_dir, exist_ok=True)

    # ── LOPO-CV ───────────────────────────────────────────────
    print(f"\n[LOPO-CV]")
    lopo_fold_df, lopo_summary_df = run_lopo_cv(df, game_name, DIMS_TO_TRY)
    print_summary(lopo_summary_df, "LOPO-CV best dim per game")

    lopo_fold_df.to_csv(
        os.path.join(game_output_dir, f"{game_name}_lopo_fold_results.csv"),
        index=False,
    )
    lopo_summary_df.to_csv(
        os.path.join(game_output_dir, f"{game_name}_lopo_summary.csv"),
        index=False,
    )

    all_lopo_folds.append(lopo_fold_df.round(4))
    all_lopo_summary.append(lopo_summary_df.round(4))

    # ── Triplet-split CV ──────────────────────────────────────
    print(f"\n[Triplet-split CV]")
    split_fold_df, split_summary_df = run_triplet_split_cv(
        df, game_name, DIMS_TO_TRY,
        train_percents=[0.2, 0.4, 0.6, 0.8],
        n_repeats=20,
    )
    print_summary(split_summary_df, "Split-CV best dim per game")

    split_fold_df.to_csv(
        os.path.join(game_output_dir, f"{game_name}_split_fold_results.csv"),
        index=False,
    )
    split_summary_df.to_csv(
        os.path.join(game_output_dir, f"{game_name}_split_summary.csv"),
        index=False,
    )

    all_split_folds.append(split_fold_df.round(4))
    all_split_summary.append(split_summary_df.round(4))

# =========================
# SAVE COMBINED RESULTS
# =========================
if all_lopo_folds:
    pd.concat(all_lopo_folds, ignore_index=True).to_csv(
        os.path.join(BASE_OUTPUT_DIR, "all_games_lopo_fold_results.csv"),
        index=False,
    )
if all_lopo_summary:
    combined = pd.concat(all_lopo_summary, ignore_index=True)
    combined.to_csv(
        os.path.join(BASE_OUTPUT_DIR, "all_games_lopo_summary.csv"),
        index=False,
    )
    print(f"\n{'='*60}")
    print("FINAL LOPO SUMMARY (all games)")
    print(f"{'='*60}")
    print(combined.sort_values(
        ["game", "mean_test_accuracy"], ascending=[True, False]
    ).to_string(index=False))

if all_split_folds:
    pd.concat(all_split_folds, ignore_index=True).to_csv(
        os.path.join(BASE_OUTPUT_DIR, "all_games_split_fold_results.csv"),
        index=False,
    )
if all_split_summary:
    pd.concat(all_split_summary, ignore_index=True).to_csv(
        os.path.join(BASE_OUTPUT_DIR, "all_games_split_summary.csv"),
        index=False,
    )

print("\nDone.")
