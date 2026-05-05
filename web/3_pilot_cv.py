import os
import numpy as np
import pandas as pd
import cy_tste
from scipy.spatial.distance import cdist

# =========================
# 1) CONFIGURATION
# =========================
games = ["pacman", "pong", "spaceinvaders"]
dims_to_try = [2, 3, 5, 8, 10]

base_input_dir = "../data/triplets_results/own_data/cleaned_results"#"../data/cleaned_results"
base_output_dir = "../data/triplets_results/own_data/cleaned_results/tste_cv_results/" #"../data/tste_cv_results"
os.makedirs(base_output_dir, exist_ok=True)


participant_col = "participant_id"

triplet_cols = ["reference", "near", "far"]

# =========================
# 2) HELPER FUNCTIONS
# =========================
def compute_triplet_accuracy(X, triplets):
    """
    Computes fraction of triplets satisfied by embedding X.
    triplet format: [reference, near, far]
    """
    D = cdist(X, X, metric='euclidean')
    satisfied = np.mean(
        D[triplets[:, 0], triplets[:, 1]] < D[triplets[:, 0], triplets[:, 2]]
    )
    return satisfied


def run_lopo_cv(df, game_name, dims_to_try, participant_col, triplet_cols,
                max_iter=1000, use_log=True):
    """
    Leave-One-Participant-Out CV for one game.
    """
    participants = sorted(df[participant_col].unique())
    print(f"\n===== GAME: {game_name} =====")
    print(f"Participants found: {participants}")

    fold_results = []

    for dim in dims_to_try:
        print(f"\n--- Testing dimension: {dim} ---")

        for fold_idx, test_participant in enumerate(participants, start=1):
            print(f"\nFold {fold_idx}/{len(participants)} - Test participant: {test_participant}")

            train_df = df[df[participant_col] != test_participant].copy()
            test_df = df[df[participant_col] == test_participant].copy()

            train_triplets = np.ascontiguousarray(
                train_df[triplet_cols].values, dtype=np.int32
            )
            test_triplets = np.ascontiguousarray(
                test_df[triplet_cols].values, dtype=np.int32
            )

            # Basic checks
            assert train_triplets.shape[1] == 3
            assert test_triplets.shape[1] == 3
            assert np.min(train_triplets) >= 0
            assert np.min(test_triplets) >= 0

            # Fit t-STE only on training triplets
            X = cy_tste.tste(
                train_triplets,
                no_dims=dim,
                max_iter=max_iter,
                verbose=False,
                use_log=use_log
            )

            # Evaluate on train and test
            train_acc = compute_triplet_accuracy(X, train_triplets)
            test_acc = compute_triplet_accuracy(X, test_triplets)

            print(f"Train acc: {train_acc:.3f} | Test acc: {test_acc:.3f}")

            fold_results.append({
                "game": game_name,
                "dimension": dim,
                "fold": fold_idx,
                "test_participant": test_participant,
                "n_train_triplets": len(train_triplets),
                "n_test_triplets": len(test_triplets),
                "train_accuracy": train_acc,
                "test_accuracy": test_acc
            })

    fold_results_df = pd.DataFrame(fold_results)

    # Summary stats
    summary_df = (
        fold_results_df
        .groupby(["game", "dimension"], as_index=False)
        .agg(
            mean_train_accuracy=("train_accuracy", "mean"),
            std_train_accuracy=("train_accuracy", "std"),
            mean_test_accuracy=("test_accuracy", "mean"),
            std_test_accuracy=("test_accuracy", "std"),
            n_folds=("fold", "count")
        )
    )

    return fold_results_df, summary_df


def run_triplet_split_cv(df, game_name, dims_to_try, triplet_cols,
                        train_percents=[0.2, 0.4, 0.6, 0.8],
                        n_repeats=10,
                        max_iter=1000,
                        use_log=True):

    print(f"\n===== GAME: {game_name} =====")

    triplets_all = np.ascontiguousarray(
        df[triplet_cols].values, dtype=np.int32
    )

    n_total = len(triplets_all)
    print(f"Total triplets: {n_total}")

    results = []

    for dim in dims_to_try:
        print(f"\n--- Testing dimension: {dim} ---")

        for p in train_percents:
            n_train = int(n_total * p)

            print(f"\nTrain percent: {p} ({n_train} triplets)")

            for repeat in range(n_repeats):

                np.random.seed(42 + repeat)

                idx = np.random.permutation(n_total)

                train_idx = idx[:n_train]
                test_idx = idx[n_train:]

                train_triplets = triplets_all[train_idx]
                test_triplets = triplets_all[test_idx]

                # Fit
                X = cy_tste.tste(
                    train_triplets,
                    no_dims=dim,
                    max_iter=max_iter,
                    verbose=False,
                    use_log=use_log
                )

                # Evaluate
                train_acc = compute_triplet_accuracy(X, train_triplets)
                test_acc = compute_triplet_accuracy(X, test_triplets)

                print(f"[dim={dim} | p={p} | rep={repeat}] "
                      f"Train: {train_acc:.3f} | Test: {test_acc:.3f}")

                results.append({
                    "game": game_name,
                    "dimension": dim,
                    "train_percent": p,
                    "repeat": repeat,
                    "n_train": len(train_triplets),
                    "n_test": len(test_triplets),
                    "train_accuracy": train_acc,
                    "test_accuracy": test_acc
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


# =========================
# 3) MAIN LOOP OVER GAMES
# =========================
all_fold_results = []
all_summary_results = []

for game_name in games:
    input_file = os.path.join(base_input_dir, f"{game_name}_tste_constraints.csv")

    if not os.path.exists(input_file):
        print(f"WARNING: {input_file} not found. Skipping.")
        continue

    print(f"\nLoading: {input_file}")
    df = pd.read_csv(input_file)

    # Check required columns
    required_cols = [participant_col] + triplet_cols
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {game_name}: {missing_cols}")

    # Convert triplet columns to int
    df[triplet_cols] = df[triplet_cols].astype(int)

    # Basic sanity check
    triplets = np.ascontiguousarray(df[triplet_cols].values, dtype=np.int32)
    print("Triplets shape:", triplets.shape)
    print("First 10 triplets:")
    print(triplets[:10])

    n_items = np.max(triplets) + 1
    print("Number of clips inferred:", n_items)

    assert triplets.shape[1] == 3, "Triplets must have shape (N, 3)"
    assert np.min(triplets) >= 0, "Triplets must be 0-indexed"
    assert np.max(triplets) < 10, "Expected clip indices 0-9 for 10 clips"

    # Optional duplicate analysis
    triplets_unique = np.unique(triplets, axis=0)
    print("Unique triplets:", len(triplets_unique), "out of", len(triplets))

    # Run LOPO-CV
    # fold_df, summary_df = run_lopo_cv(
    #     df=df,
    #     game_name=game_name,
    #     dims_to_try=dims_to_try,
    #     participant_col=participant_col,
    #     triplet_cols=triplet_cols,
    #     max_iter=1000,
    #     use_log=True
    # )

    fold_df, summary_df = run_triplet_split_cv(
        df=df,
        game_name=game_name,
        dims_to_try=dims_to_try,
        triplet_cols=triplet_cols,
        train_percents=[0.1, 0.2, 0.4, 0.6, 0.8],
        n_repeats=20,
        max_iter=1000,
        use_log=True
    )

    # Save per-game outputs
    game_output_dir = os.path.join(base_output_dir, game_name)
    os.makedirs(game_output_dir, exist_ok=True)

    fold_df.to_csv(os.path.join(game_output_dir, f"{game_name}_lopo_fold_results.csv"), index=False)
    summary_df.to_csv(os.path.join(game_output_dir, f"{game_name}_lopo_summary.csv"), index=False)

    all_fold_results.append(fold_df)
    all_summary_results.append(summary_df)

# =========================
# 4) SAVE COMBINED RESULTS
# =========================
if all_fold_results:
    all_fold_results_df = pd.concat(all_fold_results, ignore_index=True)
    all_fold_results_df.to_csv(os.path.join(base_output_dir, "all_games_lopo_fold_results.csv"), index=False)

if all_summary_results:
    all_summary_results_df = pd.concat(all_summary_results, ignore_index=True)
    all_summary_results_df.to_csv(os.path.join(base_output_dir, "all_games_lopo_summary.csv"), index=False)

    print("\n===== FINAL SUMMARY =====")
    print(all_summary_results_df.sort_values(["game", "mean_test_accuracy"], ascending=[True, False]))

#TODO: Parece que funciona. Ahora tengo que hacer el script de los csvs que los coja en bucle y los meta todos en uno, teniendo en cuento un ID de la persona, para que luego sea mas facil quitar los que no quiero segun la enquesta
#TODO: Aquí tendré que usar pandas para que en vez del video me de el índice y lo cree en una variable u otro csv para mapear indice y video
#TODO: Mirar como usar bien la funcion
#TODO: Entrenar bien los DQNs segun fuente fiables, y que sean varios para hacer medias
#TODO: Poner en todos lados qeu se lea un folder, que se cree tambien

#TODO: Despues de cenar, buscar y entrenar DQN para los juegos. CON SEMILLAS DISTINTAS
#TODO: Mañana: Hacer esto en bucle para los 3 juegos, hacer las RDMs para esto y las RDMs para los DQNs, y luego el RSA
#TODO: Probar de quitar a personas, a ver si los resultados mejoran
#!Esto no se muy bien como fucniona aún, porque esto es machine learning, y se supone que no debe haver overfiting y tal, así que mira muy bien como funciona