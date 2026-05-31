"""
Compare Human RDMs vs DQN RDMs using RSA
----------------------------------------

This script:

1. Loads one human RDM per game
2. Loads all DQN layer RDMs for that game
3. Compares human vs DQN using RSA (Spearman)
4. Saves:
   - CSV summary
   - heatmap
   - numpy matrix

Output:
One RSA score per game x DQN layer
"""

# =========================================================
# IMPORTS
# =========================================================
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rsatoolbox.rdm import RDMs, compare

# =========================================================
# CONFIGURATION
# =========================================================
GAMES = ["pong", "pacman", "spaceinvaders"] # "pacman", 
SEED="seed_42"
HUMAN_RDM_FOLDER = "../data/rdms_human_experiment_rsa" #"../data/triplets_results/own_data/cleaned_results/rdms_human_experiment_rsa" #"../data/rdms_human_experiment_rsa"
DQN_RDM_BASE_FOLDER = f"../data/test_16_rdms/pilot/{SEED}" #"../data/DQN_rdms"
SAVE_FOLDER = f"../data/test_16_RSA/optimized_RDMs/{SEED}" #"../data/triplets_results/own_data/cleaned_results/RSA_results/pilot/{SEED}" #"../data/test_16_RSA" #"../data/RSA_results"

os.makedirs(SAVE_FOLDER, exist_ok=True)

# =========================================================
# HELPER: load RDM as rsatoolbox RDM object
# =========================================================
def load_rdm_as_object(rdm_matrix, name="rdm"):
    """
    Converts a square RDM matrix into an rsatoolbox RDM object.
    """
    return RDMs(
        dissimilarities=np.array([rdm_matrix]),
        rdm_descriptors={"name": [name]}
    )

# =========================================================
# MAIN
# =========================================================
all_results = []

for game in GAMES:
    print("\n" + "="*60)
    print(f"PROCESSING GAME: {game}")
    print("="*60)

    # -----------------------------------------------------
    # Load human RDM
    # -----------------------------------------------------
    human_rdm_path = os.path.join(HUMAN_RDM_FOLDER, f"{game}_RDM.npy")

    if not os.path.exists(human_rdm_path):
        print(f"Human RDM not found for {game}: {human_rdm_path}")
        continue

    human_rdm = np.load(human_rdm_path)
    print(f"Loaded human RDM: {human_rdm.shape}")

    human_rdm_obj = load_rdm_as_object(human_rdm, name=f"{game}_human")

    # -----------------------------------------------------
    # Load DQN layer RDMs
    # -----------------------------------------------------
    game_dqn_folder = os.path.join(DQN_RDM_BASE_FOLDER, game)

    dqn_rdm_files = sorted(glob.glob(os.path.join(game_dqn_folder, f"{game}_*_correlation_rdm.npy")))

    # Exclude layer-RSA matrix if it exists
    dqn_rdm_files = [f for f in dqn_rdm_files if "RSA" not in os.path.basename(f)]

    if len(dqn_rdm_files) == 0:
        print(f"No DQN RDM files found for {game}")
        continue

    print(f"Found {len(dqn_rdm_files)} DQN layer RDMs")

    game_scores = []
    layer_names = []

    # -----------------------------------------------------
    # Compare human RDM vs each DQN layer RDM
    # -----------------------------------------------------
    for dqn_file in dqn_rdm_files:
        layer_name = os.path.basename(dqn_file).replace(f"{game}_", "").replace("_RDM.npy", "")
        layer_names.append(layer_name)

        dqn_rdm = np.load(dqn_file)
        print(f"Comparing with layer: {layer_name}, shape: {dqn_rdm.shape}")

        # Safety check: human and DQN RDMs must have same size
        if human_rdm.shape != dqn_rdm.shape:
            print(f"Shape mismatch for {game} - {layer_name}: human {human_rdm.shape}, dqn {dqn_rdm.shape}")
            continue

        dqn_rdm_obj = load_rdm_as_object(dqn_rdm, name=f"{game}_{layer_name}")

        # RSA comparison (Spearman) #! ESTO ES RANKED!!! SIN RANCKED TENDRIA QUE SER PEARSON
        rsa_score = compare(human_rdm_obj, dqn_rdm_obj, method="spearman")[0, 0]
        game_scores.append(rsa_score)

        all_results.append({
            "game": game,
            "layer": layer_name,
            "rsa_spearman": rsa_score
        })

        print(f"RSA ({game}, {layer_name}) = {rsa_score:.4f}")

    # -----------------------------------------------------
    # Save game-specific results
    # -----------------------------------------------------
    if len(game_scores) > 0:
        # Save matrix as npy
        rsa_matrix = np.array(game_scores).reshape(1, -1)
        npy_path = os.path.join(SAVE_FOLDER, f"{game}_human_vs_DQN_RSA.npy")
        np.save(npy_path, rsa_matrix)
        print(f"Saved RSA matrix: {npy_path}")

        # Plot heatmap
        plt.figure(figsize=(max(6, len(layer_names) * 1.5), 2.5))
        im = plt.imshow(rsa_matrix, cmap="viridis", aspect="auto")
        plt.colorbar(im, label="Spearman RSA")
        plt.xticks(range(len(layer_names)), layer_names, rotation=45)
        plt.yticks([0], ["Human"])
        plt.title(f"{game}: Human vs DQN Layers")
        plt.tight_layout()

        png_path = os.path.join(SAVE_FOLDER, f"{game}_human_vs_DQN_RSA_heatmap.png")
        plt.savefig(png_path, dpi=300)
        plt.close()
        print(f"Saved heatmap: {png_path}")

# =========================================================
# SAVE GLOBAL CSV SUMMARY
# =========================================================
results_df = pd.DataFrame(all_results)

csv_path = os.path.join(SAVE_FOLDER, "human_vs_DQN_RSA_summary.csv")
results_df.to_csv(csv_path, index=False)
print(f"\nSaved global RSA summary: {csv_path}")

# =========================================================
# OPTIONAL: PLOT COMBINED HEATMAP (games x layers)
# =========================================================
if len(results_df) > 0:
    pivot_df = results_df.pivot(index="game", columns="layer", values="rsa_spearman")

    plt.figure(figsize=(8, 4))
    im = plt.imshow(pivot_df.values, cmap="viridis", aspect="auto")
    plt.colorbar(im, label="Spearman RSA")

    plt.xticks(range(len(pivot_df.columns)), pivot_df.columns, rotation=45)
    plt.yticks(range(len(pivot_df.index)), pivot_df.index)
    plt.title("Human vs DQN RSA Across Games")
    plt.tight_layout()

    combined_png = os.path.join(SAVE_FOLDER, "human_vs_DQN_RSA_all_games_heatmap.png")
    plt.savefig(combined_png, dpi=300)
    plt.close()

    print(f"Saved combined heatmap: {combined_png}")

print("\nDone! Human vs DQN RSA completed.")