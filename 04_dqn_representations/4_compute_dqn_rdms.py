import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm, compare, concat

from src.config import GAMES, SEEDS, REFERENCE_SEED, REPR, get_path, ensure
from src.utils import extract_layer_name

# =========================================================
# CONFIGURATION
# =========================================================
# Remove seed_42 from the main loop; it is the reference used
# for subset selection (FILTER_CSV), not a standard eval seed.
EVAL_SEEDS = [s for s in SEEDS if s != REFERENCE_SEED]  # seed_0 … seed_3

METHOD = REPR["rdm_method"]     # "correlation"

# =========================================================
# MAIN LOOP
# =========================================================
for seed in EVAL_SEEDS:

    activations_folder = get_path("activations_pool25_seed", seed=seed)

    for game in GAMES:
        print("\n" + "=" * 60)
        print(f"PROCESSING: {game}  |  {seed}")
        print("=" * 60)

        game_save_folder = ensure("rdms_subset15", seed=seed, game=game)

        # ── Find activation files ────────────────────────
        all_files = [
            f for f in os.listdir(activations_folder)
            if f.endswith("_activations.npz") and game in f.lower()
        ]

        # ── Filter to the 15-clip subset selected by SA ──
        # Uses REFERENCE_SEED's subset CSV (canonical selection)
        filter_csv = get_path("subsets_csv", seed=REFERENCE_SEED, game=game)

        if filter_csv.exists():
            filter_df    = pd.read_csv(filter_csv)
            allowed_names = set(
                filter_df["clip_name"].astype(str).str.replace(".mp4", "", regex=False)
            )
            activation_files = [
                f for f in all_files
                if f.replace("_activations.npz", "") in allowed_names
            ]
            print(f"Filtering: {len(activation_files)} of {len(all_files)} files kept.")
        else:
            activation_files = all_files
            print(f"Warning: subset CSV not found at {filter_csv}. Using all files.")

        if not activation_files:
            print(f"No activation files for {game} after filtering — skipping.")
            continue

        print(f"Found {len(activation_files)} activation files")

        # ── Collect activations per layer ────────────────
        layer_activations = {}
        clip_names        = []

        for file in activation_files:
            data      = np.load(activations_folder / file)
            clip_name = file.replace("_activations.npz", "")
            clip_names.append(clip_name)

            for key in data.files:
                layer_name = extract_layer_name(key)
                layer_activations.setdefault(layer_name, []).append(data[key])

        print("Layers found:", sorted(layer_activations.keys()))

        # ── Compute RDMs ─────────────────────────────────
        rdm_objects = []
        layer_names = sorted(layer_activations.keys())

        for layer_name in layer_names:
            print(f"\nLayer: {layer_name}")

            activations = np.concatenate(
                layer_activations[layer_name], axis=0
            ).astype(np.float32)

            print(f"Activation shape: {activations.shape}")

            if METHOD == "correlation":
                pca_path = get_path("models_pca_layer", game=game, seed=seed) / f"{game}_{layer_name}_pca.pkl"

                if not pca_path.exists():
                    raise FileNotFoundError(f"Missing PCA model: {pca_path}")

                pca_data    = joblib.load(pca_path)
                pca, scaler = pca_data["pca"], pca_data["scaler"]

                if scaler is not None:
                    activations = scaler.transform(activations)
                activations = pca.transform(activations)
                print(f"Post-PCA shape: {activations.shape}")
            else:
                print("Skipping PCA for euclidean distance")

            n_clips = activations.shape[0]

            dataset = Dataset(
                activations,
                obs_descriptors={"clips": np.array(clip_names)},
                channel_descriptors={"units": np.arange(activations.shape[1])},
            )

            rdm_obj    = calc_rdm(dataset, method=METHOD)
            rdm_matrix = rdm_obj.get_matrices()[0]
            rdm_objects.append(rdm_obj)

            # Save RDM matrix
            npy_path = game_save_folder / f"{game}_{layer_name}_{METHOD}_RDM.npy"
            np.save(npy_path, rdm_matrix)
            print(f"Saved RDM: {npy_path}")

            # Plot RDM heatmap
            ticks = np.arange(n_clips)
            plt.figure(figsize=(8, 8))
            im = plt.imshow(rdm_matrix, cmap="coolwarm", origin="upper")
            plt.colorbar(im)
            plt.title(f"{game} - RDM ({layer_name})")
            plt.xlabel("Clip"); plt.ylabel("Clip")
            plt.xticks(ticks, rotation=90, fontsize=6)
            plt.yticks(ticks, fontsize=6)
            plt.tight_layout()
            plt.savefig(game_save_folder / f"{game}_{layer_name}_{METHOD}_RDM.png", dpi=300)
            plt.close()

        # ── Layer-level RSA matrix ────────────────────────
        print("\nComputing RSA between layers…")
        combined_rdms = concat(rdm_objects)
        rsa_matrix    = compare(combined_rdms, combined_rdms, method="spearman")

        rsa_npy = game_save_folder / f"{game}_DQN_layer_RSA_{METHOD}_matrix.npy"
        np.save(rsa_npy, rsa_matrix)
        print(f"Saved RSA matrix: {rsa_npy}")

        # Plot RSA heatmap
        plt.figure(figsize=(6, 6))
        im = plt.imshow(rsa_matrix, cmap="viridis")
        plt.colorbar(im)
        plt.xticks(range(len(layer_names)), layer_names, rotation=45)
        plt.yticks(range(len(layer_names)), layer_names)
        for i in range(rsa_matrix.shape[0]):
            for j in range(rsa_matrix.shape[1]):
                plt.text(j, i, f"{rsa_matrix[i, j]:.2f}",
                         ha="center", va="center", color="white", fontsize=8)
        plt.title(f"{game} - RSA Between DQN Layers")
        plt.tight_layout()
        plt.savefig(game_save_folder / f"{game}_DQN_layer_{METHOD}_RSA_heatmap.png", dpi=300)
        plt.close()

        print(f"Clips used: {len(clip_names)}")

    print(f"\n[{seed}] All games done.")

print("\nAll DQN RDMs and RSA matrices computed successfully.")
