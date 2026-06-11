"""
Multi-seed DQN representation pipeline
Steps per seed:
  6a → save activations from PCA-training clips  (large dataset)
  6b → save activations from RDM clips           (15 optimised clips)
  7_1 → train PCA on 6a activations
  7   → compute RDMs from 6b activations + saved PCA

Frames are already extracted — no frame extraction step here.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

script_dir = Path(__file__).resolve().parents[1]

import os
import re
import numpy as np
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import ale_py
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm, compare, concat
from tqdm import tqdm

from src.models.custom_dqn import CustomCNN
from src.wrappers.environment_wrappers import (
    RestrictPongActions,
    RestrictMsPacmanActions,
    RestrictSpaceInvadorsActions,
)

# =========================================================
# GLOBAL CONFIG
# =========================================================
SEEDS  = ["seed_0", "seed_1", "seed_2", "seed_3", "seed_42"]
GAMES  = ["pong", "pacman", "spaceinvaders"]

# ── Frame array folders (already extracted, shared across seeds) ──
# Used for PCA training (large dataset)
PCA_FRAMES_FOLDER   = "data/test_16_arrays/pca_training"
# Used for RDM computation (15 optimised clips)
RDM_FRAMES_FOLDER   = "data/test_16_arrays/selected_subset_15" #"data/test_16_arrays/big_rdm_equal_size" #"data/test_16_arrays/selected_subset_15"

# ── Model paths  (seed substituted at runtime) ──
def model_path(seed, gym_id):
    return f"models/{gym_id}/{seed}/final_model"

# ── Activation output folders ──
PCA_ACTIVATIONS_ROOT = "data/multi_seed/activations/pca_training"
RDM_ACTIVATIONS_ROOT = "data/multi_seed/activations/selected_subset_15"

# ── PCA and RDM outputs ──
PCA_MODELS_ROOT = "models/pca_models/multi_seed"
RDMS_ROOT       = "data/multi_seed/rdms"

GAME_TO_ID = {
    "pacman":        "MsPacmanNoFrameskip-v4",
    "pong":          "PongNoFrameskip-v4",
    "spaceinvaders": "SpaceInvadersNoFrameskip-v4",
}

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
N_PCA      = 100
RDM_METHOD = "correlation"


# =========================================================
# SHARED HELPERS
# =========================================================
def extract_layer_name(key):
    match = re.search(r"(conv\d+|fc)$", key)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot extract layer name from key: {key}")


def make_env(gym_id):
    env = gym.make(gym_id)
    env = AtariWrapper(env)
    if gym_id.startswith("Pong"):
        env = RestrictPongActions(env)
    elif gym_id.startswith("MsPacman"):
        env = RestrictMsPacmanActions(env)
    elif gym_id.startswith("SpaceInvaders"):
        env = RestrictSpaceInvadorsActions(env)
    else:
        raise ValueError(f"No action wrapper for {gym_id}")
    return Monitor(env)


def dqn_preprocess_from_16_frames(frames_16):
    assert frames_16.shape[0] == 16, "Expected 16 frames"
    processed = []
    for t in [3, 7, 11, 15]:
        pooled  = np.maximum(frames_16[t], frames_16[t - 1])
        gray    = cv2.cvtColor(pooled, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        processed.append(resized)
    return np.stack(processed, axis=0).astype(np.float32) / 255.0


def register_hooks(model):
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            # if output.ndim == 4:
            #     output = output.mean(dim=(2, 3))
            flat = output.detach().cpu().numpy().reshape(output.shape[0], -1)
            if name not in activations:
                activations[name] = []
            activations[name].append(flat)
        return hook

    conv_counter = 1

    for layer in model.policy.q_net.features_extractor.cnn:
        if isinstance(layer, nn.ReLU): #Ahora se calculan despues de las relus
            layer.register_forward_hook(
                get_activation(f"conv{conv_counter}")
            )
            conv_counter += 1

    for layer in model.policy.q_net.features_extractor.fc:
        if isinstance(layer, nn.ReLU): #Ahora se calculan despues de las relus (transformacion no lineal)
            layer.register_forward_hook(
                get_activation("fc")
            )

    return activations

def save_activations_for_dataset(seed, frames_root, output_root, label):
    """
    Forward-pass all clips in `frames_root` through the seed's DQN
    and save activations to `output_root/{seed}/`.

    `label` is just for print messages ("PCA-training" or "RDM").
    """
    print(f"\n  [{seed}] Saving activations — {label} dataset")

    output_folder = os.path.join(output_root, seed)
    os.makedirs(output_folder, exist_ok=True)

    for game in GAMES:
        gym_id        = GAME_TO_ID[game]
        frames_folder = os.path.join(frames_root, game)

        clip_files = [f for f in os.listdir(frames_folder) if f.endswith(".npy")]
        print(f"    {game}: {len(clip_files)} clips")

        # Build env + model once per game
        env = make_env(gym_id)
        env = VecFrameStack(DummyVecEnv([lambda: env]), n_stack=4)

        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=512),
        )
        dqn = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs,
                  buffer_size=1, learning_starts=0)
        dqn.set_parameters(model_path(seed, gym_id), exact_match=True)
        dqn.policy.to(DEVICE)
        dqn.policy.eval()

        activations = register_hooks(dqn)

        for clip_file in clip_files:
            for k in activations:
                activations[k] = []

            clip_name    = os.path.splitext(clip_file)[0]
            frames_array = np.load(os.path.join(frames_folder, clip_file))

            stack        = dqn_preprocess_from_16_frames(frames_array)
            stack_tensor = torch.tensor(
                stack[np.newaxis], dtype=torch.float32
            ).to(DEVICE)

            with torch.no_grad():
                _ = dqn.policy.q_net.features_extractor(stack_tensor)

            final_activations = {
                f"{clip_name}_{layer_name}": np.concatenate(acts, axis=0)
                for layer_name, acts in activations.items()
            }

            np.savez_compressed(
                os.path.join(output_folder, f"{clip_name}_activations.npz"),
                **final_activations,
            )

        print(f"    Saved → {output_folder}")


# =========================================================
# STEP 6 — Activations for both datasets
# =========================================================
def run_step6(seed):
    print(f"\n{'='*70}")
    print(f"STEP 6 — ACTIVATIONS  |  {seed}")
    print(f"{'='*70}")
    save_activations_for_dataset(seed, PCA_FRAMES_FOLDER,
                                 PCA_ACTIVATIONS_ROOT, "PCA-training")
    save_activations_for_dataset(seed, RDM_FRAMES_FOLDER,
                                 RDM_ACTIVATIONS_ROOT, "RDM")


# =========================================================
# STEP 7_1 — Train PCA (on PCA-training activations)
# =========================================================
def run_step7_1_pca(seed):
    print(f"\n{'='*70}")
    print(f"STEP 7_1 — PCA TRAINING  |  {seed}")
    print(f"{'='*70}")

    activations_folder = os.path.join(PCA_ACTIVATIONS_ROOT, seed)

    for game in GAMES:
        print(f"\n  [{seed}] {game}")

        activation_files = [
            f for f in os.listdir(activations_folder)
            if f.endswith("_activations.npz") and game in f.lower()
        ]
        if not activation_files:
            print(f"  No activation files found for {game}, skipping.")
            continue

        print(f"  {len(activation_files)} activation files found")

        layer_activations = {}
        for fname in tqdm(activation_files, desc=f"  Loading {game}"):
            data = np.load(os.path.join(activations_folder, fname))
            for key in data.files:
                layer = extract_layer_name(key)
                layer_activations.setdefault(layer, []).append(data[key])

        for layer_name in sorted(layer_activations.keys()):
            X = np.concatenate(layer_activations[layer_name], axis=0)

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            n_comp = min(N_PCA, X.shape[0], X.shape[1])
            pca    = PCA(n_components=n_comp, svd_solver="full")
            pca.fit(X)

            save_dir = os.path.join(PCA_MODELS_ROOT, game, seed)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{game}_{layer_name}_pca.pkl")
            joblib.dump({"pca": pca, "scaler": scaler}, save_path)

            print(f"  [{layer_name}] {n_comp} components, "
                  f"var={np.sum(pca.explained_variance_ratio_):.3f} → {save_path}")


# =========================================================
# STEP 7 — Compute RDMs (on RDM activations + saved PCA)
# =========================================================
def run_step7_rdms(seed):
    print(f"\n{'='*70}")
    print(f"STEP 7 — RDM COMPUTATION  |  {seed}")
    print(f"{'='*70}")

    activations_folder = os.path.join(RDM_ACTIVATIONS_ROOT, seed)

    for game in GAMES:
        print(f"\n  [{seed}] {game}")

        game_save_folder = os.path.join(RDMS_ROOT, seed, game)
        os.makedirs(game_save_folder, exist_ok=True)

        activation_files = [
            f for f in os.listdir(activations_folder)
            if f.endswith("_activations.npz") and game in f.lower()
        ]
        if not activation_files:
            print(f"  No activation files found for {game}, skipping.")
            continue

        layer_activations = {}
        clip_names        = []

        for fname in activation_files:
            clip_names.append(fname.replace("_activations.npz", ""))
            data = np.load(os.path.join(activations_folder, fname))
            for key in data.files:
                layer = extract_layer_name(key)
                layer_activations.setdefault(layer, []).append(data[key])

        rdm_objects = []
        layer_names = sorted(layer_activations.keys())

        for layer_name in layer_names:
            acts = np.concatenate(
                layer_activations[layer_name], axis=0
            ).astype(np.float32)

            # Load PCA trained on the large pca_training dataset
            pca_path = os.path.join(
                PCA_MODELS_ROOT, game, seed,
                f"{game}_{layer_name}_pca.pkl"
            )
            if not os.path.exists(pca_path):
                raise FileNotFoundError(f"Missing PCA model: {pca_path}")

            pca_data = joblib.load(pca_path)
            if pca_data["scaler"] is not None:
                acts = pca_data["scaler"].transform(acts)
            acts = pca_data["pca"].transform(acts)

            dataset = Dataset(
                acts,
                obs_descriptors={"clips": np.array(clip_names)},
                channel_descriptors={"units": np.arange(acts.shape[1])},
            )

            rdm_obj    = calc_rdm(dataset, method=RDM_METHOD)
            rdm_matrix = rdm_obj.get_matrices()[0]
            rdm_objects.append(rdm_obj)

            # Save RDM
            npy_path = os.path.join(
                game_save_folder,
                f"{game}_{layer_name}_{RDM_METHOD}_RDM.npy"
            )
            np.save(npy_path, rdm_matrix)

            # Plot RDM
            plt.figure(figsize=(8, 8))
            im = plt.imshow(rdm_matrix, cmap="coolwarm")
            plt.colorbar(im)
            plt.title(f"{seed} | {game} | {layer_name}")
            plt.tight_layout()
            plt.savefig(npy_path.replace(".npy", ".png"), dpi=150)
            plt.close()

            print(f"  Saved RDM [{layer_name}] → {npy_path}")

        # Layer-to-layer RSA
        combined   = concat(rdm_objects)
        rsa_matrix = compare(combined, combined, method="spearman")

        rsa_path = os.path.join(
            game_save_folder,
            f"{game}_DQN_layer_RSA_{RDM_METHOD}_matrix.npy"
        )
        np.save(rsa_path, rsa_matrix)

        plt.figure(figsize=(6, 6))
        im = plt.imshow(rsa_matrix, cmap="viridis")
        plt.colorbar(im)
        plt.xticks(range(len(layer_names)), layer_names, rotation=45)
        plt.yticks(range(len(layer_names)), layer_names)
        for i in range(rsa_matrix.shape[0]):
            for j in range(rsa_matrix.shape[1]):
                plt.text(j, i, f"{rsa_matrix[i,j]:.2f}",
                         ha="center", va="center", fontsize=8, color="white")
        plt.title(f"{seed} | {game} — RSA between DQN layers")
        plt.tight_layout()
        plt.savefig(rsa_path.replace(".npy", "_heatmap.png"), dpi=150)
        plt.close()

        print(f"  Layer RSA saved → {rsa_path}")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    for seed in SEEDS:
        print(f"\n{'#'*70}")
        print(f"  SEED: {seed}")
        print(f"{'#'*70}")
        run_step6(seed)
        run_step7_1_pca(seed)
        run_step7_rdms(seed)

    print("\n\nAll seeds processed successfully.")
