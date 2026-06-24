# src/config.py

from pathlib import Path
import sys
import torch

# =========================================================
# PROJECT ROOT
# =========================================================
ROOT = Path(__file__).resolve().parents[1]

# =========================================================
# GAMES & SEEDS
# =========================================================
GAMES = ["pong", "pacman", "spaceinvaders"]

SEEDS = ["seed_0", "seed_1", "seed_2", "seed_3", "seed_42"]

REFERENCE_SEED = "seed_42"

GAME_TO_GYM_ID = {
    "pong":          "PongNoFrameskip-v4",
    "pacman":        "MsPacmanNoFrameskip-v4",
    "spaceinvaders": "SpaceInvadersNoFrameskip-v4",
}

GYM_ID_TO_GAME = {v: k for k, v in GAME_TO_GYM_ID.items()}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# DQN HYPERPARAMETERS
# =========================================================
DQN = {
    "total_timesteps":        25_000_000,
    "checkpoint_freq":        500_000,
    "batch_size":             32,
    "buffer_size":            100_000,
    "learning_starts":        50_000,
    "learning_rate":          1e-4,
    "gamma":                  0.99,
    "train_freq":             4,
    "gradient_steps":         4,
    "target_update_interval": 10_000,
    "exploration_fraction":   0.1,
    "exploration_final_eps":  0.1,
    "n_envs":                 4,
    "features_dim":           512,
}

# =========================================================
# REPRESENTATION PIPELINE
# =========================================================
REPR = {
    "n_pca_components": 100,
    "rdm_method":       "correlation",
    "frame_stack":      4,
    "selected_frames":  [3, 7, 11, 15],
    "frame_size":       (84, 84),
}

# =========================================================
# TRIPLET / t-STE PARAMETERS
# =========================================================
TSTE = {
    "dim":               2,
    "n_repeats":         100,
    "max_iter":          1000,
    "easy_percent":      0.2,
    "hard_percent":      0.2,
    "medium_low":        0.4,
    "medium_high":       0.6,
    "structure_pctile":  20,
    "n_clips":           15,
}

# =========================================================
# BASE DIRECTORIES
# =========================================================
DATA    = ROOT / "data"
MODELS  = ROOT / "models"
RESULTS = ROOT / "results"

_lib = ROOT / "lib"
if str(_lib) not in sys.path:
    sys.path.insert(0, str(_lib))

# =========================================================
# PATHS
# =========================================================
PATHS = {

    # ── Raw gameplay recordings ──────────────────────────
    "recordings_root":              DATA / "recordings",
    "recordings_pca":               DATA / "recordings" / "pca",

    # ── Clips (mp4) ──────────────────────────────────────
    "clips_root":                   DATA / "clips",
    "clips_pca":                    DATA / "clips" / "pca",
    "clips_pool25_game":            DATA / "clips" / "{game}" / "pool25",
    "clips_subset15_game":          DATA / "clips" / "{game}" / "subset15",

    # ── Frame arrays (.npy) ──────────────────────────────
    "arrays_pca_game":              DATA / "arrays" / "pca" / "{game}",
    "arrays_pool25_game":           DATA / "arrays" / "pool25" / "{game}",
    "arrays_subset15_game":         DATA / "arrays" / "subset15" / "{game}",
    "arrays_bigset_game":           DATA / "arrays" / "bigset" / "{game}",

    # ── DQN layer activations (.npz) ─────────────────────
    "activations_pool25_seed":      DATA / "activations" / "pool25" / "{seed}",
    "activations_pca_seed":         DATA / "activations" / "pca" / "{seed}",
    "activations_subset15_seed":    DATA / "activations" / "subset15" / "{seed}",
    "activations_pca_multi_seed":   DATA / "activations" / "pca_multi" / "{seed}",

    # ── PCA models ───────────────────────────────────────
    "models_pca_layer":             MODELS / "pca_layer" / "{game}" / "{seed}",
    "models_pca_pixel":             MODELS / "pca_pixel" / "{game}" / "{seed}",

    # ── Trained DQN models ───────────────────────────────
    "models_dqn":                   MODELS / "{gym_id}" / "{seed}" / "final_model",

    # ── DQN pixel/state/q-value outputs (.npz) ───────────
    "states_pca_game":              DATA / "states" / "{seed}" / "pca" / "{game}",
    "states_subset15_game":         DATA / "states" / "{seed}" / "subset15" / "{game}",
    "states_rdms_game":             DATA / "states" / "{seed}" / "subset15" / "{game}" / "rdms",

    # ── RDMs (.npy) ──────────────────────────────────────
    "rdms_pilot":                   DATA / "rdms" / "pilot" / "{seed}" / "{game}",
    "rdms_subset15":                DATA / "rdms" / "subset15" / "{seed}" / "{game}",
    "rdms_bigset":                  DATA / "rdms" / "bigset" / "{seed}" / "{game}",
    "rdms_hcf":                     DATA / "rdms" / "bigset" / "pong" / "hcf" / "pong_hcf_rdm.npy",

    # ── Clip index maps (CSV) ─────────────────────────────
    "maps_subset15_game":           DATA / "maps" / "subset15" / "{game}_clip_map.csv",
    "maps_pool25_game":             DATA / "maps" / "pool25" / "{game}_clip_map.csv",

    # ── SA subset selection ───────────────────────────────
    "subsets_seed":                 DATA / "subsets" / "{seed}",
    "subsets_csv":                  DATA / "subsets" / "{seed}" / "{game}_best_subset_indices.csv",

    # ── Triplet pools and scores ──────────────────────────
    "triplets_scores_dir":          DATA / "triplets" / "scores",
    "triplets_scores_csv":          DATA / "triplets" / "scores" / "triplet_scores_{game}.csv",
    "triplets_tste_results":        DATA / "triplets" / "tste_reconstruction" / "{seed}",
    "triplets_viz":                 DATA / "triplets" / "viz" / "{seed}" / "difficulties",

    # ── Experiment data ───────────────────────────────────
    "experiment_individual_raw":    DATA / "experiment" / "individual",
    "experiment_individual":        DATA / "experiment" / "individual" / "cleaned_results",
    "experiment_sparse_raw":        DATA / "experiment" / "sparse" / "cleaned_results",
    "experiment_sparse":            DATA / "experiment" / "sparse" / "cleaned_results",
    "experiment_pilot":             DATA / "experiment" / "pilot" / "cleaned_results",
    "experiment_extra":             DATA / "experiment" / "extra",
    "experiment_sparse_individual_subset":     DATA / "experiment" / "sparse" / "cleaned_results" / "sparse_individual_subset",
    "experiment_individual_accuracy":   DATA / "experiment" / "individual" / "accuracy_results",

    # ── Human RDMs ───────────────────────────────────────
    "rdms_human_dir":               DATA / "experiment" / "deployment" / "human_rdms",
    "rdms_human_game":              DATA / "experiment" / "deployment" / "human_rdms" / "{game}_rdm.npy",

    # ── Analysis results ─────────────────────────────────
    "results_noise_ceiling":        DATA / "results" / "noise_ceiling",
    "results_cross_seed_rsa":       DATA / "results" / "cross_seed_rsa",
    "results_rsa_optimized":        DATA / "results" / "rsa_optimized" / "{seed}",
    "results_rsa_full":             DATA / "results" / "rsa_full" / "{seed}",
    "results_human_agreement":      DATA / "results" / "human_agreement",
    "results_agreement_bins":       DATA / "results" / "agreement_bins",
    "results_cv_pilot":             DATA / "experiment" / "pilot" / "cleaned_results" / "tste_cv_results",
    "results_cv_deployment":        DATA / "experiment" / "deployment" / "cleaned_results" / "tste_cv_results",

    # ── JSON data files ───────────────────────────────────
    "jsons_dir":                    DATA / "jsons",
    "jsons_pong_triplets":          DATA / "jsons" / "pong_final_triplet_exp.json",
    "jsons_common_games":           DATA / "jsons" / "common_games_data.json",
    "jsons_hcf_features":           DATA / "jsons" / "bigset" / "pong_hcf_features.json",
}


def get_path(key: str, **kwargs) -> Path:
    """
    Resolve a path template from PATHS, substituting
    any {placeholders} with keyword arguments.

    Example:
        get_path("rdms_subset15", seed="seed_42", game="pong")
        get_path("models_dqn", gym_id="PongNoFrameskip-v4", seed="seed_42")
    """
    template = PATHS[key]
    resolved = Path(str(template).format(**kwargs))
    return resolved


def ensure(key: str, **kwargs) -> Path:
    """
    Like get_path() but also creates the directory if it
    does not exist. Use for output paths.
    """
    p = get_path(key, **kwargs)
    p.mkdir(parents=True, exist_ok=True)
    return p
