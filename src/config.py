# src/config.py
"""
Central configuration for the TFG DQN-Human project.
All scripts import from here — never hardcode paths or
shared constants in individual scripts.
"""

from pathlib import Path

# =========================================================
# PROJECT ROOT
# Resolves to the project root regardless of where
# you import this from, as long as the package is
# installed with `pip install -e .`
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

# =========================================================
# DQN HYPERPARAMETERS
# (shared across training scripts and multi_seed_pipeline)
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
    "selected_frames":  [3, 7, 11, 15],   # indices within 16-frame clip
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
    "structure_pctile":  20,    # bottom X% discarded by structure filter
    "n_clips":           15,
}

# =========================================================
# PATHS
# All paths derived from ROOT so they work on any machine.
# =========================================================
DATA        = ROOT / "data"
MODELS      = ROOT / "models"
RESULTS     = ROOT / "results"

PATHS = {
    # ── Raw gameplay recordings ──────────────────────────
    "human_plays":          DATA / "human_plays",
    "pca_training_plays":   DATA / "human_plays" / "pca_training",

    # ── Clips (video) ────────────────────────────────────
    "clips_root":           DATA / "test_16_clips",
    "clips_buenos25":       DATA / "test_16_clips" / "{game}" / "buenos_25" / "human_dqn_visualitzation",
    "clips_selected15":     DATA / "test_16_clips" / "{game}" / "selected_15",

    # ── Frame arrays (.npy) ───────────────────────────────
    "arrays_buenos25":      DATA / "test_16_arrays" / "buenos_25" / "{game}",
    "arrays_selected15":    DATA / "test_16_arrays" / "selected_subset_15" / "{game}",
    "arrays_pca_training":  DATA / "test_16_arrays" / "pca_training" / "{game}",

    # ── Activations ───────────────────────────────────────
    "activations_buenos25": DATA / "test_16_PRUEBAS" / "buenos_25" / "{seed}",
    "activations_selected": DATA / "multi_seed" / "activations" / "selected_subset_15" / "{seed}",
    "activations_pca":      DATA / "multi_seed" / "activations" / "pca_training" / "{seed}",

    # ── PCA models ────────────────────────────────────────
    "pca_layer_models":     MODELS / "pca_models" / "multi_seed" / "{game}" / "{seed}",
    "pca_pixel_models":     MODELS / "pca_models" / "pixel_pca_models" / "{game}" / "{seed}",

    # ── Trained DQN models ────────────────────────────────
    "dqn_model":            MODELS / "{gym_id}" / "{seed}" / "final_model",

    # ── RDMs ──────────────────────────────────────────────
    "rdms_selected15":      DATA / "test_16_rdms" / "selected_subset_15" / "{seed}" / "{game}",
    "rdms_big":             DATA / "test_16_rdms" / "big_rdm_equal_size" / "{seed}" / "{game}",
    "rdm_hcf":              DATA / "test_16_rdms" / "big_rdm_equal_size" / "pong" / "hcf" / "pong_hcf_rdm.npy",

    # ── Clip index maps ───────────────────────────────────
    "clip_maps":            DATA / "maps" / "selected_15" / "{game}_clip_map.csv",

    # ── Subset selection (SA output) ─────────────────────
    "subset_selection":     DATA / "subset_selection" / "{seed}",
    "subset_csv":           DATA / "subset_selection" / "{seed}" / "{game}_best_subset_indices.csv",

    # ── Triplet pools ─────────────────────────────────────
    "triplet_scores":       DATA / "triplets_results" / "triplet_scores",
    "triplet_score_csv":    DATA / "triplets_results" / "triplet_scores" / "triplet_scores_{game}.csv",

    # ── Experiment results ────────────────────────────────
    "exp_raw":              DATA / "triplets_results" / "final_experiment",
    "exp_cleaned":          DATA / "triplets_results" / "final_experiment" / "cleaned_results",
    "exp2_cleaned":         DATA / "triplets_results" / "exp2" / "cleaned_results",

    # ── Human RDMs ────────────────────────────────────────
    "human_rdms":           DATA / "triplets_results" / "final_experiment" / "cleaned_results" / "rdms_human_experiment_rsa",
    "human_rdm_file":       DATA / "triplets_results" / "final_experiment" / "cleaned_results" / "rdms_human_experiment_rsa" / "{game}_rdm.npy",

    # ── Analysis outputs ──────────────────────────────────
    "noise_ceiling":        DATA / "triplets_results" / "noise_ceiling",
    "cross_seed_rsa":       DATA / "multi_seed" / "cross_seed_rsa" / "big_rdm_equal_size",
    "rsa_results":          DATA / "triplets_results" / "final_experiment" / "cleaned_results" / "test_16_RSA" / "optimized_RDMs" / "{seed}",
    "full_rsa":             DATA / "dqn_state_action_qvalue" / "RSA" / "{seed}" / "selected_subset_15",

    # ── JSONs ─────────────────────────────────────────────
    "jsons":                DATA / "jsons",
    "pong_triplets_json":   DATA / "jsons" / "pong_final_triplet_exp.json",
    "common_games_json":    DATA / "jsons" / "common_games_data.json",
    "hcf_features_json":    DATA / "jsons" / "big_rdm_equal_size" / "pong_hcf_features.json",
}

def get_path(key: str, **kwargs) -> Path:
    """
    Resolve a path template from PATHS, substituting
    any {placeholders} with keyword arguments.

    Example:
        get_path("rdms_selected15", seed="seed_42", game="pong")
        get_path("dqn_model", gym_id="PongNoFrameskip-v4", seed="seed_42")
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