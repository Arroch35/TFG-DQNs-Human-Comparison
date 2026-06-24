# src/config.py
"""
Central configuration for the TFG DQN-Human project.
All scripts import from here — never hardcode paths or
shared constants in individual scripts.

FOLDER RENAME GUIDE
───────────────────
Run the following shell commands from the project root to migrate
your existing data directory to the new structure.

  # Recordings
  mv data/human_plays                                   data/recordings
  mv data/recordings/pca_training                       data/recordings/pca

  # Clips
  mv data/test_16_clips                                 data/clips
  # (repeat per game: pong, pacman, spaceinvaders)
  mv data/clips/{game}/buenos_25/human_dqn_visualitzation  data/clips/{game}/pool25
  mv data/clips/{game}/selected_15                      data/clips/{game}/subset15

  # Frame arrays
  mv data/test_16_arrays                                data/arrays
  mv data/arrays/pca_training                           data/arrays/pca
  mv data/arrays/buenos_25                              data/arrays/pool25
  mv data/arrays/selected_subset_15                     data/arrays/subset15
  mv data/arrays/big_rdm_equal_size                     data/arrays/bigset

  # Activations
  mv data/test_16_PRUEBAS/buenos_25                     data/activations/pool25
  mv data/test_16_PRUEBAS/pca_training                  data/activations/pca
  mv data/multi_seed/activations/selected_subset_15     data/activations/subset15
  mv data/multi_seed/activations/pca_training           data/activations/pca_multi

  # Models (in models/ not data/)
  mv models/pca_models/multi_seed                       models/pca_layer
  mv models/pca_models/pixel_pca_models                 models/pca_pixel

  # States
  mv data/dqn_state_action_qvalue                       data/states
  # (repeat per seed)
  mv data/states/{seed}/pca_training_set                data/states/{seed}/pca
  mv data/states/{seed}/selected_subset_15              data/states/{seed}/subset15
  mv data/states/RSA                                    data/results/rsa_full

  # RDMs
  mv data/test_16_rdms                                  data/rdms
  mv data/rdms/selected_subset_15                       data/rdms/subset15
  mv data/rdms/big_rdm_equal_size                       data/rdms/bigset

  # Maps
  mv data/maps/selected_15                              data/maps/subset15
  mv data/maps/buenos_25                                data/maps/pool25

  # Subsets
  mv data/subset_selection                              data/subsets

  # Triplets
  mkdir -p data/triplets
  mv data/triplets_results/triplet_scores               data/triplets/scores
  mv data/triplet_experiment_results/selected_subset_15 data/triplets/tste_reconstruction
  mv data/triplet_visualization_subset/selected_15      data/triplets/viz
  # (inside viz, rename per seed)
  mv data/triplets/viz/{seed}/filtered_all_difficulties data/triplets/viz/{seed}/difficulties

  # Experiment data
  mkdir -p data/experiment
  mv data/triplets_results/final_experiment             data/experiment/deployment
  mv data/triplets_results/exp2                         data/experiment/sparse
  mv data/triplets_results/own_data                     data/experiment/pilot
  mv data/extra                                         data/experiment/extra
  mv data/experiment/deployment/cleaned_results/rdms_human_experiment_rsa  data/experiment/deployment/human_rdms
  mv data/experiment/deployment/cleaned_results/test_16_RSA/optimized_RDMs data/results/rsa_optimized

  # Results
  mkdir -p data/results
  mv data/triplets_results/noise_ceiling                data/results/noise_ceiling
  mv data/multi_seed/cross_seed_rsa/big_rdm_equal_size  data/results/cross_seed_rsa
  mv data/triplets_results/human_agreement_by_difficulty data/results/human_agreement
  mv data/triplets_results/agreement_vs_difficulty      data/results/agreement_bins
  # tste_cv_results folders live inside experiment/pilot and experiment/deployment
  # — no mv needed, they will be created fresh on next run.

  # JSONs
  mv data/jsons/big_rdm_equal_size                      data/jsons/bigset

PATH KEY NAMING CONVENTION
──────────────────────────
Keys follow the pattern:  {domain}_{variant}_{qualifier}

  domain     — artifact kind:
                recordings  raw gameplay frame recordings
                clips       mp4 clip files
                arrays      .npy frame arrays
                activations DQN layer activation .npz files
                rdms        RDM .npy files
                states      pixel/state/q-value .npz files
                maps        clip index ↔ filename CSV maps
                subsets     SA-selected subset CSVs
                triplets    triplet pool CSVs and scores
                experiment  raw/cleaned participant data
                results     analysis output folders
                jsons       JSON data files
                models      trained model files

  variant    — dataset slice:
                pca         PCA training set
                pool25      pool of 25 candidate clips
                subset15    final 15-clip experiment subset
                bigset      big RDM equal-size set
                pilot       pilot experiment
                sparse      sparse second experiment run
                deployment  full deployment experiment

  qualifier  — extra specificity (omit when unambiguous):
                game        path contains {game} placeholder
                seed        path contains {seed} placeholder
                hcf         hand-crafted features (pong only)
                human       human participant RDMs
                layer       DQN-layer PCA
                pixel       pixel-space PCA
                cv          cross-validation outputs
"""

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
    # old: data/human_plays
    "recordings_root":              DATA / "recordings",
    # old: data/human_plays/pca_training
    "recordings_pca":               DATA / "recordings" / "pca",

    # ── Clips (mp4) ──────────────────────────────────────
    # old: data/test_16_clips
    "clips_root":                   DATA / "clips",
    # old: data/test_16_clips/pca_training
    "clips_pca":                    DATA / "clips" / "pca",
    # old: data/test_16_clips/{game}/buenos_25/human_dqn_visualitzation
    "clips_pool25_game":            DATA / "clips" / "{game}" / "pool25",
    # old: data/test_16_clips/{game}/selected_15
    "clips_subset15_game":          DATA / "clips" / "{game}" / "subset15",

    # ── Frame arrays (.npy) ──────────────────────────────
    # old: data/test_16_arrays/pca_training/{game}
    "arrays_pca_game":              DATA / "arrays" / "pca" / "{game}",
    # old: data/test_16_arrays/buenos_25/{game}
    "arrays_pool25_game":           DATA / "arrays" / "pool25" / "{game}",
    # old: data/test_16_arrays/selected_subset_15/{game}
    "arrays_subset15_game":         DATA / "arrays" / "subset15" / "{game}",
    # old: data/test_16_arrays/big_rdm_equal_size/{game}
    "arrays_bigset_game":           DATA / "arrays" / "bigset" / "{game}",

    # ── DQN layer activations (.npz) ─────────────────────
    # old: data/test_16_PRUEBAS/buenos_25/{seed}
    "activations_pool25_seed":      DATA / "activations" / "pool25" / "{seed}",
    # old: data/test_16_PRUEBAS/pca_training/{seed}
    "activations_pca_seed":         DATA / "activations" / "pca" / "{seed}",
    # old: data/multi_seed/activations/selected_subset_15/{seed}
    "activations_subset15_seed":    DATA / "activations" / "subset15" / "{seed}",
    # old: data/multi_seed/activations/pca_training/{seed}
    "activations_pca_multi_seed":   DATA / "activations" / "pca_multi" / "{seed}",

    # ── PCA models ───────────────────────────────────────
    # old: models/pca_models/multi_seed/{game}/{seed}
    "models_pca_layer":             MODELS / "pca_layer" / "{game}" / "{seed}",
    # old: models/pca_models/pixel_pca_models/{game}/{seed}
    "models_pca_pixel":             MODELS / "pca_pixel" / "{game}" / "{seed}",

    # ── Trained DQN models ───────────────────────────────
    # old: models/{gym_id}/{seed}/final_model
    "models_dqn":                   MODELS / "{gym_id}" / "{seed}" / "final_model",

    # ── DQN pixel/state/q-value outputs (.npz) ───────────
    # old: data/dqn_state_action_qvalue/{seed}/pca_training_set/{game}
    "states_pca_game":              DATA / "states" / "{seed}" / "pca" / "{game}",
    # old: data/dqn_state_action_qvalue/{seed}/selected_subset_15/{game}
    "states_subset15_game":         DATA / "states" / "{seed}" / "subset15" / "{game}",
    # old: data/dqn_state_action_qvalue/{seed}/selected_subset_15/{game}/rdms
    "states_rdms_game":             DATA / "states" / "{seed}" / "subset15" / "{game}" / "rdms",

    # ── RDMs (.npy) ──────────────────────────────────────
    # old: data/test_16_rdms/pilot/{seed}/{game}
    "rdms_pilot":                   DATA / "rdms" / "pilot" / "{seed}" / "{game}",
    # old: data/test_16_rdms/selected_subset_15/{seed}/{game}
    "rdms_subset15":                DATA / "rdms" / "subset15" / "{seed}" / "{game}",
    # old: data/test_16_rdms/big_rdm_equal_size/{seed}/{game}
    "rdms_bigset":                  DATA / "rdms" / "bigset" / "{seed}" / "{game}",
    # old: data/test_16_rdms/big_rdm_equal_size/pong/hcf/pong_hcf_rdm.npy
    "rdms_hcf":                     DATA / "rdms" / "bigset" / "pong" / "hcf" / "pong_hcf_rdm.npy",

    # ── Clip index maps (CSV) ─────────────────────────────
    # old: data/maps/selected_15/{game}_clip_map.csv
    "maps_subset15_game":           DATA / "maps" / "subset15" / "{game}_clip_map.csv",
    # old: data/maps/buenos_25/{game}_clip_map.csv
    "maps_pool25_game":             DATA / "maps" / "pool25" / "{game}_clip_map.csv",

    # ── SA subset selection ───────────────────────────────
    # old: data/subset_selection/{seed}
    "subsets_seed":                 DATA / "subsets" / "{seed}",
    # old: data/subset_selection/{seed}/{game}_best_subset_indices.csv
    "subsets_csv":                  DATA / "subsets" / "{seed}" / "{game}_best_subset_indices.csv",

    # ── Triplet pools and scores ──────────────────────────
    # old: data/triplets_results/triplet_scores
    "triplets_scores_dir":          DATA / "triplets" / "scores",
    # old: data/triplets_results/triplet_scores/triplet_scores_{game}.csv
    "triplets_scores_csv":          DATA / "triplets" / "scores" / "triplet_scores_{game}.csv",
    # old: data/triplet_experiment_results/selected_subset_15/{seed}
    "triplets_tste_results":        DATA / "triplets" / "tste_reconstruction" / "{seed}",
    # old: data/triplet_visualization_subset/selected_15/{seed}/filtered_all_difficulties
    "triplets_viz":                 DATA / "triplets" / "viz" / "{seed}" / "difficulties",

    # ── Experiment data ───────────────────────────────────
    # old: data/triplets_results/final_experiment
    "experiment_individual_raw":    DATA / "experiment" / "individual",
    # old: data/triplets_results/final_experiment/cleaned_results
    "experiment_individual":        DATA / "experiment" / "individual" / "cleaned_results",
    # old: data/triplets_results/exp2/cleaned_results
    "experiment_sparse_raw":            DATA / "experiment" / "sparse" / "cleaned_results",
    # old: data/triplets_results/exp2/cleaned_results
    "experiment_sparse":            DATA / "experiment" / "sparse" / "cleaned_results",
    # old: data/triplets_results/own_data/cleaned_results
    "experiment_pilot":             DATA / "experiment" / "pilot" / "cleaned_results",
    # old: data/extra
    "experiment_extra":             DATA / "experiment" / "extra",
    # old: data/triplets_results/exp2/cleaned_results/sparse_individual_subset
    "experiment_sparse_individual_subset":     DATA / "experiment" / "sparse" / "cleaned_results" / "sparse_individual_subset",
    # old: data/triplets_results/exp2/accuracy_results
    "experiment_individual_accuracy":   DATA / "experiment" / "individual" / "accuracy_results",

    # ── Human RDMs ───────────────────────────────────────
    # old: data/triplets_results/final_experiment/cleaned_results/rdms_human_experiment_rsa
    "rdms_human_dir":               DATA / "experiment" / "deployment" / "human_rdms",
    # old: data/triplets_results/final_experiment/cleaned_results/rdms_human_experiment_rsa/{game}_rdm.npy
    "rdms_human_game":              DATA / "experiment" / "deployment" / "human_rdms" / "{game}_rdm.npy",

    # ── Analysis results ─────────────────────────────────
    # old: data/triplets_results/noise_ceiling
    "results_noise_ceiling":        DATA / "results" / "noise_ceiling",
    # old: data/multi_seed/cross_seed_rsa/big_rdm_equal_size
    "results_cross_seed_rsa":       DATA / "results" / "cross_seed_rsa",
    # old: data/triplets_results/final_experiment/cleaned_results/test_16_RSA/optimized_RDMs/{seed}
    "results_rsa_optimized":        DATA / "results" / "rsa_optimized" / "{seed}",
    # old: data/dqn_state_action_qvalue/RSA/{seed}/selected_subset_15
    "results_rsa_full":             DATA / "results" / "rsa_full" / "{seed}",
    # old: data/triplets_results/human_agreement_by_difficulty
    "results_human_agreement":      DATA / "results" / "human_agreement",
    # old: data/triplets_results/agreement_vs_difficulty
    "results_agreement_bins":       DATA / "results" / "agreement_bins",
    # old: data/triplets_results/own_data/cleaned_results/tste_cv_results
    "results_cv_pilot":             DATA / "experiment" / "pilot" / "cleaned_results" / "tste_cv_results",
    # old: data/triplets_results/final_experiment/cleaned_results/tste_cv_results
    "results_cv_deployment":        DATA / "experiment" / "deployment" / "cleaned_results" / "tste_cv_results",

    # ── JSON data files ───────────────────────────────────
    # old: data/jsons
    "jsons_dir":                    DATA / "jsons",
    # old: data/jsons/pong_final_triplet_exp.json
    "jsons_pong_triplets":          DATA / "jsons" / "pong_final_triplet_exp.json",
    # old: data/jsons/common_games_data.json
    "jsons_common_games":           DATA / "jsons" / "common_games_data.json",
    # old: data/jsons/big_rdm_equal_size/pong_hcf_features.json
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
