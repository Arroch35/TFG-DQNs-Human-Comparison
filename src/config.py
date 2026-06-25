# src/config.py
"""
Central configuration for the TFG DQN-Human project.

All scripts import constants and paths from here.
Never hardcode paths or shared constants in individual scripts.

DATASET SCOPES
--------------
Four dataset scopes exist. Several pipeline stages must be run once
per scope. See README.md — "Understanding the Data Scopes" for details.

  pca_training  ~900 clips/game  Training PCA models (layer activations
                                 and pixel states). Never used directly
                                 in experiments.

  pool25        25 clips/game    Manually selected candidate clips from
                                 which the final 15 are chosen by
                                 simulated annealing (Experiment 7).

  bigset        909 clips/game   Full reference dataset. Used to evaluate
                                 how representative any subset is
                                 (Experiments 5-6), for the HCF model,
                                 and for cross-seed RSA (Experiment 11).

  subset15      15 clips/game    Final optimized subset selected by SA
                                 from pool25. Used in all human
                                 experiments and main analyses.
                                 NOT a separate recording — it is a
                                 filtered view of pool25 clips, defined
                                 by subsets_csv after running
                                 03_clip_selection/.

SEED CONVENTION
---------------
Five seeds: seed_0, seed_1, seed_2, seed_3, seed_42.
REFERENCE_SEED (seed_42) is used for all stimulus selection,
triplet generation, and experiment design.
The other four seeds are used exclusively for reliability
analyses (Experiments 11-13).

Triplet generation (06_triplet_generation/) always uses
REFERENCE_SEED only — it does not loop over all seeds.
"""


from pathlib import Path
import sys
import os
import torch

# =========================================================
# PROJECT ROOT
# =========================================================
ROOT = Path(os.environ.get("TFG_ROOT", str(Path(__file__).resolve().parents[1])))

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
# buffer_size reduced from original Mnih et al. (2015) 1M to
# 100K due to computational constraints. Optimizer is Adam
# (not RMSProp as in original). See paper Section 4.2.3.
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
# Shared across activation extraction, PCA, and RDM scripts.
# =========================================================
REPR = {
    "n_pca_components": 100,        # components for layer activation PCA
    "rdm_method":       "correlation",  # distance metric used for all RDMs
    "frame_stack":      4,
    "selected_frames":  [3, 7, 11, 15], # indices within 16-frame clip (frame skip=4)
    "frame_size":       (84, 84),
}

# =========================================================
# TRIPLET / t-STE PARAMETERS
# Shared across all triplet generation and t-STE fitting scripts.
# Difficulty bucketing is applied to the 80% of triplets that
# pass the structure filter (bottom structure_pctile% discarded).
# =========================================================
TSTE = {
    "dim":               2,    # embedding dimensionality (best across all games)
    "n_repeats":         100,  # t-STE runs per condition (best of N kept)
    "max_iter":          1000, # optimization iterations per run
    "easy_percent":      0.2,  # top 20% by difficulty score → easy bucket
    "hard_percent":      0.2,  # bottom 20% → hard bucket
    "medium_low":        0.4,
    "medium_high":       0.6,
    "structure_pctile":  20,   # discard bottom 20% least-structured triplets
    "n_clips":           15,   # clips in subset15 (nodes in t-STE graph)
}

# =========================================================
# BASE DIRECTORIES
# =========================================================
DATA    = ROOT / "data"
MODELS  = ROOT / "models"
RESULTS = ROOT / "results"

# Register lib/ so cy_tste is importable from any script.
# Runs automatically when any script does: from src.config import ...
_lib = ROOT / "lib"
if str(_lib) not in sys.path:
    sys.path.insert(0, str(_lib))

# =========================================================
# PATHS
#
# Keys use {placeholders} for variable parts.
# Resolve with:   get_path(key, game="pong", seed="seed_42")
# Create folder:  ensure(key, game="pong", seed="seed_42")
#
# Scope suffixes in key names:
#   _pca      → pca_training scope (PCA fitting data, never in experiments)
#   _pool25   → pool25 scope (25 candidate clips per game)
#   _bigset   → bigset scope (909-clip reference dataset)
#   _subset15 → subset15 scope (final 15-clip experiment set)
# =========================================================
PATHS = {

    # ── Raw gameplay recordings (.npz chunks) ───────────
    # 02_stimulus_generation/1_record_human_gameplay.py writes here.
    # Run once per scope: pca_training, pool25, bigset.
    "recordings_root":              DATA / "recordings",
    "recordings_pca":               DATA / "recordings" / "pca",
    "recordings_pool25":            DATA / "recordings" / "pool25",
    "recordings_bigset":            DATA / "recordings" / "bigset",

    # ── Clips (.mp4) ────────────────────────────────────
    # 02_stimulus_generation/2_extract_clips_from_gameplay.py writes here.
    # subset15 clips are NOT separately recorded — they are a filtered
    # view of pool25, defined by subsets_csv after 03_clip_selection/.
    "clips_root":                   DATA / "clips",
    "clips_pca":                    DATA / "clips" / "pca",
    "clips_pool25_game":            DATA / "clips" / "{game}" / "pool25",
    "clips_bigset_game":            DATA / "clips" / "{game}" / "bigset",
    "clips_subset15_game":          DATA / "clips" / "{game}" / "subset15",

    # ── Frame arrays (.npy, shape 16×H×W×3) ─────────────
    # 04_dqn_representations/1_extract_frame_arrays.py writes here.
    # Run for pca_training, pool25, bigset. subset15 arrays are not
    # extracted separately — they are indexed from pool25 via subsets_csv.
    "arrays_pca_game":              DATA / "arrays" / "pca" / "{game}",
    "arrays_pool25_game":           DATA / "arrays" / "pool25" / "{game}",
    "arrays_bigset_game":           DATA / "arrays" / "bigset" / "{game}",
    "arrays_subset15_game":         DATA / "arrays" / "subset15" / "{game}",

    # ── DQN layer activations (.npz) ─────────────────────
    # 04_dqn_representations/2_extract_activations.py writes here.
    # pca: seed_42 only, used to fit PCA models.
    # pca_multi: all seeds, used by multi_seed_pipeline.py.
    # pool25: seed_42 only, used during clip selection (03/).
    # bigset: seed_42 only, intermediate step for bigset RDMs (Exp 5-6, 11).
    # subset15: all 5 seeds, used in main analyses (second pass of 04/).
    "activations_pca_seed":         DATA / "activations" / "pca" / "{seed}",
    "activations_pca_multi_seed":   DATA / "activations" / "pca_multi" / "{seed}",
    "activations_pool25_seed":      DATA / "activations" / "pool25" / "{seed}",
    "activations_bigset_seed":      DATA / "activations" / "bigset" / "{seed}",
    "activations_subset15_seed":    DATA / "activations" / "subset15" / "{seed}",

    # ── PCA models (.pkl) ────────────────────────────────
    # layer: fitted on pca_training activations, applied to pool25/
    #        subset15 activations before RDM computation (100 components).
    # pixel: fitted on pca_training pixel states, used for the pixel
    #        PCA theoretical model (Appendix A.5).
    "models_pca_layer":             MODELS / "pca_layer" / "{game}" / "{seed}",
    "models_pca_pixel":             MODELS / "pca_pixel" / "{game}" / "{seed}",

    # ── Trained DQN models ───────────────────────────────
    # Use GAME_TO_GYM_ID[game] to get gym_id from a game name.
    "models_dqn":                   MODELS / "{gym_id}" / "{seed}" / "final_model",

    # ── DQN pixel/state/q-value outputs (.npz) ───────────
    # 05_theoretical_models/ writes here.
    # states_pca_game: raw pixel vectors for pca_training set
    #   (used to fit pixel PCA models).
    # states_subset15_game: pixel vectors + q-values + state values
    #   for subset15 clips (used to compute theoretical model RDMs).
    # states_rdms_game: RDMs computed from the above vectors.
    #   Contains: pixel_rdm.npy, pixel_pca_rdm.npy,
    #             qvalue_rdm.npy, state_value_rdm.npy (Appendix A.10).
    "states_pca_game":              DATA / "states" / "{seed}" / "pca" / "{game}",
    "states_subset15_game":         DATA / "states" / "{seed}" / "subset15" / "{game}",
    "states_rdms_game":             DATA / "states" / "{seed}" / "subset15" / "{game}" / "rdms",

    # ── RDMs (.npy, correlation distance unless noted) ───
    # 04_dqn_representations/4_compute_dqn_rdms.py writes here.
    # pilot:    pilot experiment clips (Appendix A.4-A.5, seed_42 only).
    # pool25:   25-candidate clips (seed_42 only), read by 03_clip_selection/.
    # bigset:   909-clip reference (seed_42 only), used in Exp 5-6, 11.
    # subset15: final 15-clip experiment set (all 5 seeds), main analyses.
    # hcf:      handcrafted feature RDM for Pong only (euclidean distance),
    #           computed from bigset clips by 05_theoretical_models/.
    "rdms_pilot":                   DATA / "rdms" / "pilot" / "{seed}" / "{game}",
    "rdms_pool25":                  DATA / "rdms" / "pool25" / "{seed}" / "{game}",
    "rdms_bigset":                  DATA / "rdms" / "bigset" / "{seed}" / "{game}",
    "rdms_subset15":                DATA / "rdms" / "subset15" / "{seed}" / "{game}",
    "rdms_hcf":                     DATA / "rdms" / "bigset" / "pong" / "hcf" / "pong_hcf_rdm.npy",

    # ── Clip index maps (.csv) ───────────────────────────
    # Maps clip filename → integer index (0-based).
    # 06_triplet_generation/1_create_clip_index_map.py writes these.
    # subset15 map is used by all downstream experiment scripts.
    # bigset map is used by Exp 5-6 and cross-seed RSA (Exp 11).
    "maps_pool25_game":             DATA / "maps" / "pool25" / "{game}_clip_map.csv",
    "maps_bigset_game":             DATA / "maps" / "bigset" / "{game}_clip_map.csv",
    "maps_subset15_game":           DATA / "maps" / "subset15" / "{game}_clip_map.csv",

    # ── SA subset selection ──────────────────────────────
    # Output of 03_clip_selection/exp7_simulated_annealing.py.
    # Always written for REFERENCE_SEED only.
    # subsets_csv defines which pool25 clips form subset15.
    "subsets_seed":                 DATA / "subsets" / "{seed}",
    "subsets_csv":                  DATA / "subsets" / "{seed}" / "{game}_best_subset_indices.csv",

    # ── Triplet pools and scores ─────────────────────────
    # Always computed from REFERENCE_SEED (seed_42) only.
    # triplets_scores_csv: all 455 triplets per game with
    #   difficulty_score, structure_score, bucket label, and
    #   per-seed answer columns (output of 06_triplet_generation/).
    # triplets_tste_results: t-STE reconstruction quality by seed
    #   (output of 03_clip_selection/exp8_triplet_reconstruction.py).
    # triplets_viz: triplet video folders organised by difficulty,
    #   used when uploading stimuli to cognition.run.
    "triplets_scores_dir":          DATA / "triplets" / "scores",
    "triplets_scores_csv":          DATA / "triplets" / "scores" / "triplet_scores_{game}.csv",
    "triplets_tste_results":        DATA / "triplets" / "tste_reconstruction" / "{seed}",
    "triplets_viz":                 DATA / "triplets" / "viz" / "{seed}" / "difficulties",

    # ── Experiment data ──────────────────────────────────
    # Place downloaded cognition.run CSVs into the _raw folders
    # before running 07_human_experiments/1_clean_raw_experiment_csvs.py.
    # sparse: deployment experiment (~41 participants, 6 triplets/game each).
    # individual: Pong-only experiment (9 participants, 60 triplets each).
    # pilot: small pilot experiment (6 participants, Appendix A.4).
    # sparse_individual_subset: 60 Pong triplets shared between both
    #   experiments (output of 07/.../5_filter_sparse_to_pong60_subset.py).
    # individual_accuracy: output of 08_consistency_analyses/exp9.
    "experiment_sparse_raw":        DATA / "experiment" / "sparse",
    "experiment_sparse":            DATA / "experiment" / "sparse" / "cleaned_results",
    "experiment_individual_raw":    DATA / "experiment" / "individual",
    "experiment_individual":        DATA / "experiment" / "individual" / "cleaned_results",
    "experiment_pilot_raw":         DATA / "experiment" / "pilot" / "raw",
    "experiment_pilot":             DATA / "experiment" / "pilot" / "cleaned_results",
    "experiment_extra":             DATA / "experiment" / "extra",
    "experiment_sparse_individual_subset":   DATA / "experiment" / "sparse" / "cleaned_results" / "sparse_individual_subset",
    "experiment_individual_accuracy":        DATA / "experiment" / "individual" / "accuracy_results",

    # ── Human RDMs ───────────────────────────────────────
    # rdms_human_sparse_*: t-STE RDMs fitted on ALL sparse experiment
    #   responses pooled (one RDM per game, 2D Euclidean embedding).
    #   Written by 07_human_experiments/4_human_rdms_computation.py.
    # rdms_human_individual_dir: per-participant RDMs for the individual
    #   Pong experiment (one .npy per participant).
    #   Written by 07_human_experiments/3_individual_RDM_reconstructions.py.
    # rdms_human_dir / rdms_human_game: final human RDMs used by
    #   exp15_full_rsa_matrix.py. Copy or symlink from rdms_human_sparse_*
    #   once you have confirmed the deployment results.
    "rdms_human_sparse_dir":        DATA / "experiment" / "sparse" / "cleaned_results" / "rdms_human_experiment_rsa",
    "rdms_human_sparse_game":       DATA / "experiment" / "sparse" / "cleaned_results" / "rdms_human_experiment_rsa" / "{game}_rdm.npy",
    "rdms_human_individual_dir":    DATA / "experiment" / "individual" / "cleaned_results" / "rdms_human_experiment_rsa",
    "rdms_human_dir":               DATA / "experiment" / "deployment" / "human_rdms",
    "rdms_human_game":              DATA / "experiment" / "deployment" / "human_rdms" / "{game}_rdm.npy",

    # ── Analysis results ─────────────────────────────────
    # results_cross_seed_rsa: parent folder for Exp 11 outputs.
    # results_cross_seed_rsa_game: per-game subfolder where heatmaps
    #   and .npy matrices are saved (one subfolder per game).
    "results_noise_ceiling":        DATA / "results" / "noise_ceiling",
    "results_cross_seed_rsa":       DATA / "results" / "cross_seed_rsa",
    "results_cross_seed_rsa_game":  DATA / "results" / "cross_seed_rsa" / "{game}",
    "results_rsa_optimized":        DATA / "results" / "rsa_optimized" / "{seed}",
    "results_rsa_full":             DATA / "results" / "rsa_full" / "{seed}",
    "results_human_agreement":      DATA / "results" / "human_agreement",
    "results_agreement_bins":       DATA / "results" / "agreement_bins",
    "results_cv_pilot":             DATA / "experiment" / "pilot" / "cleaned_results" / "tste_cv_results",
    "results_cv_deployment":        DATA / "experiment" / "deployment" / "cleaned_results" / "tste_cv_results",

    # ── JSON configuration files ─────────────────────────
    # pong_final_triplet_exp.json: the 60 Pong triplets used in
    #   the individual experiment (produced by 07/.../select_pong60).
    # common_games_data.json: triplets seen in both experiments
    #   (produced by 07/.../0_extract_triplets.py).
    # pong_hcf_features.json: 6-D handcrafted feature vectors
    #   (ball pos/velocity, paddle positions) for each bigset Pong clip.
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
    if key not in PATHS:
        raise KeyError(
            f"Unknown path key: '{key}'. "
            f"Available keys: {sorted(PATHS.keys())}"
        )
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


def require(key: str, **kwargs) -> Path:
    """
    Like get_path() but raises FileNotFoundError immediately if
    the path does not exist. Use for input paths at the top of
    scripts to catch missing upstream data before computation starts.

    Example:
        rdm = np.load(
            require("rdms_subset15", seed=seed, game=game)
            / f"{game}_fc_correlation_RDM.npy"
        )
    """
    p = get_path(key, **kwargs)
    if not p.exists():
        raise FileNotFoundError(
            f"Required path does not exist: {p}\n"
            f"  config key : '{key}'\n"
            f"  kwargs     : {kwargs}\n"
            f"  Check that the upstream pipeline stage has been run."
        )
    return p
