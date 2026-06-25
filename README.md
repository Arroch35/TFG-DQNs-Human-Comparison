# Characterizing Human State Space Representations with Deep Reinforcement Learning Models

**Oscar Arrocha Gascón** — Final Degree Project in Artificial Intelligence  
Escola d'Enginyeria (EE), Universitat Autònoma de Barcelona (UAB)  
Academic Year 2025/26 · Supervisor: Daniel Pacheco (Basic Psychology Area)

---

## Overview

This project investigates the alignment between human perceptual representations of Atari game states and the internal representations learned by a Deep Q-Network (DQN) across its processing hierarchy. Human similarity judgments are collected through two online behavioral experiments using a triplet odd-one-out paradigm and compared with DQN layer activations using Representational Similarity Analysis (RSA) across three games: Pong, Ms. Pac-Man, and Space Invaders.

The full paper is available in `paper/TFG.pdf`.  
Experiment stimuli, trained model weights, and anonymised participant data are available at: **http://bit.ly/4uOyKpN**  
Online experiments: [Sparse (Spanish)](https://f7pu2mdvua.cognition.run) · [Sparse (English)](https://nlelxrroxi.cognition.run) · [Individual](https://chdkpkrwed.cognition.run)

---

## Requirements

```bash
pip install -e .          # installs the src package in editable mode
pip install -r requirements.txt
```

Python 3.11, 64-bit is required. See `requirements.txt` for the full list of dependencies. Key libraries: `stable-baselines3`, `torch`, `rsatoolbox`, `scipy`, `gymnasium`, `ale-py`, `joblib`, `pandas`, `matplotlib`.

---

## `cy_tste` — Stochastic Triplet Embedding

This project uses a compiled Python extension (`cy_tste`) implementing **Stochastic Triplet Embedding (t-STE)**, introduced by van der Maaten and Weinberger (2012):

> van der Maaten, L., & Weinberger, K. (2012). *Stochastic Triplet Embedding*.

The `cy_tste` module included with this project is based on the Cython port developed by Michael Wilber:

* https://github.com/gcr/cython_tste

Wilber's implementation is itself a Cython port of Laurens van der Maaten's original t-STE implementation.

### Modifications in this Project

The original `cy_tste` code targets Python 2 and is no longer actively maintained. The version distributed with this project has been lightly modified to support modern Python environments, including:

* Converting Python 2 `print` statements to Python 3 syntax.
* Making integer division explicit where required.
* Updating Cython and NumPy integration for modern build toolchains.
* Minor compatibility fixes for Python 3.x.

No algorithmic changes have been made.

### Using the Included Binary

A precompiled binary is provided in:

```text
lib/cy_tste.cp311-win_amd64.pyd
```

This binary was built for:

* Python 3.11
* 64-bit Windows

If your environment matches these requirements, no compilation is necessary.

### Recompiling for Another Platform

If the included binary is incompatible with your system (different operating system, Python version, or CPU architecture), rebuild the extension from source.

```bash
# Clone the original Cython implementation
git clone https://github.com/gcr/cython_tste

# Copy the source files into this project's lib directory
cp cython_tste/cy_tste.pyx lib/
cp cython_tste/setup.py lib/

# Apply the Python 3 compatibility changes
# (or use the modified sources already provided)

cd lib
python setup.py build_ext --inplace
```

The resulting extension module should be placed in `lib/`:

* Windows: `cy_tste.cp311-win_amd64.pyd`
* Linux/macOS: `cy_tste.so`

### Attribution and License

`cy_tste` is derived from software originally developed by Laurens van der Maaten and later ported to Cython by Michael Wilber.

The version included in this project contains compatibility modifications for Python 3 and modern NumPy/Cython toolchains. These changes are distributed as a modified version of the original software and are clearly identified as such.

Original copyright notices and license terms remain applicable to the `cy_tste` source code. See the accompanying license file for details.

---


## Project Structure

```
project/
│
├── src/                            # Shared package — imported by all scripts
│   ├── __init__.py
│   ├── config.py                   # Central paths, constants, hyperparameters
│   ├── utils.py                    # Shared utility functions
│   ├── models/
│   │   ├── custom_dqn.py           # Custom CNN feature extractor
│   │   └── per_dqn.py              # PER-DQN (future work, not used in paper)
│   └── wrappers/
│       └── environment_wrappers.py # Action-restricted Atari wrappers
│
├── lib/
│   └── cy_tste.cp311-win_amd64.pyd                 # Compiled t-STE extension (see above)
│
├── 01_training/                    # DQN agent training
│   ├── train_dqn_pong.py
│   ├── train_dqn_pacman.py
│   └── train_dqn_spaceinvaders.py
│
├── 02_stimulus_generation/         # Human gameplay recording and clip extraction
│   ├── 1_record_human_gameplay.py
│   ├── 2_extract_clips_from_gameplay.py
│   └── 3_render_4frame_clips.py
│
├── 03_clip_selection/              # Experiments 5–8: subset selection pipeline
│   ├── exp5_clip_representativeness.py
│   ├── exp6_optimal_n_clips.py
│   ├── exp7_simulated_annealing.py
│   └── exp8_triplet_reconstruction.py
│
├── 04_dqn_representations/         # Activation extraction, PCA, RDM computation
│   ├── 1_extract_frame_arrays.py
│   ├── 2_extract_activation_units.py
│   ├── 3_train_pca_models.py
│   └── 4_compute_dqn_rdms.py
│
├── 05_theoretical_models/          # Pixel, pixel PCA, Q-value, HCF models
│   ├── 1_build_pixel_pca_training_set.py
│   ├── 2_train_pixel_pca_models.py
│   ├── 3_extract_pixel_qvalue_states.py
│   ├── 4_compute_pixel_qvalue_rdms.py
│   ├── 5_extract_hcf_features_pong.py
│   └── 6_compute_hcf_rdm_pong.py
│
├── 06_triplet_generation/          # Triplet scoring, difficulty bucketing, CSV export
│   ├── 1_create_clip_index_map.py
│   ├── 2_score_and_bucket_triplets.py
│   └── 3_export_triplet_scores_csv.py
│
├── 07_human_experiments/           # Response cleaning, t-STE CV, human RDMs
│   ├── 1_clean_raw_experiment_csvs.py
│   ├── 2_index_triplets_by_clip.py
│   ├── 3_convert_to_tste_constraints.py
│   ├── 4_1_pilot_tste_cv.py
│   ├── 4_2_deployment_tste_cv.py
│   └── 5_filter_sparse_to_pong60_subset.py
│
├── 08_consistency_analyses/        # Experiments 9–10: human consistency
│   ├── exp9_individual_consistency.py
│   └── exp10_sparse_individual_agreement.py
│
├── 09_dqn_reliability/             # Experiments 11–13: cross-seed RSA, noise ceiling
│   ├── exp11_cross_seed_rsa.py
│   ├── exp12_triplet_difficulty_agreement.py
│   └── exp13_noise_ceiling.py
│
├── 10_main_analyses/               # Experiments 14–15 + Appendix A.9
│   ├── exp14_human_dqn_difficulty.py
│   ├── exp15_full_rsa_matrix.py
│   └── exp_appendix_continuous_difficulty.py
│
├── cognition/                      # Code used for the experiments design
│   ├── individual_experiment.py
│   └── sparse_experiment.py
│
├── paper/
│   └── TFG.pdf
│
├── pyproject.toml
├── requirements.txt
└── README.md
```


---

## Understanding the Data Scopes

Before reading the execution order, it is important to understand that several pipeline stages need to be run **more than once**, each time targeting a different dataset scope.

Four dataset scopes exist, each serving a different purpose:

| Scope name | Approx. size | Purpose |
|---|---|---|
| `pca_training` | ~900 clips/game | Training the PCA models for layer activations and pixel states. Large and diverse to ensure stable PCA components. Never used directly in experiments. |
| `pool25` | 25 clips/game | The manually selected candidate pool from which the final 15 clips are chosen by simulated annealing (Experiment 7). |
| `bigset` | 909 clips/game | The reference dataset used to evaluate how well any clip subset captures the full DQN representational geometry (Experiments 5–6). Also used for the HCF model and cross-seed RSA (Experiment 11). |
| `subset15` | 15 clips/game | The final optimized clip subset used in all human experiments and the main analyses. Selected by simulated annealing from `pool25` to best reproduce the `bigset` RSA structure. These are **not** separately recorded — they are a filtered subset of `pool25` clips, defined by `subsets_csv` in `src/config.py` after running `03_clip_selection/`. |

The scripts in `02_stimulus_generation/` and `04_dqn_representations/` must each be run once per scope where relevant. The table below summarizes which scopes each stage targets:

| Stage | `pca_training` | `pool25` | `bigset` | `subset15` |
|---|:---:|:---:|:---:|:---:|
| `02` — Record gameplay | ✓ | ✓ | ✓ | — |
| `02` — Extract clips | ✓ | ✓ | ✓ | — |
| `02` — Render 4-frame previews | — | ✓ | — | — |
| `04` — Extract frame arrays | ✓ | ✓ | ✓ | — |
| `04` — Extract activations | ✓ (seed_42) | ✓ (seed_42) | — | ✓ (all seeds, after 03) |
| `04` — Train PCA models | ✓ | — | — | — |
| `04` — Compute DQN RDMs | — | ✓ (seed_42) | ✓ (seed_42) | ✓ (all seeds, after 03) |
| `05` — Pixel PCA training set | ✓ | — | — | — |
| `05` — Train pixel PCA models | ✓ | — | — | — |
| `05` — Q-value / state-value RDMs | — | — | — | ✓ |
| `05` — HCF features and RDM (Pong) | — | — | ✓ | — |

`subset15` clips are not recorded or extracted separately — they are selected from `pool25` after running `03_clip_selection/exp7_simulated_annealing.py`. This is why `04_dqn_representations/` must be partially re-run after `03_clip_selection/` completes.
---


## Execution Order

Scripts must be run in pipeline order. Each stage depends on outputs from the previous one.

The pipeline has two phases that are not strictly sequential: `04_dqn_representations/` must run partially **before** `03_clip_selection/` and partially **after**. Read the scope table above before starting. Note that folder `04` runs before folder `03` — this is intentional, not a typo.

```
01_training/
  train_dqn_pong/pacman/spaceinvaders.py
  → Produces: models/{gym_id}/{seed}/final_model  (5 seeds × 3 games)
        │
        ▼
02_stimulus_generation/  [RUN 3 TIMES — once per scope]
  Scope 1 — pca_training : large diverse gameplay for PCA fitting
  Scope 2 — pool25       : 25 candidate clips per game (manual selection)
  Scope 3 — bigset       : 909-clip reference dataset
  → Produces: data/recordings/, data/clips/, data/arrays/ for each scope
        │
        ▼
04_dqn_representations/  [FIRST PASS — scopes: pca_training, pool25, bigset]
  1_extract_frame_arrays.py    ← run for pca_training, pool25, bigset
  2_extract_activations.py     ← run for pca_training (seed_42), pool25 (seed_42)
  3_train_pca_models.py        ← run for pca_training only
  4_compute_dqn_rdms.py        ← run for pool25 and bigset (seed_42)
  → Produces: data/activations/, models/pca_layer/,
              data/rdms/pool25/, data/rdms/bigset/
        │
        ▼
03_clip_selection/  [Uses pool25 and bigset RDMs from first pass of 04]
  exp5_clip_representativeness.py  ← compares pool25 RSA to bigset RSA (Exp 5)
  exp6_optimal_n_clips.py          ← determines 15 is optimal subset size (Exp 6)
  exp7_simulated_annealing.py      ← selects best 15 from pool25 (Exp 7)
  exp8_triplet_reconstruction.py   ← validates subset15 t-STE quality (Exp 8)
  → Produces: data/subsets/seed_42/{game}_best_subset_indices.csv
        │
        ▼
04_dqn_representations/  [SECOND PASS — scope: subset15, all 5 seeds]
  2_extract_activations.py     ← subset15, all seeds
  4_compute_dqn_rdms.py        ← subset15, all seeds
  multi_seed_pipeline.py       ← convenience: runs both for all seeds at once
  → Produces: data/rdms/subset15/{seed}/{game}/
        │
        ├─────────────────────────────────────────┐
        ▼                                         ▼
05_theoretical_models/                  06_triplet_generation/
  [Uses subset15 + bigset RDMs]           [Uses subset15 RDMs — REFERENCE_SEED only]
  1_build_pixel_pca_training_set.py       1_create_clip_index_map.py
  2_train_pixel_pca_models.py             2_score_and_bucket_triplets.py
  3_extract_pixel_qvalue_states.py        3_export_triplet_scores_csv.py
  4_compute_pixel_qvalue_rdms.py          → Produces: data/triplets/scores/
  5_extract_hcf_features_pong.py            triplet_scores_{game}.csv with
  6_compute_hcf_rdm_pong.py                 easy/medium/hard difficulty labels
  → Produces: data/rdms/bigset/pong/hcf/
    data/states/{seed}/subset15/{game}/
        │                                         │
        └─────────────────────────────────────────┘
                          │
                          ▼
        [MANUAL STEP — Upload to cognition.run]
        The files from 06_triplet_generation/ define which triplets
        participants see and their difficulty labels. Upload these to
        cognition.run to configure the online experiments.
        Two experiments were run:
          - Sparse experiment (~41 participants, 6 triplets/game each)
            Code: cognition/sparse_experiment.js
          - Individual experiment (9 participants, 60 Pong triplets each)
            Code: cognition/individual_experiment.js
        Download resulting CSVs from cognition.run into:
          data/experiment/sparse/       (sparse experiment)
          data/experiment/individual/   (individual experiment)
        │
        ▼
07_human_experiments/  [Process downloaded experiment data]
  1_clean_raw_experiment_csvs.py
  2_index_triplets_by_clip.py
  3_convert_to_tste_constraints.py
  4_1_pilot_tste_cv.py              ← pilot only (Appendix A.4)
  4_2_deployment_tste_cv.py         ← deployment (Appendix A.7)
  5_filter_sparse_to_pong60_subset.py
  → Produces: data/experiment/*/cleaned_results/ including human RDMs
        │
        ▼
08_consistency_analyses/
  exp9_individual_consistency.py       ← requires individual experiment data
  exp10_sparse_individual_agreement.py ← requires both experiments data
        │
        ▼
09_dqn_reliability/
  exp11_cross_seed_rsa.py              ← requires bigset RDMs, all seeds
  exp12_triplet_difficulty_agreement.py ← requires triplet scores from 06
  exp13_noise_ceiling.py               ← requires subset15 RDMs + sparse data
        │
        ▼
10_main_analyses/
  exp14_human_dqn_difficulty.py        ← requires 06 triplet scores + 07 sparse data
  exp15_full_rsa_matrix.py             ← requires all RDMs: 04 + 05 + 07 human RDMs
  exp_appendix_continuous_difficulty.py ← requires 06 triplet scores + 07 sparse data
```


---

## Shared Utilities

All scripts import from `src/`. The two key modules are:

**`src/config.py`** — central configuration. Contains `GAMES`, `SEEDS`, `REFERENCE_SEED`, all hyperparameter dictionaries (`DQN`, `REPR`, `TSTE`), and the full path registry. Use `get_path(key, **kwargs)` to resolve any path and `ensure(key, **kwargs)` to resolve and create output directories.


**`src/utils.py`** — shared functions used across multiple scripts:

| Function | Used in |
|---|---|
| `dqn_preprocess_from_16_frames(frames)` | Activation extraction, clip rendering |
| `extract_layer_name(key)` | All activation and RDM scripts |
| `rdm_spearman(rdm1, rdm2)` | All RSA comparisons |
| `upper_tri(rdm)` | RDM vectorization |
| `embedding_to_rdm(X)` | All t-STE pipelines |
| `build_triplets_from_rdm(rdm)` | Triplet generation |
| `remove_symmetric_triplets(triplets)` | Triplet generation |
| `add_symmetric_triplets(triplets)` | t-STE constraint preparation |
| `majority_vote_with_ties(group)` | Human response analysis |

---

## Reproducing the Paper Results

If you only want to reproduce the reported results without re-running the full pipeline from scratch, download the pre-generated data from **http://bit.ly/4uOyKpN** and place it under `data/`. Then run only the scripts in `08_consistency_analyses/`, `09_dqn_reliability/`, and `10_main_analyses/`.

Each experiment in the paper maps to a specific script:

| Experiment | Script | Section |
|---|---|---|
| Exp 1 — t-STE embedding validation (pilot) | `07_human_experiments/4_1_pilot_tste_cv.py` | Appendix A.4.1 |
| Exp 5 — Clip representativeness | `03_clip_selection/exp5_clip_representativeness.py` | Section 5.3.1 |
| Exp 6 — Optimal number of clips | `03_clip_selection/exp6_optimal_n_clips.py` | Section 5.3.2 |
| Exp 7 — Optimized clip subset (SA) | `03_clip_selection/exp7_simulated_annealing.py` | Section 5.3.3 |
| Exp 8 — Triplet reconstruction analysis | `03_clip_selection/exp8_triplet_reconstruction.py` | Section 5.3.4 |
| Exp 9 — Individual experiment consistency | `08_consistency_analyses/exp9_individual_consistency.py` | Section 5.4.1 |
| Exp 10 — Sparse–individual agreement | `08_consistency_analyses/exp10_sparse_individual_agreement.py` | Section 5.4.2 |
| Exp 11 — Cross-seed RDM consistency | `09_dqn_reliability/exp11_cross_seed_rsa.py` | Section 5.5.1 |
| Exp 12 — Difficulty and cross-seed agreement | `09_dqn_reliability/exp12_triplet_difficulty_agreement.py` | Section 5.5.2 |
| Exp 13 — DQN noise ceiling | `09_dqn_reliability/exp13_noise_ceiling.py` | Section 5.5.3 |
| Exp 14 — Human agreement by difficulty | `10_main_analyses/exp14_human_dqn_difficulty.py` | Section 5.6.1 |
| Exp 15 — Full RSA matrix (Fig. 1) | `10_main_analyses/exp15_full_rsa_matrix.py` | Section 5.7 |
| Appendix A.9 — Continuous difficulty bins | `10_main_analyses/exp_appendix_continuous_difficulty.py` | Appendix A.9 |
| Appendix A.7 — Deployment t-STE CV | `07_human_experiments/4_2_deployment_tste_cv.py` | Appendix A.7 |

---

## Citation

```
Arrocha Gascón, O. (2026). Characterizing human state space representations
with Deep Reinforcement Learning Models. Final Degree Project in Artificial
Intelligence, Escola d'Enginyeria, Universitat Autònoma de Barcelona.
```

---

## References

Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.

Kriegeskorte, N., Mur, M., & Bandettini, P. (2008). Representational similarity analysis. *Frontiers in Systems Neuroscience*, 2, 4.

Cross, L., Cockburn, J., Yue, Y., & O'Doherty, J. P. (2021). Using deep reinforcement learning to reveal how the brain encodes abstract state-space representations. *Neuron*, 109(4), 724–738.

van der Maaten, L., & Weinberger, K. (2012). Stochastic triplet embedding. *IEEE MLSP*, 1–6.
