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
│   ├── 2_extract_activations_seed42.py
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


## Execution Order

Scripts must be run in pipeline order. Each stage depends on outputs from the previous one.

```
01_training/                        Train 5 seeds × 3 games = 15 DQN models
        ↓
02_stimulus_generation/             Record gameplay, extract and organize clips
        ↓
04_dqn_representations/             Extract activations, train PCA, compute RDMs
        ↓
03_clip_selection/                  Select optimal 15-clip subsets (Exp 5–8)
        ↓
05_theoretical_models/              Compute pixel, Q-value, HCF reference RDMs
06_triplet_generation/              Score and bucket triplets, export for experiment
        ↓
[Upload triplets to cognition.run]  Run sparse and individual online experiments
        ↓
07_human_experiments/               Clean responses, run t-STE CV, compute human RDMs
        ↓
08_consistency_analyses/            Exp 9–10: internal consistency and cross-paradigm agreement
09_dqn_reliability/                 Exp 11–13: cross-seed RSA and noise ceiling
        ↓
10_main_analyses/                   Exp 14–15 + Appendix: full RSA matrices and final results
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
