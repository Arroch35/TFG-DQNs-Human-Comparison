"""
exp_appendix_continuous_difficulty.py
Plot DQN-seed and human agreement against continuous difficulty score,
binned into N_BINS equal-width bins.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import GAMES, get_path, ensure
from src.utils import majority_vote_with_ties   # not used directly but kept for consistency

# =========================================================
# CONFIG
# =========================================================
N_BINS               = 10
MIN_TRIPLETS_PER_BIN = 3
OUTPUT_DIR = ensure("results_agreement_bins")

# =========================================================
# HELPERS
# =========================================================
def bin_agreement(difficulty_scores, correct, n_bins, min_per_bin):
    bins = np.linspace(0, 1, n_bins + 1)
    centers, means, sems, counts = [], [], [], []
    for i in range(n_bins):
        mask = (difficulty_scores >= bins[i]) & (difficulty_scores < bins[i + 1])
        if mask.sum() < min_per_bin:
            continue
        vals = correct[mask]
        centers.append((bins[i] + bins[i + 1]) / 2)
        means.append(np.mean(vals))
        sems.append(np.std(vals) / np.sqrt(len(vals)))
        counts.append(len(vals))
    return np.array(centers), np.array(means), np.array(sems), np.array(counts)


def save_and_close(fig, path):
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# =========================================================
# MAIN
# =========================================================
for game in GAMES:
    print(f"\n{'='*60}\nGAME: {game}\n{'='*60}")

    # ── Master CSV ────────────────────────────────────────
    master_csv = get_path("triplets_scores_csv", game=game)
    if not master_csv.exists():
        print(f"  Missing master CSV: {master_csv}, skipping."); continue
    master_df = pd.read_csv(master_csv)

    seed_answer_cols = [c for c in master_df.columns if c.endswith("_answer") and c != "seed_42_answer"]
    other_seeds      = [c.replace("_answer", "") for c in seed_answer_cols]

    # ── DQN agreement vs difficulty ───────────────────────
    dqn_available = bool(seed_answer_cols)
    if dqn_available:
        agree_matrix = np.column_stack([
            (master_df[col] == master_df["odd_idx"]).astype(float) for col in seed_answer_cols
        ])
        dqn_agree = agree_matrix.mean(axis=1)
        dqn_centers, dqn_means, dqn_sems, dqn_counts = bin_agreement(
            master_df["difficulty_score"].values, dqn_agree, N_BINS, MIN_TRIPLETS_PER_BIN)
        print(f"  DQN bins: {len(dqn_centers)}")
    else:
        print("  No other seed columns — skipping DQN curve.")

    # ── Human agreement vs difficulty ─────────────────────
    sparse_csv = get_path("experiment_sparse") / f"{game}_triplets_indexed_with_difficulty.csv"
    human_available = sparse_csv.exists()

    if human_available:
        sparse_df = pd.read_csv(sparse_csv)
        sparse_df["triplet_fs"] = sparse_df.apply(
            lambda r: frozenset({int(r["similar_clip_1_idx"]), int(r["similar_clip_2_idx"]), int(r["odd_clip_idx"])}),
            axis=1,
        )
        majority = (
            sparse_df.groupby("triplet_fs")["odd_clip_idx"]
            .agg(lambda x: x.value_counts().index[0])
            .reset_index()
            .rename(columns={"odd_clip_idx": "human_majority"})
        )
        master_df["triplet_fs"] = master_df.apply(
            lambda r: frozenset({int(r["similar_1_idx"]), int(r["similar_2_idx"]), int(r["odd_idx"])}), axis=1)

        sparse_master = master_df.merge(majority, on="triplet_fs", how="inner")
        sparse_master["human_correct"] = (sparse_master["human_majority"] == sparse_master["odd_idx"]).astype(float)

        print(f"  Sparse triplets matched: {len(sparse_master)}")
        print(f"  Human overall agreement: {sparse_master['human_correct'].mean():.3f}")

        hum_centers, hum_means, hum_sems, hum_counts = bin_agreement(
            sparse_master["difficulty_score"].values,
            sparse_master["human_correct"].values,
            N_BINS, MIN_TRIPLETS_PER_BIN,
        )
        print(f"  Human bins: {len(hum_centers)}")
    else:
        print(f"  Missing sparse responses: {sparse_csv}, skipping human curve.")

    # ── Print tables ──────────────────────────────────────
    if dqn_available:
        print("\n  DQN agreement bins:")
        for c, m, s, n in zip(dqn_centers, dqn_means, dqn_sems, dqn_counts):
            print(f"    score={c:.2f}  agreement={m:.3f} ± {s:.3f}  (n={n})")
    if human_available:
        print("\n  Human agreement bins:")
        for c, m, s, n in zip(hum_centers, hum_means, hum_sems, hum_counts):
            print(f"    score={c:.2f}  agreement={m:.3f} ± {s:.3f}  (n={n})")

    # ── Combined plot ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    if dqn_available:
        ax.plot(dqn_centers, dqn_means, "o-", color="#2980b9",
                label=f"DQN seeds (n={len(other_seeds)})", linewidth=2)
        ax.fill_between(dqn_centers, dqn_means - dqn_sems, dqn_means + dqn_sems, alpha=0.2, color="#2980b9")
    if human_available:
        ax.plot(hum_centers, hum_means, "s--", color="#e74c3c", label="Humans (majority vote)", linewidth=2)
        ax.fill_between(hum_centers, hum_means - hum_sems, hum_means + hum_sems, alpha=0.2, color="#e74c3c")
    ax.axhline(1/3, color="black", linestyle=":", linewidth=1.2, label="Chance (1/3)")
    ax.axhline(1.0, color="gray",  linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Difficulty score  (0 = hardest, 1 = easiest)", fontsize=12)
    ax.set_ylabel("Agreement with seed_42 answer", fontsize=12)
    ax.set_title(f"{game.capitalize()} — Agreement vs Difficulty Score", fontsize=13)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    save_and_close(fig, OUTPUT_DIR / f"{game}_agreement_vs_difficulty.png")
    print(f"\n  Saved combined plot")

    # ── DQN-only plot ─────────────────────────────────────
    if dqn_available:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(dqn_centers, dqn_means, "o-", color="#2980b9",
                label=f"DQN seeds (n={len(other_seeds)})", linewidth=2)
        ax.fill_between(dqn_centers, dqn_means - dqn_sems, dqn_means + dqn_sems, alpha=0.2, color="#2980b9")
        ax.axhline(1/3, color="black", linestyle=":", linewidth=1.2, label="Chance (1/3)")
        ax.axhline(1.0, color="gray",  linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Difficulty score  (0 = hardest, 1 = easiest)", fontsize=12)
        ax.set_ylabel("Agreement with seed_42 answer", fontsize=12)
        ax.set_title(f"{game.capitalize()} — DQN Agreement vs Difficulty Score", fontsize=13)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        save_and_close(fig, OUTPUT_DIR / f"{game}_dqn_agreement_vs_difficulty.png")
        print(f"  Saved DQN-only plot")

    # ── Human-only plot ───────────────────────────────────
    if human_available:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(hum_centers, hum_means, "s--", color="#e74c3c", label="Humans (majority vote)", linewidth=2)
        ax.fill_between(hum_centers, hum_means - hum_sems, hum_means + hum_sems, alpha=0.2, color="#e74c3c")
        ax.axhline(1/3, color="black", linestyle=":", linewidth=1.2, label="Chance (1/3)")
        ax.axhline(1.0, color="gray",  linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Difficulty score  (0 = hardest, 1 = easiest)", fontsize=12)
        ax.set_ylabel("Agreement with seed_42 answer", fontsize=12)
        ax.set_title(f"{game.capitalize()} — Human Agreement vs Difficulty Score", fontsize=13)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        save_and_close(fig, OUTPUT_DIR / f"{game}_human_agreement_vs_difficulty.png")
        print(f"  Saved human-only plot")

    # ── Save CSVs ─────────────────────────────────────────
    if dqn_available:
        pd.DataFrame({"bin_center": dqn_centers, "mean_agreement": dqn_means,
                      "sem": dqn_sems, "n_triplets": dqn_counts}).to_csv(
            OUTPUT_DIR / f"{game}_dqn_agreement_bins.csv", index=False)
    if human_available:
        pd.DataFrame({"bin_center": hum_centers, "mean_agreement": hum_means,
                      "sem": hum_sems, "n_triplets": hum_counts}).to_csv(
            OUTPUT_DIR / f"{game}_human_agreement_bins.csv", index=False)

print("\nDONE.")
