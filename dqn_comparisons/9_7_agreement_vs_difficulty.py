import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# =========================================================
# CONFIG
# =========================================================
GAMES        = ["pacman", "pong", "spaceinvaders"]

TRIPLET_SCORES_DIR   = "../data/triplets_results/triplet_scores"
SPARSE_RESPONSES_DIR = "../data/triplets_results/final_experiment/cleaned_results"
OUTPUT_DIR           = "../data/triplets_results/agreement_vs_difficulty"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_BINS      = 10   # number of equal-width bins along difficulty_score axis
MIN_TRIPLETS_PER_BIN = 3  # bins with fewer triplets are dropped

# =========================================================
# HELPERS
# =========================================================
def bin_agreement(difficulty_scores, correct, n_bins, min_per_bin):
    """
    Bin triplets by difficulty_score, compute mean agreement per bin.
    Returns (bin_centers, mean_agreement, sem_agreement, counts).
    """
    bins   = np.linspace(0, 1, n_bins + 1)
    centers, means, sems, counts = [], [], [], []
    for i in range(n_bins):
        mask = (difficulty_scores >= bins[i]) & (difficulty_scores < bins[i+1])
        if mask.sum() < min_per_bin:
            continue
        vals = correct[mask]
        centers.append((bins[i] + bins[i+1]) / 2)
        means.append(np.mean(vals))
        sems.append(np.std(vals) / np.sqrt(len(vals)))
        counts.append(len(vals))
    return (np.array(centers), np.array(means),
            np.array(sems),    np.array(counts))


# =========================================================
# MAIN LOOP
# =========================================================
for game in GAMES:

    print("\n" + "=" * 60)
    print(f"GAME: {game}")
    print("=" * 60)

    # --- Load 9_5 master CSV ---
    master_csv = os.path.join(TRIPLET_SCORES_DIR, f"triplet_scores_{game}.csv")
    if not os.path.exists(master_csv):
        print(f"  Missing master CSV: {master_csv}, skipping.")
        continue
    master_df = pd.read_csv(master_csv)

    # Identify other-seed answer columns
    seed_answer_cols = [c for c in master_df.columns
                        if c.endswith("_answer") and c != "seed_42_answer"]
    other_seeds = [c.replace("_answer", "") for c in seed_answer_cols]

    if not seed_answer_cols:
        print("  No other seed columns found — skipping DQN curve.")

    # -----------------------------------------------
    # DQN AGREEMENT vs DIFFICULTY SCORE (all 455)
    # -----------------------------------------------
    # For each triplet, agreement = fraction of other seeds that match seed_42
    if seed_answer_cols:
        agree_matrix = np.column_stack([
            (master_df[col] == master_df["odd_idx"]).astype(float)
            for col in seed_answer_cols
        ])  # shape (n_triplets, n_other_seeds)
        dqn_agree_per_triplet = agree_matrix.mean(axis=1)

        dqn_centers, dqn_means, dqn_sems, dqn_counts = bin_agreement(
            master_df["difficulty_score"].values,
            dqn_agree_per_triplet,
            N_BINS, MIN_TRIPLETS_PER_BIN
        )
        print(f"  DQN bins: {len(dqn_centers)}")

    # -----------------------------------------------
    # HUMAN AGREEMENT vs DIFFICULTY SCORE (sparse)
    # -----------------------------------------------
    sparse_csv = os.path.join(SPARSE_RESPONSES_DIR,
                              f"{game}_triplets_indexed_with_difficulty.csv")
    if not os.path.exists(sparse_csv):
        print(f"  Missing sparse responses: {sparse_csv}, skipping human curve.")
        human_available = False
    else:
        human_available = True
        sparse_df = pd.read_csv(sparse_csv)

        # Majority vote per triplet
        sparse_df["triplet_fs"] = sparse_df.apply(
            lambda r: frozenset({int(r["similar_clip_1_idx"]),
                                  int(r["similar_clip_2_idx"]),
                                  int(r["odd_clip_idx"])}), axis=1
        )
        majority = (
            sparse_df.groupby("triplet_fs")["odd_clip_idx"]
            .agg(lambda x: x.value_counts().index[0])
            .reset_index()
            .rename(columns={"odd_clip_idx": "human_majority"})
        )

        # Match to master_df to get difficulty_score
        master_df["triplet_fs"] = master_df.apply(
            lambda r: frozenset({int(r["similar_1_idx"]),
                                  int(r["similar_2_idx"]),
                                  int(r["odd_idx"])}), axis=1
        )
        sparse_master = master_df.merge(majority, on="triplet_fs", how="inner")
        sparse_master["human_correct"] = (
            sparse_master["human_majority"] == sparse_master["odd_idx"]
        ).astype(float)

        print(f"  Sparse triplets matched: {len(sparse_master)}")
        print(f"  Human overall agreement with seed_42: "
              f"{sparse_master['human_correct'].mean():.3f}")

        hum_centers, hum_means, hum_sems, hum_counts = bin_agreement(
            sparse_master["difficulty_score"].values,
            sparse_master["human_correct"].values,
            N_BINS, MIN_TRIPLETS_PER_BIN
        )
        print(f"  Human bins: {len(hum_centers)}")

    # -----------------------------------------------
    # PRINT TABLE
    # -----------------------------------------------
    print("\n  Difficulty score bins — DQN agreement:")
    if seed_answer_cols:
        for c, m, s, n in zip(dqn_centers, dqn_means, dqn_sems, dqn_counts):
            print(f"    score={c:.2f}  agreement={m:.3f} ± {s:.3f}  (n={n})")

    if human_available:
        print("\n  Difficulty score bins — Human agreement:")
        for c, m, s, n in zip(hum_centers, hum_means, hum_sems, hum_counts):
            print(f"    score={c:.2f}  agreement={m:.3f} ± {s:.3f}  (n={n})")

    # -----------------------------------------------
    # PLOT
    # -----------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    if seed_answer_cols:
        ax.plot(dqn_centers, dqn_means, "o-", color="#2980b9",
                label=f"DQN seeds (n={len(other_seeds)})", linewidth=2)
        ax.fill_between(dqn_centers,
                        dqn_means - dqn_sems,
                        dqn_means + dqn_sems,
                        alpha=0.2, color="#2980b9")

    if human_available:
        ax.plot(hum_centers, hum_means, "s--", color="#e74c3c",
                label="Humans (majority vote)", linewidth=2)
        ax.fill_between(hum_centers,
                        hum_means - hum_sems,
                        hum_means + hum_sems,
                        alpha=0.2, color="#e74c3c")

    ax.axhline(1/3, color="black", linestyle=":", linewidth=1.2, label="Chance (1/3)")
    ax.axhline(1.0, color="gray",  linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Difficulty score  (0 = hardest, 1 = easiest)", fontsize=12)
    ax.set_ylabel("Agreement with seed_42 answer", fontsize=12)
    ax.set_title(f"{game.capitalize()} — Agreement vs Difficulty Score", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"{game}_agreement_vs_difficulty.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n  Saved plot → {plot_path}")

    # -----------------------------------------------
    # SAVE CSVs
    # -----------------------------------------------
    if seed_answer_cols:
        dqn_out = pd.DataFrame({
            "bin_center":  dqn_centers,
            "mean_agreement": dqn_means,
            "sem": dqn_sems,
            "n_triplets": dqn_counts,
        })
        dqn_out.to_csv(os.path.join(OUTPUT_DIR, f"{game}_dqn_agreement_bins.csv"),
                       index=False)

    if human_available:
        hum_out = pd.DataFrame({
            "bin_center":     hum_centers,
            "mean_agreement": hum_means,
            "sem":            hum_sems,
            "n_triplets":     hum_counts,
        })
        hum_out.to_csv(os.path.join(OUTPUT_DIR, f"{game}_human_agreement_bins.csv"),
                       index=False)

    # -----------------------------------------------
    # SEPARATE PLOTS
    # -----------------------------------------------
    if seed_answer_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(dqn_centers, dqn_means, "o-", color="#2980b9",
                label=f"DQN seeds (n={len(other_seeds)})", linewidth=2)
        ax.fill_between(dqn_centers,
                        dqn_means - dqn_sems,
                        dqn_means + dqn_sems,
                        alpha=0.2, color="#2980b9")
        ax.axhline(1/3, color="black", linestyle=":", linewidth=1.2, label="Chance (1/3)")
        ax.axhline(1.0, color="gray",  linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Difficulty score  (0 = hardest, 1 = easiest)", fontsize=12)
        ax.set_ylabel("Agreement with seed_42 answer", fontsize=12)
        ax.set_title(f"{game.capitalize()} — DQN Agreement vs Difficulty Score", fontsize=13)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, f"{game}_dqn_agreement_vs_difficulty.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"  Saved DQN-only plot -> {plot_path}")

    if human_available:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(hum_centers, hum_means, "s--", color="#e74c3c",
                label="Humans (majority vote)", linewidth=2)
        ax.fill_between(hum_centers,
                        hum_means - hum_sems,
                        hum_means + hum_sems,
                        alpha=0.2, color="#e74c3c")
        ax.axhline(1/3, color="black", linestyle=":", linewidth=1.2, label="Chance (1/3)")
        ax.axhline(1.0, color="gray",  linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Difficulty score  (0 = hardest, 1 = easiest)", fontsize=12)
        ax.set_ylabel("Agreement with seed_42 answer", fontsize=12)
        ax.set_title(f"{game.capitalize()} — Human Agreement vs Difficulty Score", fontsize=13)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, f"{game}_human_agreement_vs_difficulty.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"  Saved human-only plot -> {plot_path}")


print("\nDONE.")
