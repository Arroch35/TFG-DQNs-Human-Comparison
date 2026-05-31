import numpy as np
import random
import math
import os

# =========================================================
# CONFIG
# =========================================================
GAMES = ["pacman", "pong", "spaceinvaders"]
SEED = "seed_42"
BASE_RDM_FOLDER =  f"../data/test_16_rdms/buenos_25/{SEED}/" #"../data/test_16_rdms/big_rdm_equal_size"

N_SELECT = 15
K_NEIGHBORS = 10

T0 = 1.0
T_MIN = 1e-4
DECAY = 0.9995
N_STEPS = 50000

SAVE_FOLDER = f"../data/sa_results/buenos_25/{SEED}/" #"../data/sa_results"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# =========================================================
# OBJECTIVE FUNCTION
# =========================================================
def subset_score(S, rdm):

    S = list(S)
    total = 0.0
    count = 0

    for i in S:

        dists = rdm[i]

        subset_dists = [(j, dists[j]) for j in S if j != i]
        subset_dists.sort(key=lambda x: x[1])

        if len(subset_dists) < 2:
            continue

        near = subset_dists[:K_NEIGHBORS]
        far = subset_dists[-K_NEIGHBORS:]

        local_scores = []

        for j, dij in near:
            for k, dik in far:
                s = (dik - dij) / (dik + dij + 1e-8)
                local_scores.append(s)

        if local_scores:
            total += np.mean(local_scores)
            count += 1

    return total / max(count, 1)

# =========================================================
# SIMULATED ANNEALING FUNCTION
# =========================================================
def run_sa(rdm, game_name):

    N = rdm.shape[0]

    print("\n" + "="*60)
    print(f"GAME: {game_name}")
    print("="*60)

    # Initial random subset
    current = set(random.sample(range(N), N_SELECT))
    current_score = subset_score(current, rdm)

    best = set(current)
    best_score = current_score

    print("Initial score:", current_score)

    T = T0

    for step in range(N_STEPS):

        # propose move
        current_list = list(current)

        out_idx = random.choice(current_list)
        in_idx = random.choice([i for i in range(N) if i not in current])

        new_set = set(current)
        new_set.remove(out_idx)
        new_set.add(in_idx)

        # evaluate
        new_score = subset_score(new_set, rdm)
        delta = new_score - current_score

        # accept / reject
        if delta > 0 or random.random() < math.exp(delta / max(T, 1e-8)):
            current = new_set
            current_score = new_score

            if current_score > best_score:
                best = set(current)
                best_score = current_score

        # cooling
        T = max(T * DECAY, T_MIN)

        if step % 500 == 0:
            print(f"[{game_name}] Step {step}, T={T:.5f}, score={current_score:.4f}, best={best_score:.4f}")

    return list(best), best_score

# =========================================================
# MAIN LOOP
# =========================================================
all_results = []

for game in GAMES:

    rdm_path = os.path.join(
        BASE_RDM_FOLDER,
        game,
        f"{game}_fc_correlation_RDM.npy"
    )

    if not os.path.exists(rdm_path):
        print(f"Missing RDM for {game}")
        continue

    rdm = np.load(rdm_path)

    best_indices, best_score = run_sa(rdm, game)

    print(f"\nFINAL BEST SCORE ({game}): {best_score}")
    print(f"SELECTED INDICES ({game}): {best_indices}")

    # Save indices
    save_path = os.path.join(SAVE_FOLDER, f"best_15_sa_{game}.npy")
    np.save(save_path, np.array(best_indices))

    # Save score
    with open(os.path.join(SAVE_FOLDER, f"{game}_score.txt"), "w") as f:
        f.write(f"{best_score:.6f}")

    all_results.append({
        "game": game,
        "score": best_score
    })

# =========================================================
# SUMMARY
# =========================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

for r in all_results:
    print(f"{r['game']}: {r['score']:.4f}")