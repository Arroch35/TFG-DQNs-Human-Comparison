"""
3_convert_to_tste_constrains.py
Convert indexed triplet CSVs (reference/near/far symmetric constraints)
for use as t-STE input.
"""
import pandas as pd

from src.config import GAMES, get_path, ensure

# =========================================================
# PATHS
# =========================================================
INPUT_DIR  = get_path("experiment_individual") 
OUTPUT_DIR = ensure("experiment_individual")

# =========================================================
# MAIN
# =========================================================
for game in GAMES:
    input_csv  = INPUT_DIR  / f"{game}_triplets_indexed_with_difficulty.csv"
    output_csv = OUTPUT_DIR / f"{game}_triplets_constraints_with_difficulty.csv"

    df = pd.read_csv(input_csv)

    expected_cols = {"participant_id", "similar_clip_1_idx", "similar_clip_2_idx", "odd_clip_idx"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"[{game}] Missing columns: {missing}")

    rows = []
    for _, row in df.iterrows():
        participant = row["participant_id"]
        difficulty  = row.get("difficulty", "unknown")
        sim1 = int(row["similar_clip_1_idx"])
        sim2 = int(row["similar_clip_2_idx"])
        odd  = int(row["odd_clip_idx"])

        # Two directed constraints per triplet response
        rows.append({"participant_id": participant, "reference": sim1, "near": sim2, "far": odd, "difficulty": difficulty})
        rows.append({"participant_id": participant, "reference": sim2, "near": sim1, "far": odd, "difficulty": difficulty})

    constraints_df = pd.DataFrame(rows)
    constraints_df.to_csv(output_csv, index=False)

    print(f"[{game}] Saved {len(constraints_df)} directed constraints → {output_csv}")
    print(constraints_df.head(5))
