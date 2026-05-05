import pandas as pd

# =========================
# CONFIG
# =========================
games=["pacman", "pong", "spaceinvaders"]

for game in games:

    input_csv = f"../data/triplets_results/own_data/cleaned_results/{game}_triplets_indexed.csv" #f"../data/cleaned_results/{game}_triplets_indexed.csv"
    output_csv = f"../data/triplets_results/own_data/cleaned_results/{game}_tste_constraints.csv" #f"../data/cleaned_results/{game}_tste_constraints.csv"

    # =========================
    # LOAD ORIGINAL CSV
    # =========================
    df = pd.read_csv(input_csv)

    # Check expected columns
    expected_cols = {"participant_id", "similar_clip_1_idx", "similar_clip_2_idx", "odd_clip_idx"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # =========================
    # CONVERT TO 2 DIRECTED CONSTRAINTS
    # =========================
    rows = []

    for _, row in df.iterrows():
        participant = row["participant_id"]
        sim1 = int(row["similar_clip_1_idx"])
        sim2 = int(row["similar_clip_2_idx"])
        odd = int(row["odd_clip_idx"])

        # Constraint 1: sim1 is closer to sim2 than to odd
        rows.append({
            "participant_id": participant,
            "reference": sim1,
            "near": sim2,
            "far": odd
        })

        # Constraint 2: sim2 is closer to sim1 than to odd
        rows.append({
            "participant_id": participant,
            "reference": sim2,
            "near": sim1,
            "far": odd
        })

    # Create output DataFrame
    constraints_df = pd.DataFrame(rows)

    # =========================
    # SAVE
    # =========================
    constraints_df.to_csv(output_csv, index=False)

    print(f"Saved {len(constraints_df)} directed constraints to: {output_csv}")
    print(constraints_df.head(10))

#TODO: Mirar que este cósigo esté bien, cambiar los paths, ejecutarlo, cambaiar los paths de los siguientes ficheros para que usen este y sacar resultados
#TODO: Tendré que cambiar los resultados de la presentacion