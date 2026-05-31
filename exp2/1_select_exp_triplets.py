import json
import os
import random

common_games_data_path = "../data/jsons/common_games_data.json"

# 2. Open and load the JSON data
try:
    with open(common_games_data_path, "r") as file:
        common_games_data = json.load(file)
    
    # 'data' is now a normal Python dictionary!
    print("Data loaded successfully!")

except FileNotFoundError:
    print(f"Oops, couldn't find the file at {common_games_data_path}. Did you save it first?")



# Reproducibility (optional)
random.seed(42)

# -------------------------------------------------------
# SELECT 20 RANDOM TRIPLETS PER DIFFICULTY FROM PONG
# -------------------------------------------------------
pong_selected_triplets = {
    "PongNoFrameskip-v4": {}
}

game_name = "PongNoFrameskip-v4"

for difficulty in ["easy_triplets", "medium_triplets", "hard_triplets"]:

    triplets = common_games_data[game_name][difficulty]

    # Safety check
    if len(triplets) < 20:
        raise ValueError(
            f"Not enough triplets in {difficulty}. "
            f"Found {len(triplets)}, need 20."
        )

    # Random selection without replacement
    selected_triplets = random.sample(triplets, 20)

    pong_selected_triplets[game_name][difficulty] = selected_triplets

# -------------------------------------------------------
# PRINT RESULT
# -------------------------------------------------------
print("\nSELECTED PONG TRIPLETS")
print("=" * 80)

for difficulty, triplets in pong_selected_triplets[game_name].items():

    print(f"\n{difficulty}")
    print(f"Total selected: {len(triplets)}")

    for t in triplets:
        print(t)

# -------------------------------------------------------
# FULL STRUCTURE
# -------------------------------------------------------
print("\n\nFULL DATA STRUCTURE:\n")
print(pong_selected_triplets)


file_path = "../data/jsons/pong_final_triplet_exp.json"

# 4. Automatically create the directory if it doesn't exist
os.makedirs(os.path.dirname(file_path), exist_ok=True)


with open(file_path, "w") as file:
    json.dump(pong_selected_triplets, file)