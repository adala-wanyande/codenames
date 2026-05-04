"""
Generate 100 deterministic benchmark boards for reproducible tournament evaluation.
Run once: python data/generate_benchmark_boards.py
"""

import json
import os
import random

random.seed(42)

SAMPLE_VOCAB = [
    "Apple", "Beach", "Car", "Dog", "Elephant", "Frog", "Ghost", "Hat", "Ice", "Jacket",
    "Kite", "Lemon", "Moon", "Nut", "Ocean", "Piano", "Queen", "Rose", "Star", "Tree",
    "Umbrella", "Van", "Water", "Xylophone", "Yacht", "Zebra", "Airplane", "Bear", "Cat",
    "Dance", "Eagle", "Fire", "Gold", "Helicopter", "Island", "Jungle", "Kangaroo", "Lion",
    "Mountain", "Ninja", "Octopus", "Pirate", "Robot", "Spider", "Train", "Unicorn",
    "Vampire", "Whale", "Zombie"
]

boards = []
for i in range(100):
    words = random.sample(SAMPLE_VOCAB, 25)
    identities = ["Red"] * 9 + ["Blue"] * 8 + ["Neutral"] * 7 + ["Assassin"] * 1
    random.shuffle(identities)
    boards.append({"id": i, "words": words, "identities": identities})

out_path = os.path.join(os.path.dirname(__file__), "benchmark_boards.json")
with open(out_path, "w") as f:
    json.dump(boards, f, indent=2)

print(f"Generated {len(boards)} benchmark boards → {out_path}")
