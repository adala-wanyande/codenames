"""
Parse tournament_run.log into match-level and turn-level CSVs.
Run: python data/parse_log.py
"""

import re
import csv
import os

LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "tournament_run.log")
OUT_MATCH = os.path.join(os.path.dirname(__file__), "logs", "tournament_results.csv")
OUT_TURNS = os.path.join(os.path.dirname(__file__), "logs", "tournament_results_turns.csv")


def parse_log(path):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    matches = []
    turns   = []

    # Current config state
    current_config = {"spymaster_type": None, "shot": None, "model_name": None, "temperature": None}
    game_num = 0

    # Current game state
    in_game = False
    match_id = None
    turn_num = 0
    current_turn = None
    game_turns = []

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # ── Config header ──────────────────────────────────────────────────
        m = re.search(r"Config: (\S+) \| shot=(\S+) \| model=(\S+) \| temp=(\S+)", line)
        if m:
            current_config = {
                "spymaster_type": m.group(1),
                "shot":           int(m.group(2)),
                "model_name":     m.group(3),
                "temperature":    float(m.group(4)),
            }
            game_num = 0
            i += 1
            continue

        # ── Game start ─────────────────────────────────────────────────────
        m = re.search(r"Game (\d+)/\d+", line)
        if m:
            game_num = int(m.group(1))
            match_id = "{spymaster_type}_{shot}_{model_name}_{n}".format(
                **current_config, n=game_num - 1
            )
            in_game = True
            turn_num = 0
            game_turns = []
            current_turn = None
            i += 1
            continue

        if not in_game:
            i += 1
            continue

        # ── Turn header ────────────────────────────────────────────────────
        m = re.search(r"TURN (\d+)", line)
        if m:
            if current_turn:
                game_turns.append(current_turn)
            turn_num = int(m.group(1))
            current_turn = {
                "match_id":      match_id,
                "turn":          turn_num,
                "team":          None,
                "words_left":    None,
                "clue":          None,
                "count":         None,
                "guesses":       0,
                "correct":       0,
                "wrong":         0,
                "assassin":      0,
                "invalid_clue":  0,
                "spymaster_type": current_config["spymaster_type"],
                "model_name":    current_config["model_name"],
                "temperature":   current_config["temperature"],
                "fallback":      0,
            }
            i += 1
            continue

        if current_turn is None:
            i += 1
            continue

        # ── Team / words left ──────────────────────────────────────────────
        m = re.search(r"Team (\w+) has (\d+) targets left", line)
        if m:
            current_turn["team"]       = m.group(1)
            current_turn["words_left"] = int(m.group(2))
            i += 1
            continue

        # ── Clue ───────────────────────────────────────────────────────────
        m = re.search(r"Spymaster says: (\S+) for \((\d+)\)", line)
        if m:
            current_turn["clue"]     = m.group(1)
            current_turn["count"]    = int(m.group(2))
            current_turn["fallback"] = 1 if m.group(1).lower() == "random" else 0
            i += 1
            continue

        # ── Invalid clue ───────────────────────────────────────────────────
        if "Illegal clue" in line:
            current_turn["invalid_clue"] = 1
            i += 1
            continue

        # ── Operative guesses list ─────────────────────────────────────────
        m = re.search(r"Operative guesses: \[(.+)\]", line)
        if m:
            raw = m.group(1)
            words = [w.strip().strip("'\"") for w in raw.split(",")]
            current_turn["guesses"] = len(words)
            i += 1
            continue

        # ── Individual guess result ────────────────────────────────────────
        if "✅ Correct guess" in line:
            current_turn["correct"] += 1
        elif "❌ Wrong guess" in line:
            current_turn["wrong"] += 1
        elif "💀 Assassin hit" in line or "Assassin" in line and "Result:" in lines[i-1]:
            current_turn["assassin"] = 1

        # ── Game over ──────────────────────────────────────────────────────
        if "GAME OVER" in line:
            if current_turn:
                game_turns.append(current_turn)
                current_turn = None

            # Next non-empty line: "The X team wins in N turns!"
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1

            winner = None
            turn_count = None
            if j < len(lines):
                gm = re.search(r"The (\w+) team wins in (\d+) turns", lines[j])
                if gm:
                    winner      = gm.group(1)
                    turn_count  = int(gm.group(2))

            red_found  = sum(1 for t in game_turns if t["team"] == "Red" and t["correct"] > 0)
            blue_found = sum(1 for t in game_turns if t["team"] == "Blue" and t["correct"] > 0)

            matches.append({
                "Match_ID":       match_id,
                "Spymaster_Type": current_config["spymaster_type"],
                "Shot":           current_config["shot"],
                "Model_Name":     current_config["model_name"],
                "Temperature":    current_config["temperature"],
                "Spymaster":      current_config["spymaster_type"],
                "Operative":      "Word2Vec Operative" if current_config["spymaster_type"] == "word2vec" else f"Operative ({current_config['model_name']})",
                "Win":            winner,
                "Turns_Taken":    turn_count,
                "Red_Cards_Found":  red_found,
                "Blue_Cards_Found": blue_found,
            })

            turns.extend(game_turns)
            game_turns = []
            in_game = False
            i += 1
            continue

        i += 1

    return matches, turns


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    matches, turns = parse_log(LOG_PATH)

    match_fields = ["Match_ID", "Spymaster", "Spymaster_Type", "Shot", "Model_Name",
                    "Temperature", "Operative", "Win", "Turns_Taken",
                    "Red_Cards_Found", "Blue_Cards_Found"]
    # Rename turn keys to match the logger / visualize.py schema
    renamed = []
    for t in turns:
        renamed.append({
            "Match_ID":       t["match_id"],
            "Turn":           t["turn"],
            "Team":           t["team"],
            "Words_Left":     t["words_left"],
            "Clue":           t["clue"],
            "Clue_Count":     t["count"],
            "Guesses":        t["guesses"],
            "Correct_Guesses": t["correct"],
            "Wrong_Guesses":  t["wrong"],
            "Assassin_Hit":   t["assassin"],
            "Invalid_Clue":   t["invalid_clue"],
            "Spymaster_Type": t["spymaster_type"],
            "Shot":           t.get("shot", 0),
            "Model_Name":     t["model_name"],
            "Temperature":    t["temperature"],
            "Fallback":       t["fallback"],
        })
    turns = renamed

    turn_fields  = ["Match_ID", "Turn", "Team", "Words_Left", "Clue", "Clue_Count",
                    "Guesses", "Correct_Guesses", "Wrong_Guesses", "Assassin_Hit",
                    "Invalid_Clue", "Spymaster_Type", "Shot", "Model_Name", "Temperature", "Fallback"]

    write_csv(OUT_MATCH, matches, match_fields)
    write_csv(OUT_TURNS, turns,   turn_fields)

    print(f"Parsed {len(matches)} matches, {len(turns)} turns")
    print(f"Saved → {OUT_MATCH}")
    print(f"Saved → {OUT_TURNS}")

    # Quick summary
    from collections import Counter
    win_counts = Counter((r["Spymaster_Type"], r["Model_Name"], r["Win"]) for r in matches)
    print("\nResults snapshot:")
    for (stype, model, winner), count in sorted(win_counts.items()):
        print(f"  {stype} | {model} | winner={winner} : {count} game(s)")
