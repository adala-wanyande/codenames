import pandas as pd
import os
import csv
import json
from datetime import datetime


class TournamentLogger:

    def __init__(self, filename="tournament_results_vs_baseline.csv"):
        self.filename = os.path.join("data", "logs", filename)

        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        self.match_data = []
        self.turn_data = []

    # =========================================================
    # PROMPT STRATEGY LABELS
    # =========================================================

    @staticmethod
    def _prompt_strategy(spymaster_type, shot):
        mapping = {
            ("word2vec",      0): "Baseline",
            ("single_cot",    0): "Zero-Shot",
            ("single_cot",    1): "Few-Shot",
            ("double_cot",    0): "CoT",
            ("double_cot",    1): "CoT+Few-Shot",
            ("double_cot_SR", 0): "SR-CoT",
            ("double_cot_SR", 1): "SR-CoT+Few-Shot",
        }

        return mapping.get(
            (spymaster_type, shot),
            f"{spymaster_type}|{shot}"
        )

    # =========================================================
    # MODEL DISPLAY LABELS
    # =========================================================

    @staticmethod
    def _model_label(model_name):

        labels = {
            "word2vec": "Word2Vec",
            "qwen2.5":  "Qwen-2.5",
            "qwen3":    "Qwen-3",
            "mistral":  "Mistral",
            "llama3":   "LLaMA-3",
        }

        return labels.get(model_name, model_name)

    # =========================================================
    # TURN-LEVEL LOGGING
    # =========================================================

    def log_turn(
        self,
        match_id,
        turn,
        team,
        words_left,
        clue,
        count,
        guesses,
        correct,
        wrong,
        assassin,
        invalid_clue,
        guess_trace,
        red_spymaster,
        blue_spymaster,
        spymaster_type,
        red_model,
        blue_model,
        model_name="qwen2.5",
        temperature=0.0,
        shot=0,
    ):

        self.turn_data.append({

            # ---------- identifiers ----------
            "Match_ID": match_id,
            "Turn": turn,
            "Team": team,

            # ---------- clue ----------
            "Clue": clue,
            "Clue_Count": count,
            "Invalid_Clue": invalid_clue,
            "Fallback_Clue": int(clue == "random"),

            # ---------- guesses ----------
            "Guesses_Made": len(guesses),
            "Correct_Guesses": correct,
            "Wrong_Guesses": wrong,
            "Guess_Accuracy": (
                correct / max(len(guesses), 1)
            ),

            # ---------- board state ----------
            "Words_Left": words_left,

            # ---------- outcomes ----------
            "Assassin_Hit": assassin,

            # ---------- trace ----------
            "Guess_Trace": json.dumps(guess_trace),

            # ---------- experiment ----------
            "Spymaster_Type": spymaster_type,
            "Prompt_Strategy": self._prompt_strategy(
                spymaster_type,
                shot
            ),
            "Shot": shot,

            # ---------- models ----------
            "Red_Model": red_model,
            "Blue_Model": blue_model,
            "Model_Name": model_name,
            "Model_Label": self._model_label(model_name),
            "Temperature": temperature,

            # ---------- agents ----------
            "Red_Spymaster": red_spymaster,
            "Blue_Spymaster": blue_spymaster,

            # ---------- timestamp ----------
            "Timestamp": datetime.now().isoformat(),
        })

    # =========================================================
    # MATCH-LEVEL LOGGING
    # =========================================================

    def log_match(
        self,
        match_id,
        game_engine,

        red_spymaster,
        blue_spymaster,

        red_model,
        blue_model,

        spymaster_type,
        shot,

        model_name="qwen2.5",
        temperature=0.0,
    ):

        winner = game_engine.winner

        self.match_data.append({

            # ---------- identifiers ----------
            "Match_ID": match_id,

            # ---------- winner ----------
            "Winner": winner,
            "Red_Win": int(winner == "Red"),
            "Blue_Win": int(winner == "Blue"),

            # ---------- game stats ----------
            "Turns_Taken": game_engine.turn_count,

            "Red_Cards_Found": getattr(
                game_engine,
                "red_found",
                0
            ),

            "Blue_Cards_Found": getattr(
                game_engine,
                "blue_found",
                0
            ),

            # ---------- strategy ----------
            "Spymaster_Type": spymaster_type,

            "Prompt_Strategy": self._prompt_strategy(
                spymaster_type,
                shot
            ),

            "Shot": shot,

            # ---------- model ----------
            "Model_Name": model_name,
            "Model_Label": self._model_label(model_name),
            "Temperature": temperature,

            # ---------- teams ----------
            "Red_Spymaster": red_spymaster,
            "Blue_Spymaster": blue_spymaster,

            "Red_Model": red_model,
            "Blue_Model": blue_model,

            # ---------- timestamp ----------
            "Timestamp": datetime.now().isoformat(),
        })

    # =========================================================
    # SAVE CSV FILES
    # =========================================================

    def save_results(self):

        match_df = pd.DataFrame(self.match_data)
        turn_df = pd.DataFrame(self.turn_data)

        match_path = self.filename
        turn_path = self.filename.replace(
            ".csv",
            "_turns.csv"
        )

        match_df.to_csv(match_path, index=False)
        turn_df.to_csv(turn_path, index=False)

        print(f"\n📊 Match results saved to:")
        print(match_path)

        print(f"\n📊 Turn-level results saved to:")
        print(turn_path)

    # =========================================================
    # SUMMARY
    # =========================================================

    def print_summary(self):

        df = pd.DataFrame(self.match_data)

        if df.empty:
            print("No matches logged.")
            return

        red_win_rate = df["Red_Win"].mean() * 100

        avg_turns = (
            df[df["Winner"] == "Red"]["Turns_Taken"]
            .mean()
        )

        avg_turns = 0 if pd.isna(avg_turns) else avg_turns

        print("\n" + "=" * 40)
        print("📈 TOURNAMENT SUMMARY 📈")
        print("=" * 40)

        print(f"Games Played: {len(df)}")
        print(f"Red Win Rate: {red_win_rate:.2f}%")
        print(f"Average Turns (Red wins): {avg_turns:.2f}")

        if "Temperature" in df.columns:
            print("\nTemperature Settings:")
            print(df["Temperature"].value_counts())

        print("=" * 40)


# =============================================================
# INVALID CLUE LOGGER
# =============================================================

class invalid_clues:

    @staticmethod
    def log_invalid_clue(
        clue,
        team,
        turn,
        targets,
        board_words
    ):

        os.makedirs("data", exist_ok=True)

        filepath = os.path.join(
            "data",
            "spymaster_invalid_clues.csv"
        )

        file_exists = os.path.isfile(filepath)

        with open(
            filepath,
            mode="a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            if not file_exists:

                writer.writerow([
                    "timestamp",
                    "turn",
                    "team",
                    "clue",
                    "targets",
                    "board_words",
                ])

            writer.writerow([
                datetime.now().isoformat(),
                turn,
                team,
                clue,
                "|".join(targets),
                "|".join(board_words),
            ])