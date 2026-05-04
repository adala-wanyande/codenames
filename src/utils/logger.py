import pandas as pd
import os
import csv
from datetime import datetime

class TournamentLogger:
    def __init__(self, filename="tournament_results.csv"):
        self.filename = os.path.join("data", "logs", filename)
        # Create the directories if they don't exist
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        self.match_data = []
        self.turn_data = []
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
        return mapping.get((spymaster_type, shot), f"{spymaster_type}|{shot}")

    @staticmethod
    def _model_label(model_name):
        labels = {
            "word2vec":   "Word2Vec",
            "qwen2.5":    "Qwen-2.5",
            "qwen3":      "Qwen-3",
            "mistral":    "Mistral",
            "llama3":     "LLaMA-3",
        }
        return labels.get(model_name, model_name)

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
            spymaster_type,
            model_name="qwen2.5",
            temperature=0.0,
            shot=0,
        ):
        self.turn_data.append({
            "Match_ID":        match_id,
            "Turn":            turn,
            "Team":            team,
            "Words_Left":      words_left,
            "Clue":            clue,
            "Clue_Count":      count,
            "Guesses":         len(guesses),
            "Correct_Guesses": correct,
            "Wrong_Guesses":   wrong,
            "Assassin_Hit":    assassin,
            "Invalid_Clue":    invalid_clue,
            "Guess_Trace":     guess_trace,
            "Spymaster_Type":  spymaster_type,
            "Shot":            shot,
            "Prompt_Strategy": self._prompt_strategy(spymaster_type, shot),
            "Model_Name":      model_name,
            "Model_Label":     self._model_label(model_name),
            "Temperature":     temperature,
            "Fallback":        1 if clue == "random" else 0,
        })

    def log_match(self, match_id, spymaster_name, operative_name, game_engine,
                  spymaster_type=None, shot=None, model_name="qwen2.5", temperature=0.0):
        win = game_engine.winner

        match_stats = {
            "Match_ID":        match_id,
            "Spymaster":       spymaster_name,
            "Spymaster_Type":  spymaster_type,
            "Shot":            shot,
            "Prompt_Strategy": self._prompt_strategy(spymaster_type, shot),
            "Model_Name":      model_name,
            "Model_Label":     self._model_label(model_name),
            "Temperature":     temperature,
            "Operative":       operative_name,
            "Win":             win,
            "Turns_Taken":     game_engine.turn_count,
            "Red_Cards_Found": getattr(game_engine, "red_found", 0),
            "Blue_Cards_Found":getattr(game_engine, "blue_found", 0),
        }
        
        self.match_data.append(match_stats)

    def save_results(self):
        match_df = pd.DataFrame(self.match_data)
        turn_df = pd.DataFrame(self.turn_data)

        match_path = self.filename
        turn_path = self.filename.replace(".csv", "_turns.csv")

        match_df.to_csv(match_path, index=False)
        turn_df.to_csv(turn_path, index=False)

        print(f"\n📊 Match results saved to {match_path}")
        print(f"📊 Turn-level results saved to {turn_path}")
        
    def print_summary(self):
        df = pd.DataFrame(self.match_data)

        # ✅ Convert to numeric
        df["Red_Win"] = (df["Win"] == "Red").astype(int)

        win_rate = df["Red_Win"].mean() * 100

        avg_turns = df[df["Win"] == "Red"]["Turns_Taken"].mean()
        if pd.isna(avg_turns):
            avg_turns = 0

        print("\n" + "="*30)
        print("📈 TOURNAMENT SUMMARY 📈")
        print(f"Total Games Played: {len(df)}")
        print(f"Red Win Rate: {win_rate:.1f}%")
        print(f"Avg Turns (Red wins): {avg_turns:.1f}")

        # ✅ Safe columns (avoid crashes)
        if "Operative_Hallucinations" in df.columns:
            print(f"Total Hallucinations: {df['Operative_Hallucinations'].sum()}")

        if "Assassin_Hit" in df.columns:
            print(f"Total Assassin Hits: {df['Assassin_Hit'].sum()}")

        print("="*30)

class invalid_clues:
    def log_invalid_clue(clue, team, turn, targets, board_words):
        """
        Appends an invalid clue event to a global CSV file.
        File is shared across runs and safely created if missing.
        """

        os.makedirs("data", exist_ok=True)
        filepath = os.path.join("data", "spymaster_invalid_clues.csv")

        file_exists = os.path.isfile(filepath)

        with open(filepath, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            # Write header only once
            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "turn",
                    "team",
                    "clue",
                    "targets",
                    "board_words"
                ])

            writer.writerow([
                datetime.now().isoformat(),
                turn,
                team,
                clue,
                "|".join(targets),
                "|".join(board_words)
            ])