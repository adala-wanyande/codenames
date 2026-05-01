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

    def log_match(self, match_id, spymaster_name, operative_name, game_engine, hallucinations, illegal_clues):
        """
        Records all metrics for a single completed game.
        """
        win = 1 if game_engine.winner == 'Red' else 0
        assassin_hit = 1 if game_engine.winner == 'Assassin' else 0
        
        match_stats = {
            "Match_ID": match_id,
            "Spymaster": spymaster_name,
            "Operative": operative_name,
            "Win": win,
            "Turns_Taken": game_engine.turn_count,
            "Cards_Found": game_engine.red_found,
            "Assassin_Hit": assassin_hit,
            "Operative_Hallucinations": hallucinations,
            "Spymaster_Illegal_Clues": illegal_clues
        }
        
        self.match_data.append(match_stats)

    def save_results(self):
        """Saves the recorded data to a CSV file."""
        df = pd.DataFrame(self.match_data)
        
        # If file exists, append without headers. Otherwise, write new file.
        if os.path.exists(self.filename):
            df.to_csv(self.filename, mode='a', header=False, index=False)
        else:
            df.to_csv(self.filename, index=False)
            
        print(f"\n📊 Results successfully saved to {self.filename}")
        
    def print_summary(self):
        """Prints a quick terminal summary of the tournament."""
        df = pd.DataFrame(self.match_data)
        win_rate = df['Win'].mean() * 100
        avg_turns = df[df['Win'] == 1]['Turns_Taken'].mean()
        
        print("\n" + "="*30)
        print("📈 TOURNAMENT SUMMARY 📈")
        print(f"Total Games Played: {len(df)}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Avg Turns (in wins): {avg_turns:.1f}")
        print(f"Total Hallucinations: {df['Operative_Hallucinations'].sum()}")
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