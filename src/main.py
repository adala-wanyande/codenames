from src.engine.game import CodenamesGame
from src.agents.llm import LLMSpymaster, LLMOperative

SAMPLE_VOCAB = [
    "Apple", "Beach", "Car", "Dog", "Elephant", "Frog", "Ghost", "Hat", "Ice", "Jacket",
    "Kite", "Lemon", "Moon", "Nut", "Ocean", "Piano", "Queen", "Rose", "Star", "Tree",
    "Umbrella", "Van", "Water", "Xylophone", "Yacht", "Zebra", "Airplane", "Bear", "Cat",
    "Dance", "Eagle", "Fire", "Gold", "Helicopter", "Island", "Jungle", "Kangaroo", "Lion",
    "Mountain", "Ninja", "Octopus", "Pirate", "Robot", "Spider", "Train", "Unicorn",
    "Vampire", "Whale", "Zombie"
]

def run_automated_game():
    print("🤖 STARTING AI vs AI CODENAMES MATCH 🤖\n")
    
    game = CodenamesGame(SAMPLE_VOCAB)
    operative = LLMOperative(name="Operative Qwen", model_type="ollama", model_name="qwen2.5")
    spymaster = LLMSpymaster(name="Spymaster Qwen", model_type="ollama", model_name="qwen2.5", operative=operative)
    # EIRINI'S REQUEST: Initial Spymaster Board Visualization
    print("👀 INITIAL SPYMASTER BOARD 👀")
    game.display(view="spymaster")

    while not game.is_game_over:
        game.turn_count += 1
        print(f"\n{'='*10} TURN {game.turn_count} {'='*10}")

        # ---------------- SPYMASTER ----------------
        team_color = game.current_team
        spymaster_board = game.get_spymaster_board()
        unrevealed_targets = game.get_unrevealed_targets(team_color)

        print(f" Team to play: {team_color}")
        print(f"Total targets left: {len(unrevealed_targets)}")
        print(f"Unrevealed targets: {unrevealed_targets}")

        clue, count = spymaster.group_words(spymaster_board, unrevealed_targets)
        print(f"🎤 Spymaster says: '{clue}' for {count}")

        if not game.is_valid_clue(clue):
            print("❌ Illegal clue. Turn skipped.")
            game.switch_team()
            continue

        # ---------------- OPERATIVE ----------------
        operative_board = game.get_operative_board()
        print(f"🤔 Operative thinking on '{clue}' ({count} guesses allowed)")

        guessed_words = operative.guess_words(operative_board, clue, count)
        print(f"🤔 Operative guesses: {guessed_words}")

        guesses_left = count

        for guess in guessed_words:
            if guesses_left <= 0:
                break

            print(f" -> Revealing '{guess}'...")

            identity, continue_turn, game_over = game.process_guess(guess)

            if not identity:
                print("    ❌ Invalid word (hallucination)")
                break

            print(f"    Result: {identity}")

            # GAME OVER
            if game_over:
                break

            # ASSASSIN
            if identity == "Assassin":
                print("💀 Assassin hit!")
                break

            # WRONG TEAM (Blue/Neutral depending on current team)
            if identity != team_color:
                print("    ❌ Wrong guess → turn ends")
                break

            # CORRECT GUESS
            print("    ✅ Correct guess → can continue")
            guesses_left -= 1

            if not continue_turn:
                break

        # IMPORTANT: switch team AFTER full turn ends
        game.switch_team()
            
    # --- GAME OVER ---
    print("\n" + "*"*30)
    print("GAME OVER!")
    if game.winner == 'Red':
        print(f"🏆 AI Team WINS in {game.turn_count} turns!")
    else:
        print(f"💀 AI Team LOST")
    print("*"*30)

if __name__ == "__main__":
    run_automated_game()