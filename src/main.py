from src.engine import game
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

def run_automated_game(game = None, spymaster = None, operative = None, metrics = None):
    print("🤖 STARTING AI vs AI CODENAMES MATCH 🤖\n")
    if game is None:
        from src.tournament import init_metrics
        metrics = init_metrics()
        game = CodenamesGame(SAMPLE_VOCAB)
    if operative is None:
        operative = LLMOperative(name="Operative Qwen", model_type="ollama", model_name="qwen2.5")
    if spymaster is None:
        spymaster = LLMSpymaster(name="Spymaster Qwen", model_type="ollama", model_name="qwen2.5", operative=operative)

    
     # EIRINI'S REQUEST: Initial Spymaster Board Visualization
    print("👀 INITIAL SPYMASTER BOARD 👀")
    game.display(view="spymaster")

    while not game.is_game_over:
        game.turn_count += 1
        print(f"\n{'='*10} TURN {game.turn_count} {'='*10}")

        # ---------------- SPYMASTER ----------------
        team_color = game.current_team
        metrics["turns"][team_color] += 1
        spymaster_board = game.get_spymaster_board()
        unrevealed_targets = game.get_unrevealed_targets(team_color)

        print(f" Team to play: {team_color}")
        print(f"Total targets left: {len(unrevealed_targets)}")
        print(f"Unrevealed targets: {unrevealed_targets}")

        clue, count, combo = spymaster.group_words(spymaster_board, unrevealed_targets)
        spymaster_target_words = list(combo)
        print(f"🎤 Spymaster says: '{clue}' for {count}. Targets: {spymaster_target_words}")
        total_hallucinations = 0
        total_illegal_clues = 0
        if not game.is_valid_clue(clue):
            metrics["invalid_clues"][team_color] += 1
            game.switch_team()
            total_illegal_clues += 1
            print("❌ Illegal clue. Turn skipped.")
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
            metrics["total_guesses"][team_color] += 1
            print(f"    Result: {identity}")
            if not identity:
                total_hallucinations += 1
                metrics["hallucinations"][team_color] += 1
                print("    ❌ Invalid word (hallucination)")
                break
            if identity == "Assassin":
                metrics["assassin_hits"][team_color] += 1
                print("    💀 Assassin hit!")
                break



            if identity == team_color:
                metrics["correct_guesses"][team_color] += 1
                guesses_left -= 1
                print("    ✅ Correct guess → can continue")
            else:
                metrics["wrong_guesses"][team_color] += 1
                print("    ❌ Wrong guess → turn ends")
                break

            # GAME OVER
            if game_over:
                break

            if not continue_turn:
                break

        # IMPORTANT: switch team AFTER full turn ends
        game.switch_team()
            
    # --- GAME OVER ---
    print("\n" + "*"*30)
    print("GAME OVER!")

    if game.winner == 'Red':
        metrics["wins"][game.winner] += 1
        print(f"🏆 AI Team WINS in {game.turn_count} turns!")
    else:
        metrics["wins"][game.winner] += 1
        print(f"💀 AI Team LOST")
    print("*"*30)

    return {
        "metrics": metrics,
        "hallucinations": total_hallucinations,
        "illegal_clues": total_illegal_clues,
        "game": game
    }

if __name__ == "__main__":
    run_automated_game(game = None, spymaster = None, operative = None, metrics = None)