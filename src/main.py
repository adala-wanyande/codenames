from src.engine.game import CodenamesGame
from src.agents.llm import LLMSpymaster, LLMOperative

SAMPLE_VOCAB = [
    "Apple", "Beach", "Car", "Dog", "Elephant", "Frog", "Ghost", "Hat", "Ice", "Jacket",
    "Kite", "Lemon", "Moon", "Nut", "Ocean", "Piano", "Queen", "Rose", "Star", "Tree",
    "Umbrella", "Van", "Water", "Xylophone", "Yacht", "Zebra", "Airplane", "Bear", "Cat",
    "Dance", "Eagle", "Fire", "Gold", "Helicopter", "Island", "Jungle", "Kangaroo", "Lion",
    "Mountain", "Ninja", "Octopus", "Pirate", "Robot", "Spider", "Train", "Unicorn",
    "Vampire", "Whale", "Ninja", "Zombie"
]

def run_automated_game():
    print("🤖 STARTING AI vs AI CODENAMES MATCH 🤖\n")
    
    game = CodenamesGame(SAMPLE_VOCAB)
    spymaster = LLMSpymaster(name="Spymaster Qwen", model_type="ollama", model_name="qwen2.5")
    operative = LLMOperative(name="Operative Qwen", model_type="ollama", model_name="qwen2.5")

    # EIRINI'S REQUEST: Initial Spymaster Board Visualization
    print("👀 INITIAL SPYMASTER BOARD 👀")
    game.display(view="spymaster")

    while not game.is_game_over:
        game.turn_count += 1
        print(f"\n{'='*10} TURN {game.turn_count} {'='*10}")
        
        # --- SPYMASTER PHASE ---
        spymaster_board = game.get_spymaster_board()
        
        # EIRINI'S REQUEST: We no longer slice [:3]. We pass ALL unrevealed targets.
        unrevealed_targets = game.get_unrevealed_targets()
        print(f"Total targets left to connect: {len(unrevealed_targets)}")
        
        # The Spymaster agent now decides the clue AND the count
        clue, count = spymaster.give_clue(spymaster_board, unrevealed_targets)
        print(f"🎤 Spymaster says: '{clue}' for {count}")
        
        if not game.is_valid_clue(clue):
            print("❌ Spymaster gave an illegal clue! Skipping turn.")
            continue

        # --- OPERATIVE PHASE ---
        operative_board = game.get_operative_board()
        guessed_words = operative.guess_words(operative_board, clue, count)
        print(f"🤔 Operative guesses: {guessed_words}")
        
        for guess in guessed_words:
            print(f" -> Revealing '{guess}'...")
            identity, continue_turn, game_over = game.process_guess(guess)
            
            if identity:
                print(f"    Result: It's a [{identity}]!")
            else:
                print(f"    Result: Invalid word. AI hallucinated!")
                break
                
            if game_over:
                break
            if not continue_turn:
                print("    Turn ends early!")
                break

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