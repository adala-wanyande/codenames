import random
from src.engine import game
from src.engine.game import CodenamesGame
from src.utils.logger import invalid_clues
from src.utils.agent_config import build_spymaster
import argparse



SAMPLE_VOCAB = [
    "Apple", "Beach", "Car", "Dog", "Elephant", "Frog", "Ghost", "Hat", "Ice", "Jacket",
    "Kite", "Lemon", "Moon", "Nut", "Ocean", "Piano", "Queen", "Rose", "Star", "Tree",
    "Umbrella", "Van", "Water", "Xylophone", "Yacht", "Zebra", "Airplane", "Bear", "Cat",
    "Dance", "Eagle", "Fire", "Gold", "Helicopter", "Island", "Jungle", "Kangaroo", "Lion",
    "Mountain", "Ninja", "Octopus", "Pirate", "Robot", "Spider", "Train", "Unicorn",
    "Vampire", "Whale", "Zombie"
]

def run_automated_game(
    game=None,
    spymaster=None,
    operative=None,
    metrics=None,
    spymaster_type="single_cot",
    shot=1
):
    print("🤖 STARTING AI vs AI CODENAMES MATCH 🤖\n")

    if game is None:
        from src.tournament import init_metrics
        metrics = init_metrics()
        game = CodenamesGame(SAMPLE_VOCAB)

    if operative is None:
        from src.agents.operative_llm import LLMOperative
        operative = LLMOperative(
            name="Operative Qwen",
            model_type="ollama",
            model_name="qwen2.5"
        )

    # -------------------------
    # BUILD SPYMASTER CLEANLY
    # -------------------------
    if spymaster is None:
        spymaster = build_spymaster(
            spymaster_type=spymaster_type,
            shot=shot,
            operative=operative
        )

    print("👀 INITIAL SPYMASTER BOARD 👀")
    game.display(view="spymaster")

    print(f"🧠 Using spymaster: {spymaster_type} | shot={shot}")

    # =========================
    # GAME LOOP
    # =========================
    while not game.is_game_over:
        game.turn_count += 1
        print(f"\n{'='*10} TURN {game.turn_count} {'='*10}")

        team_color = game.current_team
        metrics["turns"][team_color] += 1

        spymaster_board = game.get_spymaster_board()
        unrevealed_targets = game.get_unrevealed_targets(team_color)
        print(f"Team {team_color} has {len(unrevealed_targets)} targets left.")
        print(f"Unrevealed targets: {unrevealed_targets}")
        if len(unrevealed_targets) == 0:
            game.is_game_over = True
            game.winner = team_color
            print(f"🏁 {team_color} wins (no targets left)")
            break


        # -------------------------
        # SPYMASTER CALL
        # -------------------------
        if getattr(spymaster, "uses_internal_policy", False):
            # ✅ SR agent controls EVERYTHING internally
            result = spymaster.group_words(spymaster_board, unrevealed_targets)
            

            if isinstance(result, tuple) and len(result) == 3:
                clue, count, target_words = result
            else:
                clue, count = "random", 1
                target_words = unrevealed_targets[:1]

        else:
            # ✅ Standard agents (single + double CoT)
            result = spymaster.give_clue(
                spymaster_board,
                unrevealed_targets,
                shot=shot
            )

            if isinstance(result, tuple):
                if len(result) == 3:
                    clue, count, target_words = result
                elif len(result) == 2:
                    clue, count = result
                    target_words = unrevealed_targets[:count]
                else:
                    clue, count = "random", 1
                    target_words = unrevealed_targets[:1]
            else:
                clue, count = "random", 1
                target_words = unrevealed_targets[:1]

        print(f"🎤 Spymaster says: {clue} for ({count}). Targets:{target_words}")

        # -------------------------
        # VALIDATION
        # ------------------------
        if not game.is_valid_clue(clue):
            spymaster.invalid_clue.add(clue)
            metrics["invalid_clues"][team_color] += 1
            board_words = [item["word"] for item in spymaster_board]

            invalid_clues.log_invalid_clue(
                clue=clue,
                team=team_color,
                turn=game.turn_count,
                targets=target_words,
                board_words=board_words
            )

            print("❌ Illegal clue. Turn skipped.")
            game.switch_team()
            continue

        # -------------------------
        # OPERATIVE
        # -------------------------
        operative_board = game.get_operative_board()
        guesses = operative.guess_words(operative_board, clue, count)

        if not guesses:
            available = [
                w["word"] for w in operative_board if not w["revealed"]
            ]
            guesses = [random.choice(available)]

        print(f"🤔  Operative thinking on '{clue}' ({count} guesses allowed) ")
        print(f"💡 Operative guesses: {guesses}")

        for guess in guesses[:count + 1]:
            print(f" -> Revealing '{guess}'...")
            identity, _, game_over = game.process_guess(guess, team_color)
            print(f"    Result: {identity}")
            if identity == team_color:
                print("    ✅ Correct guess → can continue")
            if identity is None:
                metrics["hallucinations"][team_color] += 1
                print("    ❌ Invalid word (hallucination)")
                break

            if identity == "Assassin":
                metrics["assassin_hits"][team_color] += 1
                print("    💀 Assassin hit!")
                break

            if identity != team_color:
                print("    ❌ Wrong guess → turn ends")
                break

            if game_over:
                break

        if game.is_game_over:
            break

        game.switch_team()

    # =========================
    # END
    # =========================
    print("\n" + "*" * 30)
    print("GAME OVER")
    print(f"The {game.winner} team wins in {game.turn_count} turns!")
    print(f"Winner: {game.winner}")
    print("*" * 30)

    return {
        "metrics": metrics,
        "spymaster_type": spymaster_type,
        "shot": shot,
        "game": game
    }
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Codenames AI simulation")

    parser.add_argument(
        "--spymaster_type",
        type=str,
        default="single_cot",
        choices=["double_cot_SR", "single_cot", "double_cot"]
    )

    parser.add_argument(
        "--shot",
        type=int,
        default=1,
        choices=[0, 1, 2]
    )

    args = parser.parse_args()
    run_automated_game(
        game=None,
        spymaster=None,
        operative=None,
        metrics=None,
        spymaster_type=args.spymaster_type,
        shot=args.shot
    )