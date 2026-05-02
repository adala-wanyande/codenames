import random
from src.engine import game
from src.engine.game import CodenamesGame
from src.utils.logger import invalid_clues, TournamentLogger
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
    metrics=None,   # optional runtime helper only
    spymaster_type="single_cot",
    shot=1,
    logger=None
):
    print("🤖 STARTING AI vs AI CODENAMES MATCH 🤖\n")
    if game is None:
        game = CodenamesGame(SAMPLE_VOCAB)

    if operative is None:
        from src.agents.operative_llm import LLMOperative
        operative = LLMOperative(
            name="Operative Qwen",
            model_type="ollama",
            model_name="qwen2.5"
        )

    if spymaster is None:
        spymaster = build_spymaster(
            spymaster_type=spymaster_type,
            shot=shot,
            operative=operative
        )

    match_id = f"{spymaster_type}_{shot}_{id(game)}"

    print("👀 INITIAL SPYMASTER BOARD 👀")
    game.display(view="spymaster")
    print(f"🧠 Using spymaster: {spymaster_type} | shot={shot}")

    while not game.is_game_over:
        game.turn_count += 1
        print(f"\n{'='*10} TURN {game.turn_count} {'='*10}")

        team_color = game.current_team
        spymaster_board = game.get_spymaster_board()
        unrevealed_targets = game.get_unrevealed_targets(team_color)

        print(f"Team {team_color} has {len(unrevealed_targets)} targets left.")
        print(f"Unrevealed targets: {unrevealed_targets}")

        # --------------------------------
        # TURN RECORD (single source)
        # --------------------------------
        turn_record = {
            "match_id": match_id,
            "turn": game.turn_count,
            "team": team_color,
            "words_left": len(unrevealed_targets),
            "clue": None,
            "count": None,
            "guesses": [],
            "correct": 0,
            "wrong": 0,
            "assassin": 0,
            "invalid_clue": 0,
            "guess_trace": [],
            "spymaster_type": spymaster_type
        }

        # -------------------------
        # WIN CONDITION
        # -------------------------
        if len(unrevealed_targets) == 0:
            game.is_game_over = True
            game.winner = team_color
            print(f"🏁 {team_color} wins (no targets left)")

            if logger and hasattr(logger, "log_turn"):
                logger.log_turn(**turn_record)

            break

        # -------------------------
        # SPYMASTER
        # -------------------------
        if getattr(spymaster, "uses_internal_policy", False):
            result = spymaster.group_words(
                spymaster_board,
                unrevealed_targets
            )
        else:
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

        turn_record["clue"] = clue
        turn_record["count"] = count

        print(f"🎤 Spymaster says: {clue} for ({count})")

        # -------------------------
        # CLUE VALIDATION
        # -------------------------
        invalid_flag = int(not game.is_valid_clue(clue))
        turn_record["invalid_clue"] = invalid_flag

        if invalid_flag:
            spymaster.invalid_clue.add(clue)

            board_words = [item["word"] for item in spymaster_board]

            invalid_clues.log_invalid_clue(
                clue=clue,
                team=team_color,
                turn=game.turn_count,
                targets=target_words,
                board_words=board_words
            )

            print("❌ Illegal clue. Turn skipped.")

            if logger and hasattr(logger, "log_turn"):
                logger.log_turn(**turn_record)

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

        turn_record["guesses"] = guesses

        print(f"💡 Operative guesses: {guesses}")

        correct_team_hits = 0
        incorrect_team_hits = 0
        assassin_hit = 0
        false_positive = 0

        for guess in guesses[:count + 1]:
            print(f" -> Revealing '{guess}'...")

            identity, _, game_over = game.process_guess(
                guess,
                team_color
            )

            event = {
                "guess": guess,
                "identity": identity
            }
            turn_record["guess_trace"].append(event)

            print(f"    Result: {identity}")

            if identity == team_color:
                print("    ✅ Correct guess → can continue")
                correct_team_hits += 1
                turn_record["correct"] = correct_team_hits

                if guess not in target_words:
                    false_positive += 1

            elif identity is None:
                print("❌ Hallucination")
                break

            elif identity == "Assassin":
                opponent = "Blue" if team_color == "Red" else "Red"

                game.winner = opponent
                game.is_game_over = True

                assassin_hit += 1
                turn_record["assassin"] = 1

                print(f"💀 Assassin hit! {opponent} wins!")

                if logger and hasattr(logger, "log_turn"):
                    logger.log_turn(**turn_record)

                return {
                    "spymaster_type": spymaster_type,
                    "shot": shot,
                    "game": game
                }

            else:
                incorrect_team_hits += 1
                turn_record["wrong"] = incorrect_team_hits
                print("❌ Wrong guess → turn ends")
                break

            if game_over:
                break

        # -------------------------
        # ALWAYS LOG TURN
        # -------------------------
        if logger and hasattr(logger, "log_turn"):
            logger.log_turn(**turn_record)

        if game.is_game_over:
            break

        game.switch_team()

    print("\n" + "*" * 30)
    print("GAME OVER")
    print(f"The {game.winner} team wins in {game.turn_count} turns!")
    print("*" * 30)

    return {
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