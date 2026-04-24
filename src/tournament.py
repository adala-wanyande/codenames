from src.engine.game import CodenamesGame
from src.agents.llm import LLMSpymaster, LLMOperative
from .main import SAMPLE_VOCAB, run_automated_game
from src.utils.logger import TournamentLogger

def init_metrics():
    return {
        "turns": {"Red": 0, "Blue": 0},
        "invalid_clues": {"Red": 0, "Blue": 0},
        "total_guesses": {"Red": 0, "Blue": 0},
        "correct_guesses": {"Red": 0, "Blue": 0},
        "wrong_guesses": {"Red": 0, "Blue": 0},
        "assassin_hits": {"Red": 0, "Blue": 0},
        "hallucinations": {"Red": 0, "Blue": 0},
        "wins": {"Red": 0, "Blue": 0}
    }

def run_tournament(n_games=2):

    logger = TournamentLogger()
    aggregate = init_metrics()

    for i in range(n_games):
        print(f"🎮 Game {i+1}/{n_games}")

        game = CodenamesGame(SAMPLE_VOCAB)

        operative = LLMOperative(
            name="Operative",
            model_type="ollama",
            model_name="qwen2.5"
        )

        spymaster = LLMSpymaster(
            name="Spymaster",
            model_type="ollama",
            model_name="qwen2.5",
            operative=operative
        )

        result = run_automated_game(game, spymaster, operative, aggregate)

        game_metrics = result["metrics"]
        hallucinations = result["hallucinations"]
        illegal_clues = result["illegal_clues"]
        finished_game = result["game"]

        # ✅ LOG MATCH
        logger.log_match(
            match_id=i,
            spymaster_name=spymaster.name,
            operative_name=operative.name,
            game_engine=finished_game,
            hallucinations=hallucinations,
            illegal_clues=illegal_clues
        )

        # ✅ AGGREGATE
        for key in aggregate:
            for team in aggregate[key]:
                aggregate[key][team] += game_metrics[key][team]

    logger.save_results()
    logger.print_summary()

    return aggregate




if __name__ == "__main__":
    n_games = 2
    results = run_tournament(n_games=n_games)