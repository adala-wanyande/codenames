from src.engine.game import CodenamesGame
from src.agents.operative_llm import LLMOperative
from src.utils.agent_config import build_spymaster
from src.utils.logger import TournamentLogger
from .main import SAMPLE_VOCAB, run_automated_game

AGENT_CONFIGS = [
    {"spymaster_type": "single_cot", "shot": 0},
    {"spymaster_type": "single_cot", "shot": 1},
    {"spymaster_type": "double_cot", "shot": 0},
    {"spymaster_type": "double_cot", "shot": 1},
    {"spymaster_type": "double_cot_SR", "shot": 0},
    {"spymaster_type": "double_cot_SR", "shot": 1},
]


def run_tournament(n_games=10):
    logger = TournamentLogger()

    all_results = {}

    for config in AGENT_CONFIGS:
        spymaster_type = config["spymaster_type"]
        shot = config["shot"]

        print(f"\n🚀 Running config: {spymaster_type} | shot={shot}")


        for i in range(n_games):
            print(f"🎮 Game {i+1}/{n_games}")

            game = CodenamesGame(SAMPLE_VOCAB)

            operative = LLMOperative(
                name="Operative",
                model_type="ollama",
                model_name="qwen2.5"
            )

            spymaster = build_spymaster(
                spymaster_type=spymaster_type,
                shot=shot,
                operative=operative
            )

            result = run_automated_game(
                game=game,
                spymaster=spymaster,
                operative=operative,
                spymaster_type=spymaster_type,
                shot=shot,
                logger=logger
            )
            game_obj = result.get("game")
            if (
                game_obj is not None
                and getattr(game_obj, "is_game_over", False)
                and getattr(game_obj, "winner", None) == "Assassin"
            ):
                print("skip")
                break
            finished_game = result["game"]

            # ✅ LOG MATCH
            logger.log_match(
                match_id=f"{spymaster_type}_{shot}_{i}",
                spymaster_name=spymaster.name,
                operative_name=operative.name,
                game_engine=finished_game,
                spymaster_type=spymaster_type,
                shot=shot
            )

         
    logger.save_results()
    logger.print_summary()

    return all_results

if __name__ == "__main__":
    n_games = 10
    results = run_tournament(n_games=n_games)