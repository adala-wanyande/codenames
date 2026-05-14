import random
import json
import os

from src.engine.game import CodenamesGame
from src.utils.agent_config import build_spymaster, build_operative
from src.utils.logger import TournamentLogger
from .main import SAMPLE_VOCAB, run_automated_game

BENCHMARK_BOARDS_PATH = os.path.join("data", "benchmark_boards.json")

# ---------------------------------------------------------------------------
# Experiment matrix
#
# Each entry defines one agent configuration to evaluate.  Every config is
# tested against the same 100 benchmark boards so results are directly
# comparable (same board 42 for word2vec and qwen2.5 etc.).
#
# Temperature ablation: the last two rows run the best LLM strategy at
# temperature=0.0 (deterministic) vs temperature=0.7 (creative/risky).
# ---------------------------------------------------------------------------
AGENT_CONFIGS = [
    # ── Baseline ────────────────────────────────────────────────────────────
    {"spymaster_type": "word2vec",      "operative_type": "word2vec", "shot": 0, "model_name": "word2vec", "temperature": 0.0},

    # ── Single-CoT (qwen2.5) ────────────────────────────────────────────────
    {"spymaster_type": "single_cot",    "operative_type": "llm", "shot": 0, "model_name": "qwen2.5", "temperature": 0.0},
    {"spymaster_type": "single_cot",    "operative_type": "llm", "shot": 1, "model_name": "qwen2.5", "temperature": 0.0},

    # ── Double-CoT (qwen2.5) ────────────────────────────────────────────────
    {"spymaster_type": "double_cot",    "operative_type": "llm", "shot": 0, "model_name": "qwen2.5", "temperature": 0.0},
    {"spymaster_type": "double_cot",    "operative_type": "llm", "shot": 1, "model_name": "qwen2.5", "temperature": 0.0},

    # ── Double-CoT + Stochastic Rollout (qwen2.5) ───────────────────────────
    {"spymaster_type": "double_cot_SR", "operative_type": "llm", "shot": 0, "model_name": "qwen2.5", "temperature": 0.0},
    {"spymaster_type": "double_cot_SR", "operative_type": "llm", "shot": 1, "model_name": "qwen2.5", "temperature": 0.0},

    # ── Multi-model scaling (double_cot, shot=1) ─────────────────────────────
    {"spymaster_type": "double_cot",    "operative_type": "llm", "shot": 1, "model_name": "mistral",  "temperature": 0.0},
    {"spymaster_type": "double_cot",    "operative_type": "llm", "shot": 1, "model_name": "llama3",   "temperature": 0.0},
    {"spymaster_type": "double_cot",    "operative_type": "llm", "shot": 1, "model_name": "qwen3",    "temperature": 0.0},

    # ── Temperature ablation (double_cot, shot=1, qwen2.5) ──────────────────
    {"spymaster_type": "double_cot",    "operative_type": "llm", "shot": 1, "model_name": "qwen2.5", "temperature": 0.7},
]


def _load_benchmark_boards():
    with open(BENCHMARK_BOARDS_PATH, "r") as f:
        return json.load(f)


def run_tournament(n_games=10, compete="baseline"):
    """
    compete:
        - "baseline" -> experimental RED vs fixed Word2Vec BLUE
        - "similar"  -> RED vs BLUE use same agent family/config
    """

    random.seed(42)

    logger = TournamentLogger()

    boards = _load_benchmark_boards()

    n_games = min(n_games, len(boards))

    print(f"Using {n_games} benchmark boards for reproducible evaluation.")
    print(f"Competition mode: {compete}")

    for config in AGENT_CONFIGS:

        spymaster_type = config["spymaster_type"]

        operative_type = config.get(
            "operative_type",
            "llm"
        )

        shot = config["shot"]

        model_name = config.get(
            "model_name",
            "qwen2.5"
        )

        temperature = config.get(
            "temperature",
            0.0
        )

        print(
            f"\n🚀 Config: {spymaster_type} | "
            f"shot={shot} | "
            f"model={model_name} | "
            f"temp={temperature}"
        )

        for i in range(n_games):

            print(f"  🎮 Game {i + 1}/{n_games}")

            game = CodenamesGame(
                SAMPLE_VOCAB,
                board_data=boards[i]
            )

            # =====================================================
            # RED TEAM
            # =====================================================

            red_operative = build_operative(
                operative_type=operative_type,
                model_name=model_name,
                temperature=temperature
            )

            red_spymaster = build_spymaster(
                spymaster_type=spymaster_type,
                shot=shot,
                operative=red_operative,
                model_name=model_name,
                temperature=temperature
            )

            # =====================================================
            # BLUE TEAM
            # =====================================================

            if compete == "baseline":

                # ---------------------------------------------
                # Fixed Word2Vec baseline
                # ---------------------------------------------

                blue_operative_type = "word2vec"
                blue_model_name = "word2vec"
                blue_temperature = 0.0
                blue_spymaster_type = "word2vec"
                blue_shot = 0

            elif compete == "similar":

                # ---------------------------------------------
                # Same family/config as RED
                # ---------------------------------------------

                blue_operative_type = operative_type
                blue_model_name = model_name
                blue_temperature = temperature
                blue_spymaster_type = spymaster_type
                blue_shot = shot

            else:

                raise ValueError(
                    f"Unknown compete mode: {compete}"
                )

            blue_operative = build_operative(
                operative_type=blue_operative_type,
                model_name=blue_model_name,
                temperature=blue_temperature
            )

            blue_spymaster = build_spymaster(
                spymaster_type=blue_spymaster_type,
                shot=blue_shot,
                operative=blue_operative,
                model_name=blue_model_name,
                temperature=blue_temperature
            )

            # =====================================================
            # MATCH ID
            # =====================================================

            match_id = (
                f"{compete.upper()}"
                f"_RED_{type(red_spymaster).__name__}"
                f"_BLUE_{type(blue_spymaster).__name__}"
                f"_SHOT_{shot}"
                f"_{model_name}"
                f"_game{i}"
            )

            # =====================================================
            # RUN GAME
            # =====================================================

            result = run_automated_game(
                game=game,

                red_spymaster=red_spymaster,
                blue_spymaster=blue_spymaster,

                red_operative=red_operative,
                blue_operative=blue_operative,

                spymaster_type=spymaster_type,
                shot=shot,

                logger=logger,

                model_name=model_name,
                temperature=temperature,

                match_id=match_id
            )

            finished_game = result["game"]

            # =====================================================
            # LOG RESULTS
            # =====================================================

            logger.log_match(
                match_id=match_id,

                game_engine=finished_game,

                red_spymaster=type(red_spymaster).__name__,
                blue_spymaster=type(blue_spymaster).__name__,

                red_model=model_name,
                blue_model=blue_model_name,

                spymaster_type=spymaster_type,
                shot=shot,

                model_name=model_name,
                temperature=temperature
            )

            # Flush after every game
            logger.save_results()

    logger.print_summary()


if __name__ == "__main__":
    run_tournament(n_games=50)
