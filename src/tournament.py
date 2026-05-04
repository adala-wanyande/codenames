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


def run_tournament(n_games=10):
    random.seed(42)
    logger = TournamentLogger()

    boards = _load_benchmark_boards()
    n_games = min(n_games, len(boards))
    print(f"Using {n_games} benchmark boards for reproducible evaluation.")

    for config in AGENT_CONFIGS:
        spymaster_type = config["spymaster_type"]
        operative_type = config.get("operative_type", "llm")
        shot           = config["shot"]
        model_name     = config.get("model_name", "qwen2.5")
        temperature    = config.get("temperature", 0.0)

        print(f"\n🚀 Config: {spymaster_type} | shot={shot} | model={model_name} | temp={temperature}")

        for i in range(n_games):
            print(f"  🎮 Game {i + 1}/{n_games}")

            game = CodenamesGame(SAMPLE_VOCAB, board_data=boards[i])

            operative = build_operative(
                operative_type=operative_type,
                model_name=model_name,
                temperature=temperature
            )
            spymaster = build_spymaster(
                spymaster_type=spymaster_type,
                shot=shot,
                operative=operative,
                model_name=model_name,
                temperature=temperature
            )

            match_id = f"{spymaster_type}_{shot}_{model_name}_{i}"

            result = run_automated_game(
                game=game,
                spymaster=spymaster,
                operative=operative,
                spymaster_type=spymaster_type,
                shot=shot,
                logger=logger,
                model_name=model_name,
                temperature=temperature
            )

            finished_game = result["game"]

            logger.log_match(
                match_id=match_id,
                spymaster_name=spymaster.name,
                operative_name=operative.name,
                game_engine=finished_game,
                spymaster_type=spymaster_type,
                shot=shot,
                model_name=model_name,
                temperature=temperature
            )
            logger.save_results()  # flush after every game so kills don't lose data

    logger.print_summary()


if __name__ == "__main__":
    run_tournament(n_games=3)
