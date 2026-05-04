import pytest
from src.utils.logger import TournamentLogger


# ---------------------------------------------------------------------------
# _prompt_strategy
# ---------------------------------------------------------------------------

class TestPromptStrategy:
    @pytest.mark.parametrize("spymaster_type,shot,expected", [
        ("word2vec",      0, "Baseline"),
        ("single_cot",    0, "Zero-Shot"),
        ("single_cot",    1, "Few-Shot"),
        ("double_cot",    0, "CoT"),
        ("double_cot",    1, "CoT+Few-Shot"),
        ("double_cot_SR", 0, "SR-CoT"),
        ("double_cot_SR", 1, "SR-CoT+Few-Shot"),
    ])
    def test_known_mappings(self, spymaster_type, shot, expected):
        assert TournamentLogger._prompt_strategy(spymaster_type, shot) == expected

    def test_unknown_returns_fallback_string(self):
        result = TournamentLogger._prompt_strategy("unknown_type", 3)
        assert "unknown_type" in result
        assert "3" in result


# ---------------------------------------------------------------------------
# _model_label
# ---------------------------------------------------------------------------

class TestModelLabel:
    @pytest.mark.parametrize("model_name,expected", [
        ("word2vec", "Word2Vec"),
        ("qwen2.5",  "Qwen-2.5"),
        ("qwen3",    "Qwen-3"),
        ("mistral",  "Mistral"),
        ("llama3",   "LLaMA-3"),
    ])
    def test_known_labels(self, model_name, expected):
        assert TournamentLogger._model_label(model_name) == expected

    def test_unknown_model_returns_name_unchanged(self):
        assert TournamentLogger._model_label("my_custom_model") == "my_custom_model"


# ---------------------------------------------------------------------------
# log_turn
# ---------------------------------------------------------------------------

class TestLogTurn:
    def _base_kwargs(self):
        return dict(
            match_id="test_match_1",
            turn=1,
            team="Red",
            words_left=9,
            clue="fruit",
            count=2,
            guesses=["Apple", "Tree"],
            correct=2,
            wrong=0,
            assassin=0,
            invalid_clue=0,
            guess_trace=[],
            spymaster_type="double_cot",
            shot=1,
            model_name="qwen2.5",
            temperature=0.0,
        )

    def test_log_turn_appends_record(self):
        logger = TournamentLogger()
        logger.log_turn(**self._base_kwargs())
        assert len(logger.turn_data) == 1

    def test_log_turn_prompt_strategy_column(self):
        logger = TournamentLogger()
        logger.log_turn(**self._base_kwargs())
        assert logger.turn_data[0]["Prompt_Strategy"] == "CoT+Few-Shot"

    def test_log_turn_model_label_column(self):
        logger = TournamentLogger()
        logger.log_turn(**self._base_kwargs())
        assert logger.turn_data[0]["Model_Label"] == "Qwen-2.5"

    def test_log_turn_shot_column(self):
        logger = TournamentLogger()
        logger.log_turn(**self._base_kwargs())
        assert logger.turn_data[0]["Shot"] == 1

    def test_log_turn_temperature_column(self):
        logger = TournamentLogger()
        logger.log_turn(**self._base_kwargs())
        assert logger.turn_data[0]["Temperature"] == 0.0

    def test_log_turn_fallback_flag_random_clue(self):
        logger = TournamentLogger()
        kwargs = self._base_kwargs()
        kwargs["clue"] = "random"
        logger.log_turn(**kwargs)
        assert logger.turn_data[0]["Fallback"] == 1

    def test_log_turn_fallback_flag_normal_clue(self):
        logger = TournamentLogger()
        logger.log_turn(**self._base_kwargs())
        assert logger.turn_data[0]["Fallback"] == 0

    def test_log_turn_guesses_count_from_list(self):
        logger = TournamentLogger()
        logger.log_turn(**self._base_kwargs())
        assert logger.turn_data[0]["Guesses"] == 2


# ---------------------------------------------------------------------------
# log_match
# ---------------------------------------------------------------------------

class TestLogMatch:
    def _make_game(self, winner="Red", turns=8):
        class FakeGame:
            pass
        g = FakeGame()
        g.winner = winner
        g.turn_count = turns
        g.red_found = 9
        g.blue_found = 3
        return g

    def test_log_match_appends_record(self):
        logger = TournamentLogger()
        logger.log_match("m1", "SM", "OP", self._make_game(),
                         spymaster_type="double_cot", shot=1,
                         model_name="qwen2.5", temperature=0.0)
        assert len(logger.match_data) == 1

    def test_log_match_prompt_strategy_column(self):
        logger = TournamentLogger()
        logger.log_match("m1", "SM", "OP", self._make_game(),
                         spymaster_type="single_cot", shot=0,
                         model_name="qwen2.5", temperature=0.0)
        assert logger.match_data[0]["Prompt_Strategy"] == "Zero-Shot"

    def test_log_match_model_label_column(self):
        logger = TournamentLogger()
        logger.log_match("m1", "SM", "OP", self._make_game(),
                         spymaster_type="double_cot", shot=1,
                         model_name="mistral", temperature=0.0)
        assert logger.match_data[0]["Model_Label"] == "Mistral"

    def test_log_match_winner_recorded(self):
        logger = TournamentLogger()
        logger.log_match("m1", "SM", "OP", self._make_game(winner="Blue"),
                         spymaster_type="double_cot", shot=1)
        assert logger.match_data[0]["Win"] == "Blue"

    def test_log_match_turns_recorded(self):
        logger = TournamentLogger()
        logger.log_match("m1", "SM", "OP", self._make_game(turns=12),
                         spymaster_type="double_cot", shot=1)
        assert logger.match_data[0]["Turns_Taken"] == 12
