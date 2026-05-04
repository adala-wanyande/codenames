import pytest
import numpy as np
from unittest.mock import MagicMock, patch


VOCAB = ["apple", "tree", "dog", "bomb", "car", "fruit", "nature", "leaf", "random"]

TEST_BOARD = [
    {"word": "Apple", "identity": "Red",      "revealed": False},
    {"word": "Tree",  "identity": "Red",      "revealed": False},
    {"word": "Dog",   "identity": "Blue",     "revealed": False},
    {"word": "Bomb",  "identity": "Assassin", "revealed": False},
    {"word": "Car",   "identity": "Neutral",  "revealed": False},
]


def _make_mock_model():
    """Fake gensim KeyedVectors without any network download."""
    vecs = {w: np.eye(len(VOCAB))[i] for i, w in enumerate(VOCAB)}

    sims = {
        ("fruit", "apple"): 0.85,
        ("fruit", "tree"):  0.70,
        ("fruit", "dog"):   0.10,
        ("fruit", "bomb"):  0.05,
        ("fruit", "car"):   0.15,
        ("leaf",  "tree"):  0.80,
        ("leaf",  "apple"): 0.60,
    }

    model = MagicMock()
    model.index_to_key = VOCAB
    model.__contains__ = lambda self, key: key.lower() in VOCAB
    model.__getitem__  = lambda self, key: vecs.get(key.lower(), np.zeros(len(VOCAB)))
    model.similarity   = lambda a, b: sims.get((a.lower(), b.lower()), 0.20)
    model.similar_by_vector = lambda vec, topn=500: [
        ("fruit", 0.90), ("leaf", 0.75), ("nature", 0.50), ("dog", 0.30)
    ]
    return model


@pytest.fixture(autouse=True)
def patch_model(monkeypatch):
    mock = _make_mock_model()
    monkeypatch.setattr("src.agents.baseline._load_model", lambda *a, **k: mock)
    return mock


# ---------------------------------------------------------------------------
# Word2VecSpymaster
# ---------------------------------------------------------------------------

class TestWord2VecSpymaster:
    def test_give_clue_returns_tuple(self):
        from src.agents.baseline import Word2VecSpymaster
        sm = Word2VecSpymaster()
        result = sm.give_clue(TEST_BOARD, ["Apple", "Tree"])
        assert isinstance(result, tuple) and len(result) == 2

    def test_give_clue_returns_string_and_int(self):
        from src.agents.baseline import Word2VecSpymaster
        sm = Word2VecSpymaster()
        clue, count = sm.give_clue(TEST_BOARD, ["Apple", "Tree"])
        assert isinstance(clue, str)
        assert isinstance(count, int) and count >= 1

    def test_give_clue_skips_board_words(self):
        from src.agents.baseline import Word2VecSpymaster
        sm = Word2VecSpymaster()
        # similar_by_vector returns "dog" which is on the board — it should be skipped
        clue, _ = sm.give_clue(TEST_BOARD, ["Apple"])
        board_words_lower = {item["word"].lower() for item in TEST_BOARD}
        assert clue.lower() not in board_words_lower or clue == "random"

    def test_give_clue_skips_invalid_clue_memory(self):
        from src.agents.baseline import Word2VecSpymaster
        sm = Word2VecSpymaster()
        sm.invalid_clue.add("fruit")
        sm.invalid_clue.add("leaf")
        clue, _ = sm.give_clue(TEST_BOARD, ["Apple", "Tree"])
        assert clue.lower() not in {"fruit", "leaf"}

    def test_give_clue_fallback_when_no_target_vectors(self, patch_model):
        from src.agents.baseline import Word2VecSpymaster
        # Make the model not contain any target words
        patch_model.__contains__ = lambda self, key: False
        sm = Word2VecSpymaster()
        clue, count = sm.give_clue(TEST_BOARD, ["Apple", "Tree"])
        assert clue == "random"
        assert count == 1

    def test_give_clue_skips_non_alpha_candidates(self, patch_model):
        from src.agents.baseline import Word2VecSpymaster
        patch_model.similar_by_vector = lambda vec, topn=500: [
            ("has_underscore", 0.95), ("fruit", 0.90)
        ]
        sm = Word2VecSpymaster()
        clue, _ = sm.give_clue(TEST_BOARD, ["Apple"])
        assert "_" not in clue

    def test_count_is_at_least_one(self):
        from src.agents.baseline import Word2VecSpymaster
        sm = Word2VecSpymaster()
        _, count = sm.give_clue(TEST_BOARD, ["Apple"])
        assert count >= 1


# ---------------------------------------------------------------------------
# Word2VecOperative
# ---------------------------------------------------------------------------

class TestWord2VecOperative:
    def test_guess_returns_list(self):
        from src.agents.baseline import Word2VecOperative
        op = Word2VecOperative()
        result = op.guess_words(TEST_BOARD, "fruit", 2)
        assert isinstance(result, list)

    def test_guess_respects_num_guesses(self):
        from src.agents.baseline import Word2VecOperative
        op = Word2VecOperative()
        result = op.guess_words(TEST_BOARD, "fruit", 2)
        assert len(result) <= 2

    def test_guess_only_unrevealed_words(self):
        from src.agents.baseline import Word2VecOperative
        board = [
            {"word": "Apple", "identity": "Red",  "revealed": True},
            {"word": "Tree",  "identity": "Red",  "revealed": False},
            {"word": "Dog",   "identity": "Blue", "revealed": False},
        ]
        op = Word2VecOperative()
        result = op.guess_words(board, "fruit", 2)
        assert "Apple" not in result

    def test_guess_fallback_when_clue_not_in_model(self, patch_model):
        from src.agents.baseline import Word2VecOperative
        patch_model.__contains__ = lambda self, key: False
        op = Word2VecOperative()
        result = op.guess_words(TEST_BOARD, "unknownclue", 2)
        unrevealed = [item["word"] for item in TEST_BOARD if not item["revealed"]]
        assert result == unrevealed[:2]
