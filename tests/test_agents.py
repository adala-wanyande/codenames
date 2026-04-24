import pytest

from src.agents.llm import LLMSpymaster, LLMOperative


TEST_BOARD = [
    {"word": "Apple", "identity": "Red", "revealed": False},
    {"word": "Tree", "identity": "Red", "revealed": False},
    {"word": "Dog", "identity": "Blue", "revealed": False},
    {"word": "Bomb", "identity": "Assassin", "revealed": False},
]


# ------------------------
# Spymaster tests
# ------------------------
import sys
import types

# Fake ollama module so imports don't crash
sys.modules["ollama"] = types.ModuleType("ollama")
sys.modules["ollama"].chat = lambda *args, **kwargs: {
    "message": {"content": "mock"}
}


def test_spymaster_returns_valid_structure(monkeypatch):
    """Spymaster should return (clue, count, combo) in correct format."""

    spymaster = LLMSpymaster("Test")

    # Mock internals → deterministic behavior
    monkeypatch.setattr(spymaster, "propose_groups", lambda *args, **kwargs: [("Apple", "Tree")])
    monkeypatch.setattr(spymaster, "give_clue", lambda combo, r, board, simulation=True: ("fruit", r))
    monkeypatch.setattr(
        spymaster,
        "simulate_clue",
        lambda *args, **kwargs: {"correct": 2, "wrong": 0, "assassin": False, "hallucinations": 0}
    )

    clue, count, combo = spymaster.group_words(TEST_BOARD, ["Apple", "Tree"])

    assert isinstance(clue, str)
    assert isinstance(count, int)
    assert isinstance(combo, tuple)

    assert clue == "fruit"
    assert count == 2
    assert combo == ("Apple", "Tree")


def test_spymaster_avoids_forbidden_words(monkeypatch):
    """Clue identical to board word should be penalized."""

    spymaster = LLMSpymaster("Test")

    monkeypatch.setattr(spymaster, "propose_groups", lambda *args, **kwargs: [("Apple",)])
    monkeypatch.setattr(spymaster, "give_clue", lambda *args, **kwargs: ("Dog", 1))  # forbidden

    monkeypatch.setattr(
        spymaster,
        "simulate_clue",
        lambda *args, **kwargs: {"correct": 1, "wrong": 0, "assassin": False, "hallucinations": 0}
    )

    clue, count, combo = spymaster.group_words(TEST_BOARD, ["Apple"])

    # Should fallback or avoid bad clue
    assert clue != "Dog"


# ------------------------
# Operative tests
# ------------------------

def test_operative_returns_only_valid_words(monkeypatch):
    """Operative should only return words from board."""

    operative = LLMOperative("Test")

    fake_output = "Apple, INVALID, Tree"

    def mock_chat(*args, **kwargs):
        return {"message": {"content": fake_output}}

    monkeypatch.setattr("src.agents.llm.ollama.chat", mock_chat)

    board = [
        {"word": "Apple", "revealed": False},
        {"word": "Tree", "revealed": False},
        {"word": "Dog", "revealed": False},
    ]

    guesses = operative.guess_words(board, "fruit", 2)

    assert guesses == ["Apple", "Tree"]


# ------------------------
# Simulation behavior
# ------------------------

def test_simulation_counts_correct(monkeypatch):
    """simulate_clue should correctly count correct guesses."""

    spymaster = LLMSpymaster("Test")

    def mock_guess_words(*args, **kwargs):
        return ["Apple", "Tree"]

    monkeypatch.setattr(spymaster.operative, "guess_words", mock_guess_words)

    result = spymaster.simulate_clue("fruit", 2, TEST_BOARD)

    assert result["correct"] == 2
    assert result["wrong"] == 0


def test_simulation_detects_assassin(monkeypatch):
    """simulate_clue should detect assassin hit."""

    spymaster = LLMSpymaster("Test")

    def mock_guess_words(*args, **kwargs):
        return ["Bomb"]

    monkeypatch.setattr(spymaster.operative, "guess_words", mock_guess_words)

    result = spymaster.simulate_clue("danger", 1, TEST_BOARD)

    assert result["assassin"] is True


# ------------------------
# Integration-style test
# ------------------------

def test_spymaster_and_operative_interaction(monkeypatch):
    """End-to-end: clue → guesses → simulation."""

    spymaster = LLMSpymaster("Test")
    operative = spymaster.operative

    # Fix everything deterministically
    monkeypatch.setattr(spymaster, "propose_groups", lambda *args, **kwargs: [("Apple", "Tree")])
    monkeypatch.setattr(spymaster, "give_clue", lambda *args, **kwargs: ("fruit", 2))

    def mock_guess_words(*args, **kwargs):
        return ["Apple", "Tree"]

    monkeypatch.setattr(operative, "guess_words", mock_guess_words)

    clue, count, combo = spymaster.group_words(TEST_BOARD, ["Apple", "Tree"])
    result = spymaster.simulate_clue(clue, count, TEST_BOARD)

    assert clue == "fruit"
    assert result["correct"] == 2

def test_group_words_fallback(monkeypatch):
    spymaster = LLMSpymaster("Test")

    # no groups
    monkeypatch.setattr(spymaster, "propose_groups", lambda *a, **k: [])

    # force give_clue to fail
    monkeypatch.setattr(spymaster, "give_clue", lambda *a, **k: ("ERROR", 0))

    clue, count, combo = spymaster.group_words(TEST_BOARD, ["Apple"])

    assert clue == "random"
    assert count == 1

def test_clue_cache(monkeypatch):
    spymaster = LLMSpymaster("Test")

    monkeypatch.setattr(spymaster, "propose_groups", lambda *a, **k: [("Apple",)])
    monkeypatch.setattr(spymaster, "give_clue", lambda *a, **k: ("fruit", 1))

    spymaster.group_words(TEST_BOARD, ["Apple"])
    first_cache_size = len(spymaster.clue_cache)

    spymaster.group_words(TEST_BOARD, ["Apple"])

    assert len(spymaster.clue_cache) == first_cache_size


def test_operational_cleanup(monkeypatch):
    operative = LLMOperative("Test")

    monkeypatch.setattr("src.agents.llm.ollama.chat", lambda *a, **k: {"message": {"content": "Apple!!!, Tree??, Dog"}})

    board = [
        {"word": "Apple", "revealed": False},
        {"word": "Tree", "revealed": False},
        {"word": "Dog", "revealed": False},
    ]

    guesses = operative.guess_words(board, "fruit", 3)

    assert "Apple" in guesses
    assert "Tree" in guesses