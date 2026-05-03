import pytest
from src.tournament import run_tournament


# ------------------------
# Helpers
# ------------------------

def fake_game_result():
    return {
        "metrics": {
            "turns": {"Red": 1, "Blue": 1},
            "invalid_clues": {"Red": 0, "Blue": 0},
            "total_guesses": {"Red": 2, "Blue": 1},
            "correct_guesses": {"Red": 2, "Blue": 1},
            "wrong_guesses": {"Red": 0, "Blue": 0},
            "assassin_hits": {"Red": 0, "Blue": 0},
            "hallucinations": {"Red": 0, "Blue": 0},
            "wins": {"Red": 1, "Blue": 0},
        },
        "hallucinations": 0,
        "illegal_clues": 0,
        "game": type("MockGame", (), {
            "winner": "Red",
            "turn_count": 5,
            "red_found": 9
        })()
    }

def make_dummy_game():
    return fake_game_result()

# ------------------------
# Tests
# ------------------------


def test_logger_receives_valid_structure(monkeypatch):
    captured = []

    monkeypatch.setattr(
        "src.main.run_automated_game",
        lambda *a, **k: fake_game_result()
    )

    class DummyLogger:
        def __init__(self): pass

        def log_match(self, **kwargs):
            captured.append(kwargs)

        def save_results(self): pass
        def print_summary(self): pass

    monkeypatch.setattr("src.tournament.TournamentLogger", DummyLogger)

    run_tournament(n_games=1)

    assert len(captured) == 6  # 6 configs

    for entry in captured:
        assert "match_id" in entry
        assert "game_engine" in entry
        assert "spymaster_type" in entry
        assert "shot" in entry


def test_logger_called_correct_number_of_times(monkeypatch):
    calls = {"count": 0}

    monkeypatch.setattr(
        "src.main.run_automated_game",
        lambda *a, **k: fake_game_result()
    )

    class DummyLogger:
        def __init__(self): pass

        def log_match(self, *args, **kwargs):
            calls["count"] += 1

        def save_results(self): pass
        def print_summary(self): pass

    monkeypatch.setattr("src.tournament.TournamentLogger", DummyLogger)

    run_tournament(n_games=2)

    # 6 configs × 2 games = 12
    assert calls["count"] == 12


def test_save_and_summary_called(monkeypatch):
    """Logger should save and print summary exactly once."""

    monkeypatch.setattr(
        "src.tournament.run_automated_game",
        lambda *args, **kwargs: fake_game_result()
    )

    calls = {"save": 0, "summary": 0}

    class DummyLogger:
        def __init__(self): pass

        def log_match(self, *a, **k): pass

        def save_results(self):
            calls["save"] += 1

        def print_summary(self):
            calls["summary"] += 1

    monkeypatch.setattr("src.tournament.TournamentLogger", DummyLogger)

    run_tournament(n_games=2)

    assert calls["save"] == 1
    assert calls["summary"] == 1


def test_zero_games(monkeypatch):

    monkeypatch.setattr(
        "src.main.run_automated_game",
        lambda *a, **k: fake_game_result()
    )

    class DummyLogger:
        def __init__(self): pass
        def log_match(self, *a, **k): pass
        def save_results(self): pass
        def print_summary(self): pass

    monkeypatch.setattr("src.tournament.TournamentLogger", DummyLogger)

    results = run_tournament(n_games=0)

    assert isinstance(results, dict)
    assert results == {}

        