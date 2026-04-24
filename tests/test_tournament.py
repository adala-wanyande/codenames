import pytest
from src.tournament import run_tournament, init_metrics


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


# ------------------------
# Tests
# ------------------------

def test_no_double_counting(monkeypatch):
    """Ensure metrics are not double-counted across games."""

    def fake():
        return fake_game_result()

    monkeypatch.setattr("src.tournament.run_automated_game", lambda *a, **k: fake())

    class DummyLogger:
        def __init__(self): pass
        def log_match(self, *a, **k): pass
        def save_results(self): pass
        def print_summary(self): pass

    monkeypatch.setattr("src.tournament.TournamentLogger", DummyLogger)

    results = run_tournament(n_games=1)

    # Should be exactly 1, not 2
    assert results["wins"]["Red"] == 1

def test_run_tournament_aggregates_correctly(monkeypatch):
    """Tournament should correctly aggregate results across games."""

    # Mock run_automated_game
    monkeypatch.setattr(
        "src.tournament.run_automated_game",
        lambda *args, **kwargs: fake_game_result()
    )

    # Mock logger (no file writing)
    class DummyLogger:
        def __init__(self): pass
        def log_match(self, *a, **k): pass
        def save_results(self): pass
        def print_summary(self): pass

    monkeypatch.setattr("src.tournament.TournamentLogger", DummyLogger)

    results = run_tournament(n_games=3)

    # Each game gives Red 1 win → total 3
    assert results["wins"]["Red"] == 3
    assert results["wins"]["Blue"] == 0

    # Turns accumulate
    assert results["turns"]["Red"] == 3
    assert results["turns"]["Blue"] == 3

def test_logger_receives_correct_data(monkeypatch):
    captured = []

    monkeypatch.setattr(
        "src.tournament.run_automated_game",
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

    assert len(captured) == 1
    assert captured[0]["match_id"] == 0
    assert captured[0]["hallucinations"] == 0
    assert captured[0]["illegal_clues"] == 0
    assert captured[0]["game_engine"].winner == "Red"

def test_metrics_structure(monkeypatch):
    monkeypatch.setattr(
        "src.tournament.run_automated_game",
        lambda *a, **k: fake_game_result()
    )

    class DummyLogger:
        def __init__(self): pass
        def log_match(self, *a, **k): pass
        def save_results(self): pass
        def print_summary(self): pass

    monkeypatch.setattr("src.tournament.TournamentLogger", DummyLogger)

    results = run_tournament(n_games=1)

    expected_keys = {
        "turns", "invalid_clues", "total_guesses",
        "correct_guesses", "wrong_guesses",
        "assassin_hits", "hallucinations", "wins"
    }

    assert set(results.keys()) == expected_keys

def test_logger_called(monkeypatch):
    """Logger should be called once per game."""

    calls = {"count": 0}

    monkeypatch.setattr(
        "src.tournament.run_automated_game",
        lambda *args, **kwargs: fake_game_result()
    )

    class DummyLogger:
        def __init__(self): pass

        def log_match(self, *args, **kwargs):
            calls["count"] += 1

        def save_results(self): pass
        def print_summary(self): pass

    monkeypatch.setattr("src.tournament.TournamentLogger", DummyLogger)

    run_tournament(n_games=5)

    assert calls["count"] == 5


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
    """Running tournament with 0 games should not crash."""

    monkeypatch.setattr(
        "src.tournament.run_automated_game",
        lambda *args, **kwargs: fake_game_result()
    )

    class DummyLogger:
        def __init__(self): pass
        def log_match(self, *a, **k): pass
        def save_results(self): pass
        def print_summary(self): pass

    monkeypatch.setattr("src.tournament.TournamentLogger", DummyLogger)

    results = run_tournament(n_games=0)

    # Should be empty metrics
    assert results["wins"]["Red"] == 0
    assert results["wins"]["Blue"] == 0

def test_missing_fields_in_result(monkeypatch):
    """Tournament should fail clearly if result format is wrong."""

    monkeypatch.setattr(
        "src.tournament.run_automated_game",
        lambda *a, **k: {"metrics": {}}
    )

    class DummyLogger:
        def __init__(self): pass
        def log_match(self, *a, **k): pass
        def save_results(self): pass
        def print_summary(self): pass

    monkeypatch.setattr("src.tournament.TournamentLogger", DummyLogger)

    with pytest.raises(KeyError):
        run_tournament(n_games=1)
        