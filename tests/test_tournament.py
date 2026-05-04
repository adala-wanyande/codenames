import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FAKE_BOARDS = [
    {
        "words": [f"W{i}" for i in range(25)],
        "identities": ["Red"]*9 + ["Blue"]*8 + ["Neutral"]*7 + ["Assassin"]*1,
    }
    for _ in range(10)
]


def _make_fake_game(winner="Red", turns=5):
    g = MagicMock()
    g.winner = winner
    g.turn_count = turns
    g.red_found = 9
    g.blue_found = 3
    g.is_game_over = True
    return g


def _fake_run_game(*args, **kwargs):
    return {"game": _make_fake_game()}


@pytest.fixture(autouse=True)
def patch_tournament_deps(monkeypatch):
    monkeypatch.setattr("src.tournament._load_benchmark_boards", lambda: FAKE_BOARDS)
    monkeypatch.setattr("src.tournament.run_automated_game", _fake_run_game)
    monkeypatch.setattr("src.tournament.build_operative", lambda **k: MagicMock(name="op"))
    monkeypatch.setattr("src.tournament.build_spymaster", lambda **k: MagicMock(name="sm"))


# ---------------------------------------------------------------------------
# log_match call count
# ---------------------------------------------------------------------------

def test_log_match_called_once_per_game(monkeypatch):
    from src.tournament import run_tournament, AGENT_CONFIGS
    calls = []

    class SpyLogger:
        def __init__(self): pass
        def log_match(self, **kwargs): calls.append(kwargs)
        def save_results(self): pass
        def print_summary(self): pass

    monkeypatch.setattr("src.tournament.TournamentLogger", SpyLogger)

    n_games = 2
    run_tournament(n_games=n_games)

    assert len(calls) == len(AGENT_CONFIGS) * n_games


def test_log_match_receives_required_fields(monkeypatch):
    from src.tournament import run_tournament

    captured = []

    class SpyLogger:
        def __init__(self): pass
        def log_match(self, **kwargs): captured.append(kwargs)
        def save_results(self): pass
        def print_summary(self): pass

    monkeypatch.setattr("src.tournament.TournamentLogger", SpyLogger)

    run_tournament(n_games=1)

    for entry in captured:
        assert "match_id" in entry
        assert "game_engine" in entry
        assert "spymaster_type" in entry
        assert "shot" in entry
        assert "model_name" in entry
        assert "temperature" in entry


# ---------------------------------------------------------------------------
# Incremental save_results
# ---------------------------------------------------------------------------

def test_save_results_called_after_every_game(monkeypatch):
    from src.tournament import run_tournament, AGENT_CONFIGS

    save_calls = {"count": 0}

    class SpyLogger:
        def __init__(self): pass
        def log_match(self, **kwargs): pass
        def save_results(self): save_calls["count"] += 1
        def print_summary(self): pass

    monkeypatch.setattr("src.tournament.TournamentLogger", SpyLogger)

    n_games = 2
    run_tournament(n_games=n_games)

    assert save_calls["count"] == len(AGENT_CONFIGS) * n_games


def test_print_summary_called_once(monkeypatch):
    from src.tournament import run_tournament

    summary_calls = {"count": 0}

    class SpyLogger:
        def __init__(self): pass
        def log_match(self, **kwargs): pass
        def save_results(self): pass
        def print_summary(self): summary_calls["count"] += 1

    monkeypatch.setattr("src.tournament.TournamentLogger", SpyLogger)

    run_tournament(n_games=1)

    assert summary_calls["count"] == 1


# ---------------------------------------------------------------------------
# Benchmark board usage
# ---------------------------------------------------------------------------

def test_uses_benchmark_boards(monkeypatch):
    from src.tournament import run_tournament

    boards_used = []

    def fake_run_game(*args, **kwargs):
        boards_used.append(kwargs.get("game"))
        return {"game": _make_fake_game()}

    monkeypatch.setattr("src.tournament.run_automated_game", fake_run_game)

    class QuietLogger:
        def __init__(self): pass
        def log_match(self, **kwargs): pass
        def save_results(self): pass
        def print_summary(self): pass

    monkeypatch.setattr("src.tournament.TournamentLogger", QuietLogger)

    run_tournament(n_games=1)

    assert len(boards_used) > 0


def test_n_games_capped_at_board_count(monkeypatch):
    from src.tournament import run_tournament, AGENT_CONFIGS

    monkeypatch.setattr("src.tournament._load_benchmark_boards", lambda: FAKE_BOARDS[:2])

    calls = {"count": 0}

    class SpyLogger:
        def __init__(self): pass
        def log_match(self, **kwargs): calls["count"] += 1
        def save_results(self): pass
        def print_summary(self): pass

    monkeypatch.setattr("src.tournament.TournamentLogger", SpyLogger)

    run_tournament(n_games=999)

    assert calls["count"] == len(AGENT_CONFIGS) * 2


# ---------------------------------------------------------------------------
# n_games = 0
# ---------------------------------------------------------------------------

def test_zero_games_runs_nothing(monkeypatch):
    from src.tournament import run_tournament

    calls = {"count": 0}

    class SpyLogger:
        def __init__(self): pass
        def log_match(self, **kwargs): calls["count"] += 1
        def save_results(self): pass
        def print_summary(self): pass

    monkeypatch.setattr("src.tournament.TournamentLogger", SpyLogger)

    run_tournament(n_games=0)

    assert calls["count"] == 0
