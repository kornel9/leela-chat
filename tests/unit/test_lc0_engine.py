"""Unit tests for ``Lc0Engine`` — mocked subprocess, runs locally without Lc0.

Verifies the wrapper's I/O-shaping logic: command construction, streaming
info-string capture, telemetry, error paths. The actual Lc0 binary is
exercised by ``tests/integration/test_lc0_engine.py`` on the pod.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import chess
import chess.engine
import pytest

from leela_chat.engine.lc0_engine import Lc0Engine, Lc0SearchResult


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def fake_paths(tmp_path: Path) -> tuple[Path, Path]:
    """Two real paths that exist on disk so FileNotFoundError doesn't fire."""
    binary = tmp_path / "lc0"
    weights = tmp_path / "BT4.pb.gz"
    binary.touch()
    weights.touch()
    return binary, weights


def _fake_info_lines() -> list[dict]:
    """Stand-in for what python-chess emits from a real Lc0 search.

    Lc0 (with ``--verbose-move-stats``) emits per-move candidate stats as
    ``info string ...`` lines. python-chess parses these into InfoDicts
    where the post-prefix content lands in ``info["string"]``. We
    interleave a few non-string info dicts to match real engine traffic.
    """
    return [
        {"depth": 1},                                      # ordinary info; ignored
        {"string": "e2e4 (N: 1234) (P: 15.20%) (Q: 0.0421)"},  # candidate 1
        {"string": "d2d4 (N: 980) (P: 12.30%) (Q: 0.0312)"},   # candidate 2
        {"string": "g1f3 (N: 750) (P: 11.10%) (Q: 0.0289)"},   # candidate 3
        {"nps": 12345},                                    # ordinary info; ignored
        {"string": "c2c4 (N: 600) (P: 9.80%) (Q: 0.0201)"},    # candidate 4
        # empty / non-string entries should be ignored
        {"string": ""},
        {"string": None},                                  # not actually emitted but defensive
    ]


def _build_mock_engine(
    info_lines: list[dict],
    bestmove: chess.Move | None = chess.Move.from_uci("e2e4"),
    engine_name: str = "Lc0 v0.32.1",
) -> MagicMock:
    """Construct a MagicMock standing in for ``SimpleEngine``.

    The mock's ``analysis()`` returns a context manager that, when iterated,
    yields ``info_lines`` in order and whose ``.wait()`` returns a
    ``BestMove(move=bestmove, ponder=None)``-shaped object.
    """
    engine = MagicMock(spec=chess.engine.SimpleEngine)
    engine.id = {"name": engine_name}

    analysis_cm = MagicMock()
    analysis_cm.__enter__.return_value = analysis_cm
    analysis_cm.__exit__.return_value = None
    analysis_cm.__iter__.return_value = iter(info_lines)

    best_obj = MagicMock()
    best_obj.move = bestmove
    analysis_cm.wait.return_value = best_obj

    engine.analysis.return_value = analysis_cm
    return engine


# --- constructor tests ------------------------------------------------------


def test_init_raises_when_binary_missing(tmp_path: Path) -> None:
    weights = tmp_path / "w.pb.gz"
    weights.touch()
    with pytest.raises(FileNotFoundError, match="binary"):
        Lc0Engine(binary=tmp_path / "missing-lc0", weights=weights)


def test_init_raises_when_weights_missing(tmp_path: Path) -> None:
    binary = tmp_path / "lc0"
    binary.touch()
    with pytest.raises(FileNotFoundError, match="weights"):
        Lc0Engine(binary=binary, weights=tmp_path / "missing.pb.gz")


def test_init_spawns_with_verbose_flag_and_weights(fake_paths: tuple[Path, Path]) -> None:
    binary, weights = fake_paths
    with patch.object(chess.engine.SimpleEngine, "popen_uci") as popen:
        popen.return_value = _build_mock_engine([])
        eng = Lc0Engine(binary=binary, weights=weights)
        try:
            popen.assert_called_once()
            command = popen.call_args.args[0]
            assert command[0] == str(binary)
            assert "--verbose-move-stats" in command
            assert f"--weights={weights}" in command
            assert eng.name == "Lc0 v0.32.1"
        finally:
            eng.close()


def test_init_appends_extra_args(fake_paths: tuple[Path, Path]) -> None:
    binary, weights = fake_paths
    with patch.object(chess.engine.SimpleEngine, "popen_uci") as popen:
        popen.return_value = _build_mock_engine([])
        eng = Lc0Engine(
            binary=binary, weights=weights,
            extra_args=["--backend=cuda-fp16", "--threads=2"],
        )
        try:
            command = popen.call_args.args[0]
            assert "--backend=cuda-fp16" in command
            assert "--threads=2" in command
        finally:
            eng.close()


# --- analyse_verbose tests --------------------------------------------------


def test_analyse_verbose_collects_only_string_payloads(fake_paths: tuple[Path, Path]) -> None:
    """Each info["string"] should land in info_string_lines; other info ignored."""
    binary, weights = fake_paths
    info_lines = _fake_info_lines()
    with patch.object(chess.engine.SimpleEngine, "popen_uci") as popen:
        popen.return_value = _build_mock_engine(info_lines)
        with Lc0Engine(binary=binary, weights=weights) as eng:
            board = chess.Board()
            result = eng.analyse_verbose(board, nodes=500)

    assert isinstance(result, Lc0SearchResult)
    # Four real candidate-stat strings; empty/None are filtered.
    assert len(result.info_string_lines) == 4
    assert result.info_string_lines[0].startswith("e2e4 ")
    assert "N: 980" in result.info_string_lines[1]


def test_analyse_verbose_reports_bestmove_and_complete(fake_paths: tuple[Path, Path]) -> None:
    binary, weights = fake_paths
    with patch.object(chess.engine.SimpleEngine, "popen_uci") as popen:
        popen.return_value = _build_mock_engine(
            _fake_info_lines(),
            bestmove=chess.Move.from_uci("d2d4"),
        )
        with Lc0Engine(binary=binary, weights=weights) as eng:
            result = eng.analyse_verbose(chess.Board(), nodes=500)

    assert result.bestmove_uci == "d2d4"
    assert result.search_complete is True
    assert result.nodes_requested == 500
    assert result.wall_seconds >= 0.0


def test_analyse_verbose_handles_missing_bestmove(fake_paths: tuple[Path, Path]) -> None:
    """If the engine returns no bestmove (search interrupted / error),
    Lc0SearchResult flags search_complete=False but still surfaces lines."""
    binary, weights = fake_paths
    with patch.object(chess.engine.SimpleEngine, "popen_uci") as popen:
        popen.return_value = _build_mock_engine(
            _fake_info_lines(),
            bestmove=None,
        )
        with Lc0Engine(binary=binary, weights=weights) as eng:
            result = eng.analyse_verbose(chess.Board(), nodes=500)

    assert result.bestmove_uci is None
    assert result.search_complete is False
    # Captured lines preserved even on incomplete search.
    assert len(result.info_string_lines) == 4


def test_analyse_verbose_tolerates_wait_engine_error(fake_paths: tuple[Path, Path]) -> None:
    """If ``analysis.wait()`` raises EngineError, treat as interrupted."""
    binary, weights = fake_paths
    with patch.object(chess.engine.SimpleEngine, "popen_uci") as popen:
        mock_engine = _build_mock_engine(_fake_info_lines())
        mock_engine.analysis.return_value.wait.side_effect = chess.engine.EngineError("boom")
        popen.return_value = mock_engine
        with Lc0Engine(binary=binary, weights=weights) as eng:
            result = eng.analyse_verbose(chess.Board(), nodes=500)

    assert result.bestmove_uci is None
    assert result.search_complete is False
    assert len(result.info_string_lines) == 4


def test_analyse_verbose_passes_node_limit_to_engine(fake_paths: tuple[Path, Path]) -> None:
    """The node budget must reach engine.analysis(..., Limit(nodes=N))."""
    binary, weights = fake_paths
    with patch.object(chess.engine.SimpleEngine, "popen_uci") as popen:
        mock_engine = _build_mock_engine([])
        popen.return_value = mock_engine
        with Lc0Engine(binary=binary, weights=weights) as eng:
            eng.analyse_verbose(chess.Board(), nodes=12345)

    args, kwargs = mock_engine.analysis.call_args
    limit = args[1] if len(args) >= 2 else kwargs["limit"]
    assert limit.nodes == 12345


# --- lifecycle tests --------------------------------------------------------


def test_context_manager_closes_engine(fake_paths: tuple[Path, Path]) -> None:
    binary, weights = fake_paths
    with patch.object(chess.engine.SimpleEngine, "popen_uci") as popen:
        mock_engine = _build_mock_engine([])
        popen.return_value = mock_engine
        with Lc0Engine(binary=binary, weights=weights):
            pass
    mock_engine.quit.assert_called_once()


def test_close_tolerates_engine_error_on_quit(fake_paths: tuple[Path, Path]) -> None:
    """quit() raising EngineError should not propagate — closing is best-effort."""
    binary, weights = fake_paths
    with patch.object(chess.engine.SimpleEngine, "popen_uci") as popen:
        mock_engine = _build_mock_engine([])
        mock_engine.quit.side_effect = chess.engine.EngineError("already dead")
        popen.return_value = mock_engine
        eng = Lc0Engine(binary=binary, weights=weights)
        eng.close()  # must not raise
