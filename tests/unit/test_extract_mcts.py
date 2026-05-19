"""Unit tests for ``extract_mcts`` — mocked Lc0Engine; runs locally.

The extractor is glue (`analyse_verbose` → `parse_verbose_move_stats`).
These tests verify the wiring: that the right board/nodes flow into the
engine call, the engine's output gets fed through the parser, and the
parser's findings come back unchanged. The end-to-end behaviour against
real Lc0 is in ``tests/integration/test_extract_mcts_integration.py``.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import chess

from leela_chat.engine.lc0_engine import Lc0SearchResult
from leela_chat.extractors.mcts import extract_mcts

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "lc0"


def _fake_engine_returning(fixture_name: str, bestmove_uci: str | None) -> MagicMock:
    """Build a mock Lc0Engine that returns the given frozen fixture."""
    text = (FIXTURES_DIR / f"{fixture_name}.txt").read_text().splitlines()
    info_lines = [line for line in text if line and not line.startswith("#")]
    eng = MagicMock()
    eng.analyse_verbose.return_value = Lc0SearchResult(
        info_string_lines=info_lines,
        bestmove_uci=bestmove_uci,
        nodes_requested=10000,
        wall_seconds=0.42,
        search_complete=bestmove_uci is not None,
    )
    return eng


def test_extract_mcts_happy_path() -> None:
    """Starting position fixture → MCTSFindings with d4 as bestmove."""
    eng = _fake_engine_returning("starting_position", bestmove_uci="d2d4")
    board = chess.Board()

    findings = extract_mcts(board, engine=eng, nodes=500)

    assert findings.bestmove_uci == "d2d4"
    assert findings.bestmove_san == "d4"
    assert findings.search_complete is True
    assert findings.root_stats is not None
    assert len(findings.children) == 20  # 20 legal moves from start
    # Children sorted by visits desc (invariant from the parser).
    visits = [c.visits for c in findings.children]
    assert visits == sorted(visits, reverse=True)


def test_extract_mcts_passes_nodes_through() -> None:
    """The nodes kwarg must reach Lc0Engine.analyse_verbose unchanged."""
    eng = _fake_engine_returning("starting_position", bestmove_uci="d2d4")
    extract_mcts(chess.Board(), engine=eng, nodes=12345)

    eng.analyse_verbose.assert_called_once()
    _, kwargs = eng.analyse_verbose.call_args
    assert kwargs["nodes"] == 12345


def test_extract_mcts_passes_board_through() -> None:
    """The board kwarg must reach Lc0Engine.analyse_verbose unchanged.

    Critical because the parser uses the same board for UCI→SAN
    translation. If the extractor accidentally substituted a different
    board, SAN strings would be wrong (or python-chess would raise on
    illegal moves)."""
    eng = _fake_engine_returning(
        "walkthrough_rxd4", bestmove_uci="d8d4",
    )
    board = chess.Board("1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1")

    findings = extract_mcts(board, engine=eng, nodes=2000)

    # SAN must be correct for this board, not for a default starting position.
    assert findings.bestmove_san == "Rxd4"
    # And the engine got the exact board object we passed in.
    args, _ = eng.analyse_verbose.call_args
    assert args[0] is board


def test_extract_mcts_default_nodes() -> None:
    """Default node budget is 10000 (per Phase 2 plan / handler defaults)."""
    eng = _fake_engine_returning("starting_position", bestmove_uci="d2d4")
    extract_mcts(chess.Board(), engine=eng)

    _, kwargs = eng.analyse_verbose.call_args
    assert kwargs["nodes"] == 10000


def test_extract_mcts_propagates_search_incomplete() -> None:
    """If the engine reports search_complete=False (interrupted), the
    findings must surface the flag — handlers/LLM rely on it to avoid
    claiming authoritative verdicts on partial data."""
    eng = MagicMock()
    eng.analyse_verbose.return_value = Lc0SearchResult(
        info_string_lines=[],  # no lines emitted before interruption
        bestmove_uci=None,
        nodes_requested=10000,
        wall_seconds=0.05,
        search_complete=False,
    )

    findings = extract_mcts(chess.Board(), engine=eng)

    assert findings.search_complete is False
    assert findings.bestmove_uci is None
    assert findings.bestmove_san is None
    assert findings.children == []
    assert findings.root_stats is None
