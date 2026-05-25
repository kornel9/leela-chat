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


# === Chunk 7b: MultiPv =====================================================


def test_extract_mcts_default_multipv_is_one() -> None:
    """multipv defaults to 1 (single-PV; byte-identical with pre-7b)."""
    eng = _fake_engine_returning("starting_position", bestmove_uci="d2d4")
    extract_mcts(chess.Board(), engine=eng)

    _, kwargs = eng.analyse_verbose.call_args
    assert kwargs.get("multipv", 1) == 1


def test_extract_mcts_multipv_kwarg_threads_to_engine() -> None:
    """Caller's multipv kwarg reaches Lc0Engine.analyse_verbose unchanged."""
    eng = _fake_engine_returning("starting_position", bestmove_uci="d2d4")
    extract_mcts(chess.Board(), engine=eng, multipv=5)

    _, kwargs = eng.analyse_verbose.call_args
    assert kwargs["multipv"] == 5


def test_extract_mcts_populates_pv_san_on_matching_children() -> None:
    """When multipv > 1, principal_variations are mapped to MCTSChild.pv_san
    by matching the first move's UCI."""
    text = (FIXTURES_DIR / "walkthrough_rxd4.txt").read_text().splitlines()
    info_lines = [line for line in text if line and not line.startswith("#")]
    eng = MagicMock()
    eng.analyse_verbose.return_value = Lc0SearchResult(
        info_string_lines=info_lines,
        bestmove_uci="d8d4",
        nodes_requested=2000,
        wall_seconds=0.4,
        search_complete=True,
        multipv_requested=3,
        # Two PVs: PV1 = Rxd4 Qc3, PV2 = Rxb1+ Kxb1. First move of each
        # in UCI form is the key python-chess populates from info["pv"].
        # Kept short (2 plies each) to avoid PV-illegality flakiness on
        # this complex position.
        principal_variations={
            1: ["d8d4", "a3c3"],   # Rxd4 Qc3
            2: ["b8b1", "a1b1"],   # Rxb1+ Kxb1
        },
    )
    board = chess.Board("1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1")

    findings = extract_mcts(board, engine=eng, nodes=2000, multipv=3)

    # Find the child whose uci == d8d4; it should have pv_san populated
    # with SAN form of the PV sequence.
    rxd4_child = next((c for c in findings.children if c.uci == "d8d4"), None)
    rxb1_child = next((c for c in findings.children if c.uci == "b8b1"), None)
    assert rxd4_child is not None
    assert rxb1_child is not None
    assert rxd4_child.pv_san == ["Rxd4", "Qc3"]
    assert rxb1_child.pv_san == ["Rxb1+", "Kxb1"]


def test_extract_mcts_multipv_1_does_not_populate_pv_san() -> None:
    """When multipv == 1 (default), pv_san is left as the parser produced
    it (empty for Lc0 v0.32.1)."""
    text = (FIXTURES_DIR / "walkthrough_rxd4.txt").read_text().splitlines()
    info_lines = [line for line in text if line and not line.startswith("#")]
    eng = MagicMock()
    eng.analyse_verbose.return_value = Lc0SearchResult(
        info_string_lines=info_lines,
        bestmove_uci="d8d4",
        nodes_requested=2000,
        wall_seconds=0.4,
        search_complete=True,
        # multipv_requested defaults to 1; principal_variations defaults to {}
    )
    board = chess.Board("1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1")

    findings = extract_mcts(board, engine=eng, nodes=2000)

    # No child should have a populated pv_san — the parser leaves them empty.
    assert all(not c.pv_san for c in findings.children)


# === Chunk 7c.5.B: pv_san_truncated flag ====================================


def test_pv_san_truncated_set_when_san_walk_aborts() -> None:
    """When Lc0 emits a UCI sequence whose later moves are illegal on the
    walked board state, _populate_pv_san catches the truncation and sets
    the child's pv_san_truncated=True. The flag is surfaced as a findings-
    level signal so downstream consumers (prompts, validator) can react
    instead of seeing a silent partial PV."""
    text = (FIXTURES_DIR / "walkthrough_rxd4.txt").read_text().splitlines()
    info_lines = [line for line in text if line and not line.startswith("#")]
    eng = MagicMock()
    eng.analyse_verbose.return_value = Lc0SearchResult(
        info_string_lines=info_lines,
        bestmove_uci="d8d4",
        nodes_requested=2000,
        wall_seconds=0.4,
        search_complete=True,
        multipv_requested=1,
        # PV starts cleanly (Rxd4 Qc3) then becomes illegal at the 3rd move
        # (a3a8 would be a queen jump that's blocked / wrong square).
        principal_variations={
            1: ["d8d4", "a3c3", "a3a8"],  # 3rd move from a3 is illegal: queen just moved
        },
    )
    board = chess.Board(
        "1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1",
    )

    findings = extract_mcts(board, engine=eng, nodes=2000, multipv=1)

    rxd4 = next((c for c in findings.children if c.uci == "d8d4"), None)
    assert rxd4 is not None
    # Truncation: walker captured "Rxd4" and "Qc3" before the 3rd move
    # raised. The pv_san_truncated flag should be True.
    assert rxd4.pv_san_truncated is True, (
        f"Expected pv_san_truncated=True after walk aborted; "
        f"pv_san={rxd4.pv_san}, pv_san_truncated={rxd4.pv_san_truncated}"
    )
    # Partial PV preserved (1-2 valid moves before the abort).
    assert len(rxd4.pv_san) >= 1
    assert rxd4.pv_san[0] == "Rxd4"


def test_pv_san_truncated_false_when_walk_completes() -> None:
    """When the SAN walk completes successfully, pv_san_truncated stays
    False (default). Regression guard against accidentally flagging clean
    PVs as truncated."""
    text = (FIXTURES_DIR / "walkthrough_rxd4.txt").read_text().splitlines()
    info_lines = [line for line in text if line and not line.startswith("#")]
    eng = MagicMock()
    eng.analyse_verbose.return_value = Lc0SearchResult(
        info_string_lines=info_lines,
        bestmove_uci="d8d4",
        nodes_requested=2000,
        wall_seconds=0.4,
        search_complete=True,
        multipv_requested=1,
        # Fully legal 2-move PV.
        principal_variations={1: ["d8d4", "a3c3"]},
    )
    board = chess.Board(
        "1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1",
    )

    findings = extract_mcts(board, engine=eng, nodes=2000, multipv=1)

    rxd4 = next((c for c in findings.children if c.uci == "d8d4"), None)
    assert rxd4 is not None
    assert rxd4.pv_san == ["Rxd4", "Qc3"]
    assert rxd4.pv_san_truncated is False
