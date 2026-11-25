"""Integration test for ``Lc0Engine`` — runs against the real Lc0 binary.

Skipped automatically off-pod (when ``/workspace/bin/lc0`` is missing).
On the pod, exercises the full path: spawn Lc0 with BT4 weights, run a
500-node search on the starting position, assert the captured info-string
lines have the expected per-move-candidate shape.

Why 500 nodes (not the 10000 default): keeps this test under ~3 seconds
on the pod's GPU. Format-shape is the same; only depth differs.
"""
from __future__ import annotations

from pathlib import Path

import chess
import pytest

from leela_chat.engine.lc0_engine import Lc0Engine

LC0_BINARY = Path("/workspace/bin/lc0")
BT4_WEIGHTS = Path(
    "/workspace/weights/BT4-1024x15x32h-swa-6147500/model.pb.gz"
)


pytestmark = pytest.mark.skipif(
    not (LC0_BINARY.exists() and BT4_WEIGHTS.exists()),
    reason="requires Lc0 binary + BT4 weights at pod paths",
)


def test_lc0_engine_smoke_starting_position() -> None:
    """End-to-end smoke: real Lc0, real BT4, starting position, 500 nodes."""
    with Lc0Engine(binary=LC0_BINARY, weights=BT4_WEIGHTS) as eng:
        assert "lc0" in eng.name.lower() or "leela" in eng.name.lower(), (
            f"engine name should identify as Lc0/Leela; got {eng.name!r}"
        )
        result = eng.analyse_verbose(chess.Board(), nodes=500)

    # Search completed normally.
    assert result.search_complete is True
    assert result.bestmove_uci is not None
    assert len(result.bestmove_uci) in (4, 5), (
        f"bestmove should be valid UCI (4 or 5 chars); got {result.bestmove_uci!r}"
    )

    # We captured per-candidate stat lines. Sub-500-node search on the
    # starting position yields one stat line per legal root move; 20 legal
    # moves there, so expect 20+ stat-shaped lines (Lc0 may also emit a
    # few non-move diagnostic strings — assert "at least 15" to leave slack).
    stat_lines = [
        s for s in result.info_string_lines
        if "N:" in s and "P:" in s and "Q:" in s
    ]
    assert len(stat_lines) >= 15, (
        f"expected ≥15 per-move-candidate stat lines, got {len(stat_lines)}; "
        f"all captured: {result.info_string_lines[:3]}..."
    )

    # Each stat line should mention a UCI-shaped move at its start.
    first_token = stat_lines[0].split()[0].rstrip(":,")
    assert len(first_token) in (4, 5) and first_token[0].isalpha(), (
        f"first token of a stat line should look like a UCI move; got {first_token!r}"
    )

    # Wall-clock should be small (500 nodes is fast on the pod GPU).
    assert result.wall_seconds < 30.0, (
        f"500-node search took unexpectedly long: {result.wall_seconds:.1f}s"
    )

    # Node budget echo.
    assert result.nodes_requested == 500


def test_lc0_engine_handles_known_tactical_position() -> None:
    """The Rxd4 position from the walkthrough — sanity-check Lc0 finds Rxd4
    after enough nodes. This is more of a "system works" check than a
    behavioral assertion; only insists that *some* bestmove emerges and
    that captured lines are well-formed."""
    fen = "1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1"
    board = chess.Board(fen)
    with Lc0Engine(binary=LC0_BINARY, weights=BT4_WEIGHTS) as eng:
        result = eng.analyse_verbose(board, nodes=2000)

    assert result.search_complete is True
    assert result.bestmove_uci is not None
    # Per-move-candidate lines must be present.
    stat_lines = [
        s for s in result.info_string_lines
        if "N:" in s and "Q:" in s
    ]
    assert len(stat_lines) >= 10
