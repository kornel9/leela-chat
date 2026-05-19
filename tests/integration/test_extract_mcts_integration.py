"""End-to-end MCTS extractor test on the pod — real Lc0, real BT4.

Skipped automatically off-pod. On the pod, spawns Lc0, runs
``extract_mcts`` on the walkthrough Rxd4 position, asserts the parser
agrees with the engine: Rxd4 should be top-visit with high Q. This is
the integration-level confirmation that Phase 2's pipeline fixes the
Phase-1 walkthrough gap.
"""
from __future__ import annotations

from pathlib import Path

import chess
import pytest

from leela_chat.engine.lc0_engine import Lc0Engine
from leela_chat.extractors.mcts import extract_mcts

LC0_BINARY = Path("/workspace/bin/lc0")
BT4_WEIGHTS = Path(
    "/workspace/weights/BT4-1024x15x32h-swa-6147500/model.pb.gz"
)

pytestmark = pytest.mark.skipif(
    not (LC0_BINARY.exists() and BT4_WEIGHTS.exists()),
    reason="requires Lc0 binary + BT4 weights at pod paths",
)


def test_extract_mcts_finds_rxd4_on_walkthrough_position() -> None:
    """The flagship Phase-2 case: position 4 from the walkthrough. Phase
    1's policy-only response missed Rxd4 because the network ranked it
    only 4th (~7% prior). MCTS at 2000 nodes should commit most visits to
    Rxd4 and report it as bestmove."""
    fen = "1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1"
    board = chess.Board(fen)
    with Lc0Engine(binary=LC0_BINARY, weights=BT4_WEIGHTS) as eng:
        findings = extract_mcts(board, engine=eng, nodes=2000)

    assert findings.search_complete is True
    assert findings.bestmove_san == "Rxd4"
    assert findings.bestmove_uci == "d8d4"

    top = findings.children[0]
    assert top.uci == "d8d4"
    assert top.visits > 200, (
        f"expected Lc0 to commit most visits to Rxd4 at 2000 nodes; "
        f"got top.visits={top.visits}"
    )
    assert top.q > 0.5, (
        f"expected high Q for Rxd4 (the position is winning for Black); "
        f"got Q={top.q}"
    )
    # And the parser's structural promises hold under live engine output.
    assert findings.root_stats is not None
    assert len(findings.children) >= 10


def test_extract_mcts_starting_position_smoke() -> None:
    """Less specific check on a known-easy position — confirms the pipeline
    is wired end-to-end and 500-node budget is enough for clean output."""
    with Lc0Engine(binary=LC0_BINARY, weights=BT4_WEIGHTS) as eng:
        findings = extract_mcts(chess.Board(), engine=eng, nodes=500)

    assert findings.search_complete is True
    assert findings.bestmove_uci is not None
    assert findings.bestmove_san is not None
    assert len(findings.children) == 20  # 20 legal moves from start
    # Top move should have been visited at least a few times.
    assert findings.children[0].visits > 5
