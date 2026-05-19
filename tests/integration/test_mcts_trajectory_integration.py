"""End-to-end ``extract_mcts_trajectory`` test on the pod — real Lc0, real BT4.

Skipped off-pod. On the pod, spawns Lc0 with ``--log-live-stats`` and
runs the single-search trajectory extractor on the walkthrough Rxd4
position. Asserts the predicted dual-timescale story from
``docs/INTERPRETABILITY_NOTES.md`` §5.2:

- **Rxd4** Q rises (or stays high) within the same tree as visits
  accumulate.
- **Rxb1+** Q drops within the same tree as the Kxb1 refutation
  propagates and PUCT redirects visits.

This is the *search-time* validation of Chunk 6a v1's methodology. The
network-time half (layer-emergence) is validated by 6b's integration
test. Assertions are direction-and-magnitude (single comparison +
tolerance), not strict monotonicity — MCTS Q at low visit counts is
noisy.
"""
from __future__ import annotations

from pathlib import Path

import chess
import pytest

from leela_chat.engine.lc0_engine import Lc0Engine
from leela_chat.extractors.mcts_trajectory import extract_mcts_trajectory

LC0_BINARY = Path("/workspace/bin/lc0")
BT4_WEIGHTS = Path(
    "/workspace/weights/BT4-1024x15x32h-swa-6147500/model.pb.gz"
)

pytestmark = pytest.mark.skipif(
    not (LC0_BINARY.exists() and BT4_WEIGHTS.exists()),
    reason="requires Lc0 binary + BT4 weights at pod paths",
)

WALKTHROUGH_FEN = "1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1"

# Direction-and-magnitude thresholds for the Rxd4-walkthrough trajectory.
# Single comparison + tolerance, NOT strict monotonicity — MCTS Q at low
# visit counts is noisy and a flake-prone test would erode trust.
Q_RISE_THRESHOLD = 0.10   # for confirmed moves: q_last - q_first > +0.10
Q_DROP_THRESHOLD = -0.10  # for refuted moves:   q_last - q_first < -0.10


def test_rxd4_q_confirms_within_one_search() -> None:
    """Rxd4 is the eventual bestmove. Within one continuous tree, its Q
    at the deepest milestone should materially exceed its Q at the
    shallowest milestone (rise of >0.10) — search progressively
    confirms the move."""
    board = chess.Board(WALKTHROUGH_FEN)
    with Lc0Engine(binary=LC0_BINARY, weights=BT4_WEIGHTS) as eng:
        findings = extract_mcts_trajectory(
            board, engine=eng, budgets=[500, 1000, 2500, 5000, 10000],
        )

    assert findings.bestmove_san == "Rxd4"
    rxd4 = next(c for c in findings.children if c.uci == "d8d4")
    assert len(rxd4.trajectory) == 5

    qs = [p.q for p in rxd4.trajectory]
    visits = [p.visits for p in rxd4.trajectory]
    print(f"Rxd4 trajectory: visits={visits} qs={qs}")

    # Direction + magnitude. Allows local dips along the way.
    assert qs[-1] - qs[0] > Q_RISE_THRESHOLD, (
        f"Rxd4 Q should rise by >{Q_RISE_THRESHOLD} across the search "
        f"as PUCT confirms it; got qs={qs}"
    )
    # Final Q should be confidently winning (Rxd4 wins material).
    assert qs[-1] > 0.7, f"Rxd4 final Q should be > +0.7; got {qs[-1]}"


def test_rxb1_check_refuted_within_one_search() -> None:
    """Rxb1+ has the highest policy prior in this position (~27%) so
    early search visits it. The Kxb1 refutation propagates as visits
    deepen. Within one continuous tree, Q at the deepest milestone
    should be materially below Q at the shallowest milestone (drop
    of >0.10)."""
    board = chess.Board(WALKTHROUGH_FEN)
    with Lc0Engine(binary=LC0_BINARY, weights=BT4_WEIGHTS) as eng:
        findings = extract_mcts_trajectory(
            board, engine=eng, budgets=[500, 1000, 2500, 5000, 10000],
        )

    rxb1 = next((c for c in findings.children if c.uci == "b8b1"), None)
    assert rxb1 is not None, "Rxb1+ should appear in the deepest snapshot"
    assert len(rxb1.trajectory) == 5

    qs = [p.q for p in rxb1.trajectory]
    visits = [p.visits for p in rxb1.trajectory]
    print(f"Rxb1+ trajectory: visits={visits} qs={qs}")

    # Search-time refutation signature: final Q materially below early
    # Q. Single comparison; no monotonicity assertion.
    assert qs[-1] - qs[0] < Q_DROP_THRESHOLD, (
        f"Rxb1+ Q should drop by >{abs(Q_DROP_THRESHOLD)} across the "
        f"search as Kxb1 propagates; got qs={qs}"
    )

    # And the final Q should be meaningfully lower than Rxd4's
    # (otherwise the refutation didn't propagate clearly).
    rxd4 = next(c for c in findings.children if c.uci == "d8d4")
    assert qs[-1] < rxd4.trajectory[-1].q - 0.2, (
        f"Rxb1+ final Q {qs[-1]} should trail Rxd4's "
        f"{rxd4.trajectory[-1].q} by >=0.2"
    )


def test_trajectory_smoke_on_starting_position() -> None:
    """Sanity check: starting position is easy; trajectory should
    populate cleanly at modest budgets without errors."""
    with Lc0Engine(binary=LC0_BINARY, weights=BT4_WEIGHTS) as eng:
        findings = extract_mcts_trajectory(
            chess.Board(), engine=eng, budgets=[200, 500, 1000],
        )

    assert findings.search_complete is True
    assert len(findings.children) == 20  # 20 legal moves from start
    # Every child has a 3-point trajectory.
    for c in findings.children:
        assert len(c.trajectory) == 3
        # nodes_searched is non-decreasing across the trajectory.
        nodes = [p.nodes_searched for p in c.trajectory]
        assert nodes == sorted(nodes), f"{c.san}: nodes not ascending: {nodes}"


def test_live_stats_block_count_meaningful_for_10k_search() -> None:
    """Sanity check on Lc0's --log-live-stats cadence: a 10k-node
    search should produce more than one emit cycle (otherwise the
    trajectory has only one snapshot and the methodology degrades to
    end-of-search-only). If this fails, the investigation gate in
    PHASE_NOTES needs to record what we saw and we may need to revisit
    the milestone-selection logic."""
    from leela_chat.extractors.mcts_trajectory import _group_live_stats_blocks

    board = chess.Board(WALKTHROUGH_FEN)
    with Lc0Engine(binary=LC0_BINARY, weights=BT4_WEIGHTS) as eng:
        # Bypass extract_mcts_trajectory to inspect the raw block count.
        result = eng.analyse_verbose(board, nodes=10000, log_live_stats=True)
        blocks = _group_live_stats_blocks(result.info_string_lines)

    print(f"--log-live-stats produced {len(blocks)} blocks in a 10k search")
    assert len(blocks) >= 2, (
        f"expected --log-live-stats to emit multiple blocks during a 10k "
        f"search; got {len(blocks)}. Check Lc0's emission cadence."
    )
