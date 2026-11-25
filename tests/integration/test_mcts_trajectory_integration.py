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

# Trajectory-summary thresholds (Chunk 6a v2 + pod-validated 2026-05-19).
# These look at the q_peak / q_trough / q_first / q_last summary fields
# rather than milestone-snapshot endpoints — Lc0's live-stats cadence
# stops emitting once bestmove is confident (typically around the
# 50%-of-budget mark on easy positions), so the "deepest milestone"
# isn't the actual end-of-search state. The summary stats are taken
# over the full intra-search trajectory and are robust to this.
Q_RISE_FROM_FIRST_THRESHOLD = 0.10   # confirmed: q_peak > q_first + 0.10
Q_DIP_FROM_FIRST_THRESHOLD = -0.20   # refuted:   q_trough < q_first - 0.20


def test_rxd4_q_confirms_within_one_search() -> None:
    """Rxd4 is the eventual bestmove. Within one continuous tree, its
    Q-peak (best Q across all live-stats snapshots) should materially
    exceed its Q at the first snapshot — search progressively confirms
    the move.

    Empirical (pod 2026-05-19): Rxd4 starts at Q≈+0.56 (V-head only,
    visits=0 at the first emit) and peaks at Q≈+0.96 around visit 88-150.
    """
    board = chess.Board(WALKTHROUGH_FEN)
    with Lc0Engine(binary=LC0_BINARY, weights=BT4_WEIGHTS) as eng:
        findings = extract_mcts_trajectory(
            board, engine=eng, budgets=[500, 1000, 2500, 5000, 10000],
        )

    assert findings.bestmove_san == "Rxd4"
    rxd4 = next(c for c in findings.children if c.uci == "d8d4")
    summary = rxd4.trajectory_summary
    assert summary is not None, "Rxd4 should have a trajectory_summary"

    print(
        f"Rxd4 summary: q_first={summary.q_first:+.3f} "
        f"q_peak={summary.q_peak:+.3f}@v{summary.q_peak_at_visits} "
        f"q_trough={summary.q_trough:+.3f}@v{summary.q_trough_at_visits} "
        f"q_last={summary.q_last:+.3f} blocks={summary.block_count}"
    )

    # Confirmation signature: Q climbs materially from the first snapshot.
    assert summary.q_peak - summary.q_first > Q_RISE_FROM_FIRST_THRESHOLD, (
        f"Rxd4 q_peak should exceed q_first by >{Q_RISE_FROM_FIRST_THRESHOLD}; "
        f"got q_first={summary.q_first}, q_peak={summary.q_peak}"
    )
    # Peak should be confidently winning.
    assert summary.q_peak > 0.85, (
        f"Rxd4 q_peak should be > +0.85 (winning); got {summary.q_peak}"
    )


def test_rxb1_check_refuted_within_one_search() -> None:
    """Rxb1+ has the highest policy prior in this position (~27%) so
    early search visits it with high V-head optimism. The Kxb1
    refutation propagates as visits deepen; Q reaches a trough well
    below the initial value before partially recovering at deeper
    visits.

    Empirical (pod 2026-05-19): Rxb1+ starts at Q≈+0.74 (block 0,
    visits=1) and reaches a trough around Q≈+0.36 (around 1500 nodes
    total, Rxb1+ visits ~24). Q then partially recovers to ~+0.53 by
    the last live-stats emit. The refutation signature is the trough,
    not the final value.
    """
    board = chess.Board(WALKTHROUGH_FEN)
    with Lc0Engine(binary=LC0_BINARY, weights=BT4_WEIGHTS) as eng:
        findings = extract_mcts_trajectory(
            board, engine=eng, budgets=[500, 1000, 2500, 5000, 10000],
        )

    rxb1 = next((c for c in findings.children if c.uci == "b8b1"), None)
    assert rxb1 is not None, "Rxb1+ should appear in the deepest snapshot"
    summary = rxb1.trajectory_summary
    assert summary is not None, "Rxb1+ should have a trajectory_summary"

    print(
        f"Rxb1+ summary: q_first={summary.q_first:+.3f} "
        f"q_peak={summary.q_peak:+.3f}@v{summary.q_peak_at_visits} "
        f"q_trough={summary.q_trough:+.3f}@v{summary.q_trough_at_visits} "
        f"q_last={summary.q_last:+.3f} blocks={summary.block_count} "
        f"inflections={len(summary.inflection_points)}"
    )

    # Refutation signature: Q dips materially below q_first at some
    # point in the search. We compare q_trough to q_first; the dip's
    # magnitude is what tells the "search refuted this" story.
    assert summary.q_trough - summary.q_first < Q_DIP_FROM_FIRST_THRESHOLD, (
        f"Rxb1+ q_trough should dip >{abs(Q_DIP_FROM_FIRST_THRESHOLD)} "
        f"below q_first; got q_first={summary.q_first}, q_trough={summary.q_trough}"
    )
    # And the trough should be meaningfully lower than Rxd4's peak —
    # otherwise the dual-timescale story (search refutes A, search
    # confirms B) doesn't actually distinguish them.
    rxd4 = next(c for c in findings.children if c.uci == "d8d4")
    rxd4_summary = rxd4.trajectory_summary
    assert rxd4_summary is not None
    assert summary.q_trough < rxd4_summary.q_peak - 0.4, (
        f"Rxb1+ q_trough ({summary.q_trough}) should trail Rxd4's q_peak "
        f"({rxd4_summary.q_peak}) by >=0.4 — otherwise the divergent "
        f"signal isn't strong enough to narrate."
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
