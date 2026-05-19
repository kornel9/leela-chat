"""End-to-end ``extract_mcts_trajectory`` test on the pod — real Lc0, real BT4.

Skipped off-pod. On the pod, spawns Lc0 and runs the multi-budget
trajectory extractor on the walkthrough Rxd4 position. Asserts the
predicted dual-timescale story from
``docs/INTERPRETABILITY_NOTES.md`` §5.2:

- **Rxd4** Q rises (or stays high) across budgets as deeper search
  confirms the move.
- **Rxb1+** visits stagnate and Q drops as deeper search exposes the
  Kxb1 refutation.

This is the *search-time* validation of Chunk 6a's methodology. The
network-time half (layer-emergence) is validated by 6b's integration
test.
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


def test_rxd4_q_confirms_across_budgets() -> None:
    """Rxd4 is the eventual bestmove. Across budgets [500..10000] its Q
    should *not* fall — the deeper the search, the more confident Lc0
    becomes that this move wins material."""
    board = chess.Board(WALKTHROUGH_FEN)
    with Lc0Engine(binary=LC0_BINARY, weights=BT4_WEIGHTS) as eng:
        findings = extract_mcts_trajectory(
            board, engine=eng, budgets=[500, 1000, 2500, 5000, 10000],
        )

    assert findings.bestmove_san == "Rxd4"
    rxd4 = next(c for c in findings.children if c.uci == "d8d4")
    assert len(rxd4.trajectory) == 5

    # Q should be high at every checkpoint (Rxd4 is winning material).
    # Allow a small dip; require all points above +0.5 and final point
    # above +0.7. Tolerances calibrated to leave headroom for engine
    # nondeterminism.
    qs = [p.q for p in rxd4.trajectory]
    assert all(q > 0.5 for q in qs), f"Rxd4 Q dipped below +0.5 somewhere: {qs}"
    assert qs[-1] > 0.7, f"Rxd4 final Q should be confidently winning; got {qs[-1]}"

    # Visit count should grow strictly across budgets (deeper search keeps
    # committing to this move).
    visits = [p.visits for p in rxd4.trajectory]
    for v1, v2 in zip(visits, visits[1:]):
        assert v2 >= v1, f"Rxd4 visits non-monotonic across budgets: {visits}"


def test_rxb1_check_refuted_across_budgets() -> None:
    """Rxb1+ has the *highest* policy prior in this position (~27%) — early
    search will visit it. But it has a refutation (Kxb1 holds). Across
    budgets, its visits should stagnate (PUCT redirects to Rxd4) and its
    Q should drop as the refutation propagates from the leaves up."""
    board = chess.Board(WALKTHROUGH_FEN)
    with Lc0Engine(binary=LC0_BINARY, weights=BT4_WEIGHTS) as eng:
        findings = extract_mcts_trajectory(
            board, engine=eng, budgets=[500, 1000, 2500, 5000, 10000],
        )

    rxb1 = next((c for c in findings.children if c.uci == "b8b1"), None)
    assert rxb1 is not None, "Rxb1+ should appear in the deepest-budget findings"
    assert len(rxb1.trajectory) == 5

    qs = [p.q for p in rxb1.trajectory]
    visits = [p.visits for p in rxb1.trajectory]

    # Search-time refutation signature: final Q below early Q. We don't
    # require strict monotonicity (PUCT visit order creates noise) — just
    # that the trajectory goes down overall.
    assert qs[-1] < qs[0], (
        f"expected Rxb1+ Q to drop as search deepens; got {qs}"
    )

    # And the final Q should be meaningfully lower than Rxd4's
    # (otherwise the refutation didn't propagate).
    rxd4 = next(c for c in findings.children if c.uci == "d8d4")
    assert qs[-1] < rxd4.trajectory[-1].q - 0.2, (
        f"Rxb1+ final Q {qs[-1]} should trail Rxd4's "
        f"{rxd4.trajectory[-1].q} by ≥0.2"
    )

    # Visits should *not* keep growing aggressively — by the deepest budget
    # PUCT has redirected to Rxd4. (Loose check: visits at 10k shouldn't
    # be more than ~10× visits at 500 — would be unusual for a refuted
    # move.)
    assert visits[-1] < 10 * max(visits[0], 1), (
        f"expected Rxb1+ visits to stagnate as search deepens; got {visits}"
    )


def test_trajectory_smoke_on_starting_position() -> None:
    """Sanity check: starting position is easy; trajectory should populate
    cleanly at modest budgets without errors."""
    with Lc0Engine(binary=LC0_BINARY, weights=BT4_WEIGHTS) as eng:
        findings = extract_mcts_trajectory(
            chess.Board(), engine=eng, budgets=[200, 500, 1000],
        )

    assert findings.search_complete is True
    assert len(findings.children) == 20  # 20 legal moves from start
    # Every child has a 3-point trajectory.
    for c in findings.children:
        assert len(c.trajectory) == 3
        # Nodes searched is monotonic ascending.
        nodes = [p.nodes_searched for p in c.trajectory]
        assert nodes == sorted(nodes), f"{c.san}: nodes not ascending: {nodes}"
