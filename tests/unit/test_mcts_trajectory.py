"""Unit tests for ``extract_mcts_trajectory`` — multi-budget MCTS extractor.

Mocks the single-budget ``extract_mcts`` helper so each test can fake an
arbitrary sequence of checkpoints. Verifies trajectory assembly,
zero-fill for candidates absent from a lower-budget run, drop-from-top
behavior, validation of the ``budgets`` argument, and ascending order
of trajectory points.

The real engine is exercised on the pod via
``tests/integration/test_mcts_trajectory_integration.py``.
"""
from __future__ import annotations

import chess
import pytest

from leela_chat.extractors.mcts_parser import (
    MCTSChild,
    MCTSFindings,
    MCTSRootStats,
    MCTSTrajectoryPoint,
)
from leela_chat.extractors import mcts_trajectory as mt
from leela_chat.extractors.mcts_trajectory import (
    DEFAULT_BUDGETS,
    extract_mcts_trajectory,
)


# === fixtures ==============================================================


WALKTHROUGH_FEN = "1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1"


def _mk_child(
    uci: str,
    san: str,
    visits: int,
    q: float,
    *,
    prior_pct: float = 5.0,
) -> MCTSChild:
    """Compact factory for a structurally-valid MCTSChild fixture."""
    return MCTSChild(
        uci=uci, san=san,
        visits=visits, visits_in_flight=0,
        prior_pct=prior_pct, q=q, v=q, wl=q,
        draw=0.1, moves_left=50.0, u=0.01, s=0.1,
        pv_san=[],
    )


def _mk_findings(
    total_nodes: int,
    bestmove_uci: str,
    bestmove_san: str,
    children: list[MCTSChild],
) -> MCTSFindings:
    """Compact factory for an MCTSFindings fixture."""
    return MCTSFindings(
        total_nodes_searched=total_nodes,
        bestmove_uci=bestmove_uci,
        bestmove_san=bestmove_san,
        children=children,
        root_stats=MCTSRootStats(
            visits=total_nodes, visits_in_flight=0,
            prior_pct=95.0, q=0.5, v=0.5, wl=0.5,
            draw=0.1, moves_left=50.0,
        ),
        search_complete=True,
    )


@pytest.fixture
def board() -> chess.Board:
    return chess.Board(WALKTHROUGH_FEN)


def _patch_extract_mcts(monkeypatch, checkpoints: list[MCTSFindings]) -> list[int]:
    """Patch the module's ``extract_mcts`` to return canned checkpoints in
    order. Returns a captured list of the ``nodes`` argument used at each
    call so tests can assert the budgets reached the engine.
    """
    captured_nodes: list[int] = []
    queue = list(checkpoints)

    def fake_extract_mcts(board, *, engine, nodes):
        captured_nodes.append(nodes)
        assert queue, "extract_mcts called more times than checkpoints provided"
        return queue.pop(0)

    monkeypatch.setattr(mt, "extract_mcts", fake_extract_mcts)
    return captured_nodes


# === happy path ============================================================


def test_trajectory_assembled_across_three_budgets(monkeypatch, board):
    """Two candidates present at all three budgets → trajectory has 3 points
    each, in budget-ascending order, with correct (nodes_searched, visits, q)."""
    ck_low = _mk_findings(500, "d8d4", "Rxd4", [
        _mk_child("d8d4", "Rxd4", visits=45, q=0.85),
        _mk_child("b8b1", "Rxb1+", visits=30, q=0.70),
    ])
    ck_mid = _mk_findings(2500, "d8d4", "Rxd4", [
        _mk_child("d8d4", "Rxd4", visits=300, q=0.92),
        _mk_child("b8b1", "Rxb1+", visits=25, q=0.55),
    ])
    ck_high = _mk_findings(10000, "d8d4", "Rxd4", [
        _mk_child("d8d4", "Rxd4", visits=850, q=0.95),
        _mk_child("b8b1", "Rxb1+", visits=20, q=0.40),
    ])
    captured = _patch_extract_mcts(monkeypatch, [ck_low, ck_mid, ck_high])

    result = extract_mcts_trajectory(
        board, engine=object(), budgets=[500, 2500, 10000],
    )

    # Engine got the budgets we asked for, in order.
    assert captured == [500, 2500, 10000]

    # Result is based on the highest-budget findings.
    assert result.total_nodes_searched == 10000
    assert result.bestmove_san == "Rxd4"
    assert len(result.children) == 2

    rxd4 = next(c for c in result.children if c.uci == "d8d4")
    rxb1 = next(c for c in result.children if c.uci == "b8b1")

    # Current-checkpoint stats reflect the highest budget.
    assert rxd4.visits == 850
    assert rxd4.q == 0.95

    # Trajectory: 3 points each, in budget-ascending order.
    assert len(rxd4.trajectory) == 3
    assert [p.nodes_searched for p in rxd4.trajectory] == [500, 2500, 10000]
    assert [p.visits for p in rxd4.trajectory] == [45, 300, 850]
    assert [p.q for p in rxd4.trajectory] == [0.85, 0.92, 0.95]

    assert len(rxb1.trajectory) == 3
    assert [p.visits for p in rxb1.trajectory] == [30, 25, 20]
    assert [p.q for p in rxb1.trajectory] == [0.70, 0.55, 0.40]


def test_default_budgets_used_when_not_specified(monkeypatch, board):
    """Calling without ``budgets`` uses ``DEFAULT_BUDGETS`` in order."""
    checkpoints = [
        _mk_findings(b, "d8d4", "Rxd4", [_mk_child("d8d4", "Rxd4", 10, 0.5)])
        for b in DEFAULT_BUDGETS
    ]
    captured = _patch_extract_mcts(monkeypatch, checkpoints)

    result = extract_mcts_trajectory(board, engine=object())

    assert captured == list(DEFAULT_BUDGETS)
    assert len(result.children[0].trajectory) == len(DEFAULT_BUDGETS)


# === edge cases ============================================================


def test_candidate_absent_from_low_budget_gets_zero_fill(monkeypatch, board):
    """A move present only at higher budgets → zero-visit / zero-Q at lower
    checkpoints (search hadn't reached it yet)."""
    ck_low = _mk_findings(500, "d8d4", "Rxd4", [
        _mk_child("d8d4", "Rxd4", visits=45, q=0.85),
        # Rxb1+ absent at low budget.
    ])
    ck_high = _mk_findings(10000, "d8d4", "Rxd4", [
        _mk_child("d8d4", "Rxd4", visits=850, q=0.95),
        _mk_child("b8b1", "Rxb1+", visits=20, q=0.40),
    ])
    _patch_extract_mcts(monkeypatch, [ck_low, ck_high])

    result = extract_mcts_trajectory(
        board, engine=object(), budgets=[500, 10000],
    )

    rxb1 = next(c for c in result.children if c.uci == "b8b1")
    assert len(rxb1.trajectory) == 2
    # Low-budget checkpoint zero-filled.
    assert rxb1.trajectory[0].nodes_searched == 500
    assert rxb1.trajectory[0].visits == 0
    assert rxb1.trajectory[0].q == 0.0
    # High-budget checkpoint populated.
    assert rxb1.trajectory[1].visits == 20
    assert rxb1.trajectory[1].q == 0.40


def test_candidate_disappeared_from_top_n_at_high_budget_is_dropped(monkeypatch, board):
    """A move present only at lower budgets, not at the final budget, is
    NOT included in the result (we iterate base.children, which is the
    deepest search's reported list). This is intentional: if the deepest
    search didn't surface a move, it's not relevant to the trajectory."""
    ck_low = _mk_findings(500, "d8d4", "Rxd4", [
        _mk_child("d8d4", "Rxd4", visits=45, q=0.85),
        _mk_child("a4c6", "Bc6", visits=15, q=0.10),  # only at low budget
    ])
    ck_high = _mk_findings(10000, "d8d4", "Rxd4", [
        _mk_child("d8d4", "Rxd4", visits=850, q=0.95),
        # Bc6 dropped from the high-budget verbose-move-stats output.
    ])
    _patch_extract_mcts(monkeypatch, [ck_low, ck_high])

    result = extract_mcts_trajectory(
        board, engine=object(), budgets=[500, 10000],
    )

    ucis = [c.uci for c in result.children]
    assert "a4c6" not in ucis
    assert "d8d4" in ucis


def test_single_budget_call_produces_one_point_trajectory(monkeypatch, board):
    """Degenerate case: a single-budget call still works and gives each
    child a 1-point trajectory matching the call's stats."""
    ck = _mk_findings(10000, "d8d4", "Rxd4", [
        _mk_child("d8d4", "Rxd4", visits=850, q=0.95),
    ])
    _patch_extract_mcts(monkeypatch, [ck])

    result = extract_mcts_trajectory(
        board, engine=object(), budgets=[10000],
    )

    assert len(result.children[0].trajectory) == 1
    point = result.children[0].trajectory[0]
    assert point.nodes_searched == 10000
    assert point.visits == 850
    assert point.q == 0.95


def test_trajectory_points_are_in_budget_ascending_order(monkeypatch, board):
    """Even if budgets are passed in unusual ordering (which is rejected),
    the assembled trajectory mirrors the call order. Here we verify that
    ascending budgets produce strictly-ascending nodes_searched values."""
    checkpoints = [
        _mk_findings(b, "d8d4", "Rxd4", [_mk_child("d8d4", "Rxd4", b // 10, 0.5)])
        for b in [500, 1000, 2500]
    ]
    _patch_extract_mcts(monkeypatch, checkpoints)

    result = extract_mcts_trajectory(
        board, engine=object(), budgets=[500, 1000, 2500],
    )

    nodes = [p.nodes_searched for p in result.children[0].trajectory]
    assert nodes == sorted(nodes)


# === validation ============================================================


def test_empty_budgets_raises(board):
    with pytest.raises(ValueError, match="non-empty"):
        extract_mcts_trajectory(board, engine=object(), budgets=[])


def test_zero_or_negative_budget_raises(board):
    with pytest.raises(ValueError, match="positive"):
        extract_mcts_trajectory(board, engine=object(), budgets=[0, 1000])
    with pytest.raises(ValueError, match="positive"):
        extract_mcts_trajectory(board, engine=object(), budgets=[-1, 1000])


def test_non_ascending_budgets_raises(board):
    """Equal-adjacent and descending are both rejected — strict ascent."""
    with pytest.raises(ValueError, match="ascending"):
        extract_mcts_trajectory(board, engine=object(), budgets=[1000, 500])
    with pytest.raises(ValueError, match="ascending"):
        extract_mcts_trajectory(board, engine=object(), budgets=[1000, 1000])


# === schema sanity =========================================================


def test_returned_children_are_still_frozen_pydantic(monkeypatch, board):
    """The enriched MCTSChild objects must remain valid frozen Pydantic
    models — we used ``model_copy(update=...)`` so this should hold, but
    asserting protects future refactors."""
    ck = _mk_findings(1000, "d8d4", "Rxd4", [
        _mk_child("d8d4", "Rxd4", 100, 0.9),
    ])
    _patch_extract_mcts(monkeypatch, [ck])

    result = extract_mcts_trajectory(board, engine=object(), budgets=[1000])
    child = result.children[0]
    assert isinstance(child, MCTSChild)
    # Frozen → mutations raise.
    with pytest.raises((TypeError, ValueError)):
        child.visits = 999  # type: ignore[misc]
    # Trajectory points are MCTSTrajectoryPoint instances.
    for p in child.trajectory:
        assert isinstance(p, MCTSTrajectoryPoint)


def test_default_extract_mcts_call_leaves_trajectory_empty(monkeypatch, board):
    """Sanity: the trajectory field defaults to []; only extract_mcts_trajectory
    populates it. (Belt-and-suspenders check that backward-compat holds.)"""
    child = _mk_child("e2e4", "e4", 100, 0.05)
    assert child.trajectory == []
