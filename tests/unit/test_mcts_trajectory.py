"""Unit tests for ``extract_mcts_trajectory`` — single-search live-stats.

Mocks ``Lc0Engine.analyse_verbose`` to return a canned stream of
info-string lines covering several emit cycles. The streamed lines flow
through the real ``parse_verbose_move_stats`` and the block-grouper, so
these tests exercise both the trajectory-assembly logic and the
live-stats parsing path end-to-end.

The real engine is exercised on the pod via
``tests/integration/test_mcts_trajectory_integration.py``.

Methodology rationale lives in ``docs/INTERPRETABILITY_NOTES.md`` §4
(why we use a single search with ``--log-live-stats`` rather than
stitching K independent searches).
"""
from __future__ import annotations

import chess
import pytest

from leela_chat.engine.lc0_engine import Lc0SearchResult
from leela_chat.extractors.mcts_parser import (
    MCTSChild,
    MCTSFindings,
    MCTSTrajectoryPoint,
)
from leela_chat.extractors import mcts_trajectory as mt
from leela_chat.extractors.mcts_trajectory import (
    DEFAULT_BUDGETS,
    _group_live_stats_blocks,
    _select_milestone_snapshots,
    extract_mcts_trajectory,
)


# === fixture builders ======================================================


WALKTHROUGH_FEN = "1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1"
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _candidate_line(uci: str, visits: int, q: float, prior_pct: float) -> str:
    """One Lc0-formatted candidate stats line.

    Matches the format observed in tests/unit/fixtures/lc0/*.txt and
    parsed by ``parse_verbose_move_stats``. Note: Lc0 space-pads
    positives ("(Q:  0.85000)" not "(Q:+0.85000)"), and the parser's
    ``_FIELD_RE`` only accepts an optional ``-`` (no ``+``). Using a
    space flag (`{q: .5f}`) matches Lc0's convention.
    """
    return (
        f"{uci}  (1  ) N:{visits:7d} (+ 0) (P:{prior_pct:5.2f}%) "
        f"(WL:{q: .5f}) (D: 0.000) (M:  0.0) (Q:{q: .5f}) "
        f"(U: 0.05000) (S:  0.10000) (V:{q: .4f})"
    )


def _node_summary_line(root_visits: int) -> str:
    """One Lc0-formatted ``node`` summary line ending an emit cycle."""
    return (
        f"node  (20  ) N:{root_visits:7d} (+ 0) (P: 95.00%) "
        f"(WL:  0.50000) (D: 0.000) (M:  0.0) (Q:  0.50000) (V:  0.5000)"
    )


def _emit_cycle(
    candidates: list[tuple[str, int, float, float]], root_visits: int,
) -> list[str]:
    """One full live-stats emit cycle: candidate lines + node summary."""
    lines = [
        _candidate_line(uci, visits, q, prior_pct)
        for uci, visits, q, prior_pct in candidates
    ]
    lines.append(_node_summary_line(root_visits))
    return lines


def _make_search_result(
    cycles: list[list[str]],
    *,
    bestmove_uci: str | None,
    search_complete: bool = True,
    nodes_requested: int = 10000,
) -> Lc0SearchResult:
    """Bundle a list of emit cycles into a synthetic Lc0SearchResult."""
    info_lines: list[str] = []
    for cycle in cycles:
        info_lines.extend(cycle)
    return Lc0SearchResult(
        info_string_lines=info_lines,
        bestmove_uci=bestmove_uci,
        nodes_requested=nodes_requested,
        wall_seconds=1.0,
        search_complete=search_complete,
    )


class _StubEngine:
    """Drop-in stand-in for Lc0Engine in unit tests.

    Stores a queue of ``Lc0SearchResult`` to return on successive
    ``analyse_verbose`` calls, plus a captured log of the arguments
    each call was made with.
    """

    def __init__(self, results: list[Lc0SearchResult]) -> None:
        self._queue = list(results)
        self.calls: list[dict] = []

    def analyse_verbose(
        self, board, *, nodes: int = 10000, log_live_stats: bool = False,
    ) -> Lc0SearchResult:
        self.calls.append({"nodes": nodes, "log_live_stats": log_live_stats})
        assert self._queue, "analyse_verbose called more times than results provided"
        return self._queue.pop(0)


@pytest.fixture
def board() -> chess.Board:
    return chess.Board(WALKTHROUGH_FEN)


# === block grouper =========================================================


def test_group_live_stats_splits_at_node_summary_lines():
    """A stream of two emit cycles should produce two blocks."""
    cycle1 = _emit_cycle([("d8d4", 50, 0.85, 6.8), ("b8b1", 30, 0.70, 26.6)], 800)
    cycle2 = _emit_cycle([("d8d4", 200, 0.90, 6.8), ("b8b1", 35, 0.55, 26.6)], 1500)

    blocks = _group_live_stats_blocks(cycle1 + cycle2)

    assert len(blocks) == 2
    # Each block ends with its node-summary line.
    assert blocks[0][-1].lstrip().startswith("node ")
    assert blocks[1][-1].lstrip().startswith("node ")


def test_group_live_stats_handles_missing_final_node_summary():
    """A stream that ends mid-cycle (no terminating node-summary on the
    last block) should still surface the partial block if it contains
    candidate-shaped lines — otherwise it'd lose the final snapshot."""
    cycle1 = _emit_cycle([("d8d4", 50, 0.85, 6.8)], 800)
    partial = [_candidate_line("d8d4", 200, 0.90, 6.8)]  # no node summary

    blocks = _group_live_stats_blocks(cycle1 + partial)

    assert len(blocks) == 2
    assert blocks[1][-1].startswith("d8d4")


def test_group_live_stats_filters_pure_noise_tail():
    """A trailing block with only diagnostic chatter (no candidate
    headers) is dropped — that's just leftover engine output."""
    cycle1 = _emit_cycle([("d8d4", 50, 0.85, 6.8)], 800)
    noise = ["debug: nothing useful here", "info string: chatter"]

    blocks = _group_live_stats_blocks(cycle1 + noise)

    assert len(blocks) == 1


# === milestone selection ===================================================


def test_select_milestone_snapshots_picks_closest():
    """For each milestone, the snapshot whose total_nodes_searched is
    closest in absolute value wins."""
    snapshots = [
        MCTSFindings(
            total_nodes_searched=v, bestmove_uci=None, bestmove_san=None,
            children=[], root_stats=None, search_complete=False,
        )
        for v in [600, 1100, 2700, 5200, 9800]
    ]

    selected = _select_milestone_snapshots(snapshots, [500, 1000, 2500, 5000, 10000])

    assert [s.total_nodes_searched for s in selected] == [600, 1100, 2700, 5200, 9800]


def test_select_milestone_snapshots_allows_duplicates():
    """When Lc0's emission cadence is coarser than the requested
    milestones, multiple milestones map to the same snapshot — that's
    an honest representation of the data, not an error."""
    snapshots = [
        MCTSFindings(
            total_nodes_searched=v, bestmove_uci=None, bestmove_san=None,
            children=[], root_stats=None, search_complete=False,
        )
        for v in [1200, 9000]  # only two snapshots covering [500..10000]
    ]

    selected = _select_milestone_snapshots(snapshots, [500, 1000, 2500, 5000, 10000])

    # 500 → 1200 (only one closer than 9000); 1000 → 1200;
    # 2500 → 1200 (|2500-1200|=1300 < |2500-9000|=6500);
    # 5000 → 9000 (|5000-9000|=4000 < |5000-1200|=3800)... wait that's 3800.
    # |5000-1200|=3800, |5000-9000|=4000, so 5000 → 1200 too.
    # 10000 → 9000.
    nodes = [s.total_nodes_searched for s in selected]
    assert nodes == [1200, 1200, 1200, 1200, 9000]


def test_select_milestone_snapshots_empty_input_returns_empty():
    assert _select_milestone_snapshots([], [500, 1000, 10000]) == []


# === full extractor: happy path ============================================


def test_trajectory_assembled_from_three_emit_cycles(monkeypatch, board):
    """Three live-stats emit cycles + three milestones → trajectory has
    3 points each in milestone-ascending order, with correct values."""
    cycle1 = _emit_cycle([
        ("d8d4", 45, 0.85, 6.8),
        ("b8b1", 30, 0.70, 26.6),
    ], root_visits=500)
    cycle2 = _emit_cycle([
        ("d8d4", 300, 0.92, 6.8),
        ("b8b1", 25, 0.55, 26.6),
    ], root_visits=2500)
    cycle3 = _emit_cycle([
        ("d8d4", 850, 0.95, 6.8),
        ("b8b1", 20, 0.40, 26.6),
    ], root_visits=10000)

    stub = _StubEngine([_make_search_result(
        [cycle1, cycle2, cycle3], bestmove_uci="d8d4",
    )])

    result = extract_mcts_trajectory(
        board, engine=stub, budgets=[500, 2500, 10000],
    )

    # Engine was called exactly once with live-stats on, deepest budget.
    assert len(stub.calls) == 1
    assert stub.calls[0]["nodes"] == 10000
    assert stub.calls[0]["log_live_stats"] is True

    # Result is built from the deepest snapshot.
    assert result.total_nodes_searched == 10000
    assert result.bestmove_san == "Rxd4"

    rxd4 = next(c for c in result.children if c.uci == "d8d4")
    rxb1 = next(c for c in result.children if c.uci == "b8b1")

    # Current-snapshot stats reflect the deepest cycle.
    assert rxd4.visits == 850
    assert abs(rxd4.q - 0.95) < 1e-4

    # Trajectory: 3 points each, ascending by nodes_searched.
    assert len(rxd4.trajectory) == 3
    assert [p.nodes_searched for p in rxd4.trajectory] == [500, 2500, 10000]
    assert [p.visits for p in rxd4.trajectory] == [45, 300, 850]
    assert all(
        abs(p.q - expected) < 1e-4
        for p, expected in zip(rxd4.trajectory, [0.85, 0.92, 0.95])
    )

    assert len(rxb1.trajectory) == 3
    assert [p.visits for p in rxb1.trajectory] == [30, 25, 20]
    assert all(
        abs(p.q - expected) < 1e-4
        for p, expected in zip(rxb1.trajectory, [0.70, 0.55, 0.40])
    )


def test_default_budgets_used_when_not_specified(monkeypatch, board):
    """No ``budgets`` argument → ``DEFAULT_BUDGETS`` and engine is called
    once with the deepest as the node budget."""
    cycles = [
        _emit_cycle([("d8d4", b // 10, 0.5, 6.8)], root_visits=b)
        for b in DEFAULT_BUDGETS
    ]
    stub = _StubEngine([_make_search_result(cycles, bestmove_uci="d8d4")])

    result = extract_mcts_trajectory(board, engine=stub)

    assert stub.calls[0]["nodes"] == DEFAULT_BUDGETS[-1]
    assert stub.calls[0]["log_live_stats"] is True
    assert len(result.children[0].trajectory) == len(DEFAULT_BUDGETS)


# === edge cases ============================================================


def test_candidate_absent_from_early_snapshot_gets_zero_fill(monkeypatch, board):
    """A move that only appears in deeper snapshots gets zero-fill at
    the milestones where it wasn't reported (search hadn't reached it)."""
    cycle_early = _emit_cycle([
        ("d8d4", 45, 0.85, 6.8),
        # b8b1 absent — search hadn't visited it yet
    ], root_visits=500)
    cycle_deep = _emit_cycle([
        ("d8d4", 850, 0.95, 6.8),
        ("b8b1", 20, 0.40, 26.6),
    ], root_visits=10000)

    stub = _StubEngine([_make_search_result(
        [cycle_early, cycle_deep], bestmove_uci="d8d4",
    )])

    result = extract_mcts_trajectory(
        board, engine=stub, budgets=[500, 10000],
    )

    rxb1 = next(c for c in result.children if c.uci == "b8b1")
    assert len(rxb1.trajectory) == 2
    # Early snapshot zero-filled.
    assert rxb1.trajectory[0].nodes_searched == 500
    assert rxb1.trajectory[0].visits == 0
    assert rxb1.trajectory[0].q == 0.0
    # Deep snapshot populated.
    assert rxb1.trajectory[1].visits == 20
    assert abs(rxb1.trajectory[1].q - 0.40) < 1e-4


def test_candidate_dropped_from_deepest_snapshot_not_in_result(monkeypatch, board):
    """A move present in early snapshots but missing from the deepest
    snapshot's verbose-move-stats output is NOT included in the result
    (we iterate base.children, which is the deepest snapshot). This is
    intentional: if the deepest tree didn't surface a move, it's not
    relevant to the trajectory we're reporting."""
    cycle_early = _emit_cycle([
        ("d8d4", 45, 0.85, 6.8),
        ("a4c6", 15, 0.10, 5.4),  # only at low budget
    ], root_visits=500)
    cycle_deep = _emit_cycle([
        ("d8d4", 850, 0.95, 6.8),
        # Bc6 dropped from the deepest verbose-move-stats output.
    ], root_visits=10000)

    stub = _StubEngine([_make_search_result(
        [cycle_early, cycle_deep], bestmove_uci="d8d4",
    )])

    result = extract_mcts_trajectory(
        board, engine=stub, budgets=[500, 10000],
    )

    ucis = [c.uci for c in result.children]
    assert "a4c6" not in ucis
    assert "d8d4" in ucis


def test_single_milestone_call_produces_one_point_trajectory(monkeypatch, board):
    """A single-milestone call still works — one emit cycle, one point."""
    cycle = _emit_cycle([("d8d4", 850, 0.95, 6.8)], root_visits=10000)
    stub = _StubEngine([_make_search_result([cycle], bestmove_uci="d8d4")])

    result = extract_mcts_trajectory(board, engine=stub, budgets=[10000])

    assert len(result.children[0].trajectory) == 1
    point = result.children[0].trajectory[0]
    assert point.nodes_searched == 10000
    assert point.visits == 850
    assert abs(point.q - 0.95) < 1e-4


def test_coarser_cadence_than_milestones_produces_duplicates(monkeypatch, board):
    """When Lc0 only emits a few snapshots, requesting more milestones
    is fine — the closest-snapshot rule produces duplicate trajectory
    points. Honest representation of the data, not an error."""
    # Only two emit cycles, but five milestones requested.
    cycle1 = _emit_cycle([("d8d4", 50, 0.85, 6.8)], root_visits=1200)
    cycle2 = _emit_cycle([("d8d4", 800, 0.95, 6.8)], root_visits=9500)

    stub = _StubEngine([_make_search_result(
        [cycle1, cycle2], bestmove_uci="d8d4",
    )])

    result = extract_mcts_trajectory(
        board, engine=stub, budgets=[500, 1000, 2500, 5000, 10000],
    )

    traj = result.children[0].trajectory
    assert len(traj) == 5
    # 500/1000/2500/5000 closer to 1200; 10000 closer to 9500.
    # |5000-1200|=3800 vs |5000-9500|=4500 → 5000 → 1200.
    assert [p.nodes_searched for p in traj] == [1200, 1200, 1200, 1200, 9500]
    assert [p.visits for p in traj] == [50, 50, 50, 50, 800]


def test_live_stats_fallback_when_no_blocks_emitted(monkeypatch, board):
    """If --log-live-stats produced no parseable blocks (engine emitted
    no info-string content), the extractor falls back to a single
    end-of-search snapshot via extract_mcts."""
    # Empty info-string stream.
    empty_result = Lc0SearchResult(
        info_string_lines=[],
        bestmove_uci="d8d4",
        nodes_requested=10000,
        wall_seconds=1.0,
        search_complete=True,
    )
    stub = _StubEngine([empty_result])

    # Patch the fallback path's extract_mcts to return canned findings.
    fallback_findings = MCTSFindings(
        total_nodes_searched=10000,
        bestmove_uci="d8d4", bestmove_san="Rxd4",
        children=[MCTSChild(
            uci="d8d4", san="Rxd4",
            visits=850, visits_in_flight=0,
            prior_pct=6.8, q=0.95, v=0.95, wl=0.95,
            draw=0.01, moves_left=70.0, u=0.004, s=0.97, pv_san=[],
        )],
        root_stats=None,
        search_complete=True,
    )
    monkeypatch.setattr(
        mt, "extract_mcts",
        lambda board, *, engine, nodes: fallback_findings,
    )

    result = extract_mcts_trajectory(
        board, engine=stub, budgets=[500, 5000, 10000],
    )

    # Trajectory has one repeated point per milestone (flat = "we only
    # have one snapshot to report").
    rxd4 = result.children[0]
    assert len(rxd4.trajectory) == 3
    assert all(p.visits == 850 for p in rxd4.trajectory)
    assert all(p.nodes_searched == 10000 for p in rxd4.trajectory)


# === validation ============================================================


def test_empty_budgets_raises(board):
    stub = _StubEngine([])
    with pytest.raises(ValueError, match="non-empty"):
        extract_mcts_trajectory(board, engine=stub, budgets=[])


def test_zero_or_negative_budget_raises(board):
    stub = _StubEngine([])
    with pytest.raises(ValueError, match="positive"):
        extract_mcts_trajectory(board, engine=stub, budgets=[0, 1000])
    with pytest.raises(ValueError, match="positive"):
        extract_mcts_trajectory(board, engine=stub, budgets=[-1, 1000])


def test_non_ascending_budgets_raises(board):
    """Equal-adjacent and descending both rejected — strict ascent."""
    stub = _StubEngine([])
    with pytest.raises(ValueError, match="ascending"):
        extract_mcts_trajectory(board, engine=stub, budgets=[1000, 500])
    with pytest.raises(ValueError, match="ascending"):
        extract_mcts_trajectory(board, engine=stub, budgets=[1000, 1000])


# === schema sanity =========================================================


def test_returned_children_are_still_frozen_pydantic(monkeypatch, board):
    """Enriched MCTSChild instances must remain valid frozen Pydantic
    models. We use ``model_copy(update=...)``; assert future refactors
    don't break this."""
    cycle = _emit_cycle([("d8d4", 100, 0.90, 6.8)], root_visits=1000)
    stub = _StubEngine([_make_search_result([cycle], bestmove_uci="d8d4")])

    result = extract_mcts_trajectory(board, engine=stub, budgets=[1000])
    child = result.children[0]
    assert isinstance(child, MCTSChild)
    with pytest.raises((TypeError, ValueError)):
        child.visits = 999  # type: ignore[misc]
    for p in child.trajectory:
        assert isinstance(p, MCTSTrajectoryPoint)


def test_single_budget_extract_mcts_caller_leaves_trajectory_empty():
    """Sanity: trajectory field defaults to []. Existing Chunks 1-5
    callers of plain extract_mcts are unaffected by 6a."""
    child = MCTSChild(
        uci="e2e4", san="e4",
        visits=100, visits_in_flight=0,
        prior_pct=10.0, q=0.05, v=0.05, wl=0.05,
        draw=0.6, moves_left=180.0, u=0.05, s=0.10, pv_san=[],
    )
    assert child.trajectory == []
