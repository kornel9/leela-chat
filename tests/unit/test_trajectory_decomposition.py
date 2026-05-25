"""Unit tests for ``extract_trajectory_decomposition`` (Chunk 7c.9).

Coverage matrix:
- Eligibility gate (no trajectory_summary / no trajectory_full / shallow
  trough / focal not in children) → None.
- Real trough + clear culprit → TrajectoryDecomposition with 3 snapshots.
- Edge cases (i_at = 0 / i_at at last block).
- No non-focal child shows a negative delta → None.
"""
from __future__ import annotations

import pytest

from leela_chat.extractors.mcts_parser import (
    MCTSChild, MCTSFindings, MCTSRootStats,
    MCTSTrajectoryPoint, MCTSTrajectorySummary,
)
from leela_chat.extractors.trajectory_decomposition import (
    TrajectoryDecomposition,
    TrajectorySnapshot,
    extract_trajectory_decomposition,
)


def _point(nodes_searched: int, visits: int, q: float) -> MCTSTrajectoryPoint:
    return MCTSTrajectoryPoint(nodes_searched=nodes_searched, visits=visits, q=q)


def _summary(
    q_first: float, q_last: float, q_trough: float, q_trough_at_visits: int = 100,
) -> MCTSTrajectorySummary:
    return MCTSTrajectorySummary(
        q_first=q_first, q_last=q_last,
        q_peak=max(q_first, q_last, q_trough),
        q_peak_at_visits=10,
        q_trough=q_trough,
        q_trough_at_visits=q_trough_at_visits,
        q_stabilized_at_visits=None,
        visit_growth_rate=10.0,
        q_crossed_zero=False,
        inflection_points=[],
        block_count=5,
    )


def _child(
    uci: str, san: str,
    visits: int = 100,
    pv_san: list[str] | None = None,
    trajectory_full: list[MCTSTrajectoryPoint] | None = None,
    trajectory_summary: MCTSTrajectorySummary | None = None,
) -> MCTSChild:
    return MCTSChild(
        uci=uci, san=san,
        visits=visits, visits_in_flight=0,
        prior_pct=5.0, q=0.5,
        v=0.5, wl=0.5, draw=0.05, moves_left=20.0,
        u=0.1, s=0.5,
        pv_san=pv_san or [],
        pv_san_truncated=False,
        trajectory=[],
        trajectory_selected=[],
        trajectory_full=trajectory_full or [],
        trajectory_summary=trajectory_summary,
    )


def _findings(children: list[MCTSChild]) -> MCTSFindings:
    return MCTSFindings(
        total_nodes_searched=1000,
        bestmove_uci=children[0].uci if children else None,
        bestmove_san=children[0].san if children else None,
        children=children,
        root_stats=MCTSRootStats(
            visits=1000, visits_in_flight=0,
            prior_pct=99.0,
            q=0.5, v=0.5, wl=0.5, draw=0.05, moves_left=20.0,
        ),
        search_complete=True,
        multipv_used=5,
    )


# === Eligibility gate ======================================================


def test_returns_none_when_focal_not_in_children():
    """Focal UCI not present → None (no crash, no exception)."""
    mcts = _findings([_child("a1a2", "Ka2")])
    result = extract_trajectory_decomposition(mcts, focal_uci="e2e4")
    assert result is None


def test_returns_none_when_focal_has_no_trajectory_summary():
    """Focal has no trajectory_summary (e.g., produced by snapshot-only
    extract_mcts) → None."""
    focal = _child("a1a2", "Ka2", trajectory_summary=None)
    other = _child("b1b2", "Kb2", trajectory_summary=None)
    result = extract_trajectory_decomposition(_findings([focal, other]), focal_uci="a1a2")
    assert result is None


def test_returns_none_when_focal_has_no_trajectory_full():
    """trajectory_summary present but trajectory_full empty → None."""
    summary = _summary(q_first=0.90, q_last=0.60, q_trough=0.50)
    focal = _child("a1a2", "Ka2", trajectory_full=[], trajectory_summary=summary)
    other = _child("b1b2", "Kb2", trajectory_full=[], trajectory_summary=None)
    result = extract_trajectory_decomposition(_findings([focal, other]), focal_uci="a1a2")
    assert result is None


def test_returns_none_when_trough_is_shallow():
    """Focal max_delta ≤ 0.10 (threshold) → None."""
    # q_first=0.92, q_last=0.95, q_trough=0.90 → max_delta = max(0.02, 0.05) = 0.05.
    summary = _summary(q_first=0.92, q_last=0.95, q_trough=0.90)
    focal_tf = [_point(100, 5, 0.92), _point(200, 50, 0.91), _point(300, 100, 0.90), _point(400, 150, 0.95)]
    focal = _child("a1a2", "Ka2", trajectory_full=focal_tf, trajectory_summary=summary)
    other_tf = [_point(100, 1, 0.40), _point(200, 2, 0.30), _point(300, 3, 0.20), _point(400, 4, 0.10)]
    other = _child("b1b2", "Kb2", trajectory_full=other_tf, trajectory_summary=None)
    result = extract_trajectory_decomposition(_findings([focal, other]), focal_uci="a1a2")
    assert result is None


def test_returns_none_when_i_at_is_zero():
    """If the trough lands at block 0 (no PRIOR block to compare against),
    no "Leela was exploring this and found it worse" story exists → None."""
    # q_first=0.50, q_last=0.95, q_trough=0.50 (block 0).
    summary = _summary(q_first=0.50, q_last=0.95, q_trough=0.50)
    focal_tf = [_point(100, 5, 0.50), _point(200, 50, 0.70), _point(300, 100, 0.95)]
    focal = _child("a1a2", "Ka2", trajectory_full=focal_tf, trajectory_summary=summary)
    other_tf = [_point(100, 1, 0.40), _point(200, 2, 0.20), _point(300, 3, 0.10)]
    other = _child("b1b2", "Kb2", trajectory_full=other_tf, trajectory_summary=None)
    result = extract_trajectory_decomposition(_findings([focal, other]), focal_uci="a1a2")
    assert result is None


def test_returns_none_when_no_culprit_has_negative_delta():
    """All non-focal children's Q stayed flat or rose between pre and at-trough.
    No "explored this and found it worse" candidate → None."""
    # Focal: drops at block 2.
    summary = _summary(q_first=0.95, q_last=0.92, q_trough=0.70)
    focal_tf = [_point(100, 5, 0.95), _point(200, 50, 0.85), _point(300, 100, 0.70), _point(400, 150, 0.92)]
    focal = _child("a1a2", "Ka2", trajectory_full=focal_tf, trajectory_summary=summary)
    # Other: Q only rose.
    other_tf = [_point(100, 1, 0.10), _point(200, 2, 0.20), _point(300, 3, 0.30), _point(400, 4, 0.40)]
    other = _child("b1b2", "Kb2", trajectory_full=other_tf, trajectory_summary=None)
    result = extract_trajectory_decomposition(_findings([focal, other]), focal_uci="a1a2")
    assert result is None


# === Happy path ============================================================


def test_returns_decomposition_with_clear_culprit():
    """Focal's Q drops at block 2; one non-focal candidate's Q ALSO drops
    between block 1 (pre) and block 2 (at) — that's the culprit."""
    # Focal: q_first=0.95 (block 0), peak then drop at block 2 (q_trough=0.60),
    # recovery to q_last=0.90 at block 3.
    summary = _summary(q_first=0.95, q_last=0.90, q_trough=0.60)
    focal_tf = [
        _point(100, 5, 0.95), _point(200, 50, 0.92),
        _point(300, 100, 0.60), _point(400, 150, 0.90),
    ]
    focal = _child("a1a2", "Ka2", trajectory_full=focal_tf, trajectory_summary=summary)
    # Culprit: Q drops sharply between block 1 and block 2.
    culprit_tf = [
        _point(100, 1, 0.40), _point(200, 5, 0.45),
        _point(300, 15, 0.10), _point(400, 18, 0.05),
    ]
    culprit = _child(
        "b1b2", "Kb2", visits=18,
        pv_san=["Kb2", "Nf3", "d4"],
        trajectory_full=culprit_tf, trajectory_summary=None,
    )
    # Innocent: Q just rose monotonically.
    innocent_tf = [
        _point(100, 1, 0.10), _point(200, 2, 0.20),
        _point(300, 3, 0.30), _point(400, 4, 0.40),
    ]
    innocent = _child(
        "c1c2", "Kc2", visits=4,
        pv_san=["Kc2", "e4"],
        trajectory_full=innocent_tf, trajectory_summary=None,
    )

    result = extract_trajectory_decomposition(
        _findings([focal, culprit, innocent]), focal_uci="a1a2",
    )
    assert result is not None
    assert isinstance(result, TrajectoryDecomposition)
    assert result.uci == "b1b2"
    assert result.san == "Kb2"
    assert result.pv_san == ["Kb2", "Nf3", "d4"]
    # 3 snapshots [pre, at, post] populated from culprit's trajectory_full.
    assert len(result.snapshots) == 3
    assert [s.phase for s in result.snapshots] == ["pre", "at", "post"]
    # pre = culprit_tf[1] (i_at = 2 → i_pre = 1).
    assert result.snapshots[0].q == pytest.approx(0.45)
    assert result.snapshots[0].visits == 5
    assert result.snapshots[0].nodes_searched == 200
    # at = culprit_tf[2].
    assert result.snapshots[1].q == pytest.approx(0.10)
    # post = culprit_tf[3].
    assert result.snapshots[2].q == pytest.approx(0.05)
    # q_delta_pre_to_trough = at - pre = 0.10 - 0.45 = -0.35.
    assert result.q_delta_pre_to_trough == pytest.approx(-0.35)


def test_picks_most_negative_when_multiple_culprits():
    """Two non-focal candidates both drop; the one with the steeper drop wins."""
    summary = _summary(q_first=0.95, q_last=0.90, q_trough=0.60)
    focal_tf = [
        _point(100, 5, 0.95), _point(200, 50, 0.92),
        _point(300, 100, 0.60), _point(400, 150, 0.90),
    ]
    focal = _child("a1a2", "Ka2", trajectory_full=focal_tf, trajectory_summary=summary)
    # Mild culprit: -0.15 between block 1 and 2.
    mild_tf = [_point(100, 1, 0.40), _point(200, 5, 0.45), _point(300, 15, 0.30), _point(400, 18, 0.30)]
    mild = _child(
        "b1b2", "Kb2",
        pv_san=["Kb2", "x"], trajectory_full=mild_tf, trajectory_summary=None,
    )
    # Worst culprit: -0.35 between block 1 and 2.
    worst_tf = [_point(100, 1, 0.40), _point(200, 5, 0.45), _point(300, 15, 0.10), _point(400, 18, 0.05)]
    worst = _child(
        "c1c2", "Kc2",
        pv_san=["Kc2", "y"], trajectory_full=worst_tf, trajectory_summary=None,
    )

    result = extract_trajectory_decomposition(
        _findings([focal, mild, worst]), focal_uci="a1a2",
    )
    assert result is not None
    assert result.uci == "c1c2"  # the worst culprit
    assert result.q_delta_pre_to_trough == pytest.approx(-0.35)


def test_i_at_at_last_block_pinches_post_to_same_index():
    """If the trough is at the last block, i_post = i_at (no later block).
    Snapshots[1].nodes_searched == Snapshots[2].nodes_searched in this case."""
    # Focal: trough at block 2 (last).
    summary = _summary(q_first=0.95, q_last=0.50, q_trough=0.50)
    focal_tf = [_point(100, 5, 0.95), _point(200, 50, 0.80), _point(300, 100, 0.50)]
    focal = _child("a1a2", "Ka2", trajectory_full=focal_tf, trajectory_summary=summary)
    culprit_tf = [_point(100, 1, 0.40), _point(200, 5, 0.45), _point(300, 15, 0.10)]
    culprit = _child(
        "b1b2", "Kb2",
        pv_san=["Kb2"], trajectory_full=culprit_tf, trajectory_summary=None,
    )
    result = extract_trajectory_decomposition(_findings([focal, culprit]), focal_uci="a1a2")
    assert result is not None
    # i_at = 2 (last); i_post collapses to 2.
    assert result.snapshots[1].nodes_searched == 300
    assert result.snapshots[2].nodes_searched == 300


# === Threshold override (caller can lower for tests) =======================


def test_threshold_override_lowers_eligibility_gate():
    """Caller can pass a custom threshold (e.g., 0.05 for a position with
    a smaller real trough). Mostly for tests; production gates on the
    public TROUGH_DELTA_THRESHOLD."""
    # max_delta = 0.07 — fails default 0.10, passes 0.05.
    summary = _summary(q_first=0.95, q_last=0.92, q_trough=0.88)
    focal_tf = [_point(100, 5, 0.95), _point(200, 50, 0.92), _point(300, 100, 0.88), _point(400, 150, 0.92)]
    focal = _child("a1a2", "Ka2", trajectory_full=focal_tf, trajectory_summary=summary)
    culprit_tf = [_point(100, 1, 0.40), _point(200, 5, 0.45), _point(300, 15, 0.15), _point(400, 18, 0.05)]
    culprit = _child(
        "b1b2", "Kb2",
        pv_san=["Kb2"], trajectory_full=culprit_tf, trajectory_summary=None,
    )
    # Default threshold (0.10) → None.
    assert extract_trajectory_decomposition(_findings([focal, culprit]), focal_uci="a1a2") is None
    # Lower threshold → returns a decomposition.
    result = extract_trajectory_decomposition(
        _findings([focal, culprit]), focal_uci="a1a2", threshold=0.05,
    )
    assert result is not None
    assert result.uci == "b1b2"
