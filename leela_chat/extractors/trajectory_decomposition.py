"""Trough-decomposition extractor (Chunk 7c.9).

When the focal child's MCTS trajectory has a meaningful trough (Q dipped
by ≥ ``TROUGH_DELTA_THRESHOLD`` between q_first/q_last and q_trough), this
extractor identifies which OTHER root candidate Leela was sampling at the
moment of the dip — the "Leela was exploring this and found it worse than
expected, so PUCT redirected back to the focal" candidate. Produces a
``TrajectoryDecomposition`` the prompt can cite to write a WHY-narration
on the trough, rather than just observing "Q dipped to +X."

Algorithm (per 7c.9 plan + pre-flight findings):

1. Find focal child by UCI; if no ``trajectory_summary`` or
   max(|q_trough - q_first|, |q_trough - q_last|) ≤ threshold, return None.
2. Locate i* = index in focal's ``trajectory_full`` where q is closest to
   ``trajectory_summary.q_trough``. i_pre = max(0, i*-1); i_post =
   min(N-1, i*+1). (Children's trajectories are index-aligned — same length,
   same nodes_searched at each block — so we can index directly.)
3. For each non-focal child with a non-empty trajectory_full: compute the
   per-child Q delta between block i_pre and i*. Sort by delta ascending.
4. Pick the most-negative delta as the "culprit." If all deltas ≥ 0, no
   clear culprit → return None.
5. Build TrajectoryDecomposition: culprit's literal uci/san/pv_san fields
   (walker-trusted by literal key match — see 7c.9 pre-flight) plus a
   3-element ``snapshots`` list (pre/at/post) with rich per-snapshot
   {phase, nodes_searched, visits, q}.

**Per-snapshot PVs are NOT captured by Lc0 --log-live-stats** (only the
final state has PVs via ``info["pv"]``). The schema reflects this — only
the candidate's FINAL ``pv_san`` is surfaced, not per-block PVs. See
7c.9 pre-flight findings in PHASE_NOTES for the structural rationale.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from leela_chat.extractors.mcts_parser import MCTSFindings
from leela_chat.validator import TROUGH_DELTA_THRESHOLD


class TrajectorySnapshot(BaseModel):
    """One snapshot of a candidate's state at a focal-trough-aligned block.

    Three snapshots compose a TrajectoryDecomposition: pre-trough, at-trough,
    post-trough. Each phase records the candidate's visits + Q at that
    block, plus the search-wide nodes_searched (a block identifier shared
    across all children).
    """

    phase: Literal["pre", "at", "post"] = Field(
        ..., description="Which trough-aligned block: pre, at, or post.",
    )
    nodes_searched: int = Field(
        ..., ge=0,
        description=(
            "Search-wide total nodes at this block. Same across children "
            "for the same block index; lets the prompt cite block timing."
        ),
    )
    visits: int = Field(
        ..., ge=0,
        description="The candidate's visits at this block.",
    )
    q: float = Field(
        ..., ge=-1.0, le=1.0,
        description="The candidate's Q at this block (side-to-move perspective).",
    )

    model_config = ConfigDict(frozen=True, extra="allow")


class TrajectoryDecomposition(BaseModel):
    """The "Leela was exploring this and found it worse than expected" candidate.

    Five fields. Walker-compatible literal key names (``uci`` / ``san`` /
    ``pv_san``) so the validator's ``_collect_trusted_moves`` picks them
    up without extension (verified by 7c.9 pre-flight). Prompt cites the
    candidate's identity + its Q evolution across snapshots + its final
    pv_san to explain WHY the focal's Q dipped at the trough.

    **Forward-compat note (added 7d entry, 2026-05-25):** a sibling
    ``RecoveryExplanation`` companion field on DidYouConsiderFindings
    is a planned future addition to handle the i_at=0 case (focal Q
    starts low at search opening, rises as search redirects). 7c.9's
    algorithm returns None for that case (no PRIOR block to compare
    against; the trough-decomposition framing doesn't apply). The
    recovery case is interesting on its own terms — "Leela's first
    impression was wrong; search corrected" — and 7d's exploration
    pass is exactly the dynamic that will surface it. The schema
    leaves room: ``trough_explanation`` and ``recovery_explanation``
    can coexist as sibling fields on the findings model. No code
    changes needed in this module; the note is here so future-me
    knows the design accommodates it without re-thinking the schema.
    """

    uci: str = Field(
        ..., description="Culprit candidate's UCI move (walker-trusted literal key).",
    )
    san: str = Field(
        ..., description="Culprit candidate's SAN move (walker-trusted literal key).",
    )
    pv_san: list[str] = Field(
        default_factory=list,
        description=(
            "Culprit candidate's FINAL pv_san (Leela's settled view of "
            "the line after the trough resolved). Per-snapshot PVs are "
            "NOT captured by Lc0 --log-live-stats (see 7c.9 pre-flight "
            "findings); final-only is the structural answer. Walker-"
            "trusted by literal `pv_san` key match."
        ),
    )
    snapshots: list[TrajectorySnapshot] = Field(
        ..., min_length=3, max_length=3,
        description=(
            "Pre / at / post-trough snapshots of the culprit candidate's "
            "state. Length always 3, ordered [pre, at, post]. Lets the "
            "prompt cite the candidate's Q evolution + visit counts at "
            "the trough timing."
        ),
    )
    q_delta_pre_to_trough: float = Field(
        ...,
        description=(
            "How much the CULPRIT's Q dropped between snapshots[0] (pre) "
            "and snapshots[1] (at). Convenience scalar; computable from "
            "snapshots but the prompt benefits from one direct reference. "
            "Will always be ≤ 0 by construction (the culprit is the "
            "child with the most-negative pre→at delta)."
        ),
    )

    model_config = ConfigDict(frozen=True, extra="allow")


def _find_closest_index(values: list[float], target: float) -> int:
    """Index of the entry in ``values`` whose value is closest to ``target``.

    Linear scan; ties broken by smallest index (numpy.argmin convention).
    Caller guarantees ``values`` is non-empty.
    """
    best_i = 0
    best_d = abs(values[0] - target)
    for i in range(1, len(values)):
        d = abs(values[i] - target)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def extract_trajectory_decomposition(
    mcts: MCTSFindings,
    focal_uci: str,
    *,
    threshold: float = TROUGH_DELTA_THRESHOLD,
) -> TrajectoryDecomposition | None:
    """Build a TrajectoryDecomposition for the focal child's trough, or None.

    Args:
        mcts: MCTSFindings from a trajectory-capable extractor (i.e.,
            ``extract_mcts_trajectory``, not ``extract_mcts``). Each
            child must carry ``trajectory_full`` for the algorithm to
            work; an mcts produced by snapshot-only ``extract_mcts``
            will short-circuit to None.
        focal_uci: The candidate UCI to treat as the trough target.
            Handlers pass their focal move.
        threshold: Trough-delta threshold from
            :data:`leela_chat.validator.TROUGH_DELTA_THRESHOLD`. Test
            callers may override.

    Returns:
        TrajectoryDecomposition when (1) focal is in children, (2) focal
        has a trajectory_summary with a meaningful trough (max_delta >
        threshold), (3) at least one non-focal child has aligned
        trajectory_full data showing a negative pre→at-trough Q-delta.
        None otherwise (the prompt then skips the WHY narration cleanly).
    """
    # Locate focal child.
    focal = next((c for c in mcts.children if c.uci == focal_uci), None)
    if focal is None or focal.trajectory_summary is None:
        return None
    if not focal.trajectory_full:
        return None

    # Meaningful-trough gate.
    s = focal.trajectory_summary
    max_delta = max(abs(s.q_trough - s.q_first), abs(s.q_trough - s.q_last))
    if max_delta <= threshold:
        return None

    # Locate the focal's trough block index.
    focal_q_series = [p.q for p in focal.trajectory_full]
    i_at = _find_closest_index(focal_q_series, s.q_trough)
    i_pre = max(0, i_at - 1)
    i_post = min(len(focal.trajectory_full) - 1, i_at + 1)

    # Edge case: if i_at is at index 0, "pre" collapses to the same point
    # → the per-child delta would be 0 for everyone, and we'd return None.
    # That's structurally honest: a trough that starts at block 0 means
    # the search opened poorly; there's no PRIOR state to compare against,
    # so no "Leela was exploring this and found it worse" story to tell.
    if i_at == 0:
        return None

    # Score non-focal children by Q-delta between i_pre and i_at.
    # Children's trajectory_full are index-aligned (same length, same
    # nodes_searched at each block) — verified by inspection of
    # extract_mcts_trajectory's output. Children that for any reason
    # have shorter trajectories are skipped defensively.
    needed_len = max(i_pre, i_at, i_post) + 1
    candidates: list[tuple[float, int]] = []  # (delta, child_index)
    for idx, child in enumerate(mcts.children):
        if child.uci == focal_uci:
            continue
        if len(child.trajectory_full) < needed_len:
            continue
        delta = child.trajectory_full[i_at].q - child.trajectory_full[i_pre].q
        candidates.append((delta, idx))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0])  # ascending → most-negative first
    most_negative_delta, culprit_idx = candidates[0]

    # No culprit if no one's Q dropped: the focal's trough wasn't caused
    # by exploration of a different move that turned out worse than
    # expected — it was internal noise or an inflection in the focal's
    # own subtree. Prose can't tell the "Leela was exploring this" story
    # without a clear redirection. Honest answer: None.
    if most_negative_delta >= 0:
        return None

    culprit = mcts.children[culprit_idx]

    snapshots = [
        TrajectorySnapshot(
            phase="pre",
            nodes_searched=culprit.trajectory_full[i_pre].nodes_searched,
            visits=culprit.trajectory_full[i_pre].visits,
            q=culprit.trajectory_full[i_pre].q,
        ),
        TrajectorySnapshot(
            phase="at",
            nodes_searched=culprit.trajectory_full[i_at].nodes_searched,
            visits=culprit.trajectory_full[i_at].visits,
            q=culprit.trajectory_full[i_at].q,
        ),
        TrajectorySnapshot(
            phase="post",
            nodes_searched=culprit.trajectory_full[i_post].nodes_searched,
            visits=culprit.trajectory_full[i_post].visits,
            q=culprit.trajectory_full[i_post].q,
        ),
    ]

    return TrajectoryDecomposition(
        uci=culprit.uci,
        san=culprit.san,
        pv_san=list(culprit.pv_san),
        snapshots=snapshots,
        q_delta_pre_to_trough=most_negative_delta,
    )
