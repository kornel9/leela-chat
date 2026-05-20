"""Shared dismissal-tag classification for focal-move handlers.

Used by ``why_this_move`` (in --detailed mode) and ``did_you_consider``
(always). Centralised here so both handlers apply identical thresholds
when deriving the ``search_dismissal`` tag from a focal candidate's
MCTS trajectory summary.

Methodology source-of-truth: ``leela_chat/extractors/mcts_trajectory.py``
module docstring + INTERPRETABILITY_NOTES.md §4.4. The trough vs first
comparison is the load-bearing refutation signature — NOT q_last, since
post-trough drift reflects "search no longer sampling that branch"
under PUCT redirect, not "the move recovering."

Empirically calibrated thresholds from the pod-validation session
(2026-05-19) on the walkthrough position:

- ``explored_and_refuted``: q_trough materially below q_first (dip
  <= -0.15) AND visit share modest (< 10% — search was running but
  PUCT redirected away). Rxb1+ on the walkthrough is the canonical
  case (q_first=+0.74, q_trough=+0.36, visit share ~ 7%).
- ``explored_and_confirmed``: q_peak materially above q_first
  (rise >= +0.10) AND visit share substantial (>= 25% — search
  committed). Rxd4 on the walkthrough is the canonical case
  (q_first=+0.56, q_peak=+0.96, visit share ~84%).
- ``never_explored``: visits <= 2 at the final block (search budget
  didn't meaningfully reach this candidate).
- ``unknown``: trajectory summary missing OR thresholds inconclusive.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from leela_chat.extractors.mcts_parser import MCTSChild, MCTSTrajectorySummary


SearchDismissalTag = Literal[
    "explored_and_refuted",
    "explored_and_confirmed",
    "never_explored",
    "unknown",
]


def classify_search_dismissal(
    focal_child: "MCTSChild | None",
    summary: "MCTSTrajectorySummary | None",
    root_visits: int,
) -> SearchDismissalTag:
    """Derive ``search_dismissal`` from a focal candidate's trajectory summary.

    Rules — see module docstring + INTERPRETABILITY_NOTES.md §4.4 for
    methodology. The trough vs first comparison is the load-bearing
    refutation signature, NOT q_last (post-trough drift = "search no
    longer sampling," not "move recovering").

    Args:
        focal_child: The focal move's ``MCTSChild`` from the deepest
            live-stats emit. ``None`` when MCTS didn't run or the focal
            wasn't present.
        summary: The focal move's trajectory summary across the search.
            ``None`` when trajectory data unavailable.
        root_visits: Total visits at the root at the deepest emit.

    Returns:
        One of the four ``SearchDismissalTag`` literals.
    """
    if focal_child is None or summary is None:
        return "unknown"
    if focal_child.visits <= 2:
        return "never_explored"
    visit_share = (focal_child.visits / root_visits) if root_visits else 0.0
    q_dip = summary.q_trough - summary.q_first
    q_rise = summary.q_peak - summary.q_first
    if q_dip <= -0.15 and visit_share < 0.10:
        return "explored_and_refuted"
    if q_rise >= 0.10 and visit_share >= 0.25:
        return "explored_and_confirmed"
    return "unknown"
