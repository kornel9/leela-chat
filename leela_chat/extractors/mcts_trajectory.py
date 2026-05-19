"""Multi-budget MCTS extractor — search-time Q-trajectory.

Runs Lc0 MCTS at multiple node budgets and joins per-candidate
(visits, q) stats across the checkpoints. Output is the highest-budget
``MCTSFindings``, with each ``MCTSChild.trajectory`` populated with one
:class:`MCTSTrajectoryPoint` per checkpoint.

This is the *search-time* half of the dual-timescale dismissal
methodology described in ``docs/INTERPRETABILITY_NOTES.md`` (§4). The
network-time half lives in :mod:`leela_chat.extractors.layer_emergence`
(Chunk 6b).

Why multi-budget? See §4.3 of the methodology doc: a candidate's visit-
and-Q evolution across budgets is a window into search-time refutation
("policy looked promising, search found a hole") vs. confirmation
("policy was modest, but Q held up under deeper exploration").

Cost: 5 budgets [500, 1000, 2500, 5000, 10000] sum to ~19k nodes total.
Warm-engine latency on the pod GPU is ~2× a single 10k MCTS call
(roughly 5-8 s vs. 2-3 s). Acceptable for ``did_you_consider``; gated
behind a ``--detailed`` flag elsewhere.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from leela_chat.extractors.mcts import extract_mcts
from leela_chat.extractors.mcts_parser import (
    MCTSChild,
    MCTSFindings,
    MCTSTrajectoryPoint,
)

if TYPE_CHECKING:
    import chess

    from leela_chat.engine.lc0_engine import Lc0Engine


# Logarithmically-spaced default budgets. The first checkpoint (500) is
# small enough to catch "PUCT exploring policy priors before refutations
# propagate"; 10000 is the standard handler budget. Override per-call
# if a different cadence is needed.
DEFAULT_BUDGETS: tuple[int, ...] = (500, 1000, 2500, 5000, 10000)


def extract_mcts_trajectory(
    board: "chess.Board",
    *,
    engine: "Lc0Engine",
    budgets: tuple[int, ...] | list[int] = DEFAULT_BUDGETS,
) -> MCTSFindings:
    """Run MCTS at each budget and return findings enriched with per-
    candidate trajectories.

    Args:
        board: Position to analyse. Caller-owned; not mutated.
        engine: Live :class:`Lc0Engine` instance. Caller owns lifecycle.
        budgets: Ascending sequence of node budgets to run. Must be
            non-empty and strictly ascending; the highest budget's
            findings are the base of the returned object (bestmove,
            children order, root_stats).

    Returns:
        :class:`MCTSFindings` cloned from the highest-budget run, with
        each child's ``trajectory`` field populated. The trajectory is
        in budget-ascending order. Candidates absent from a lower-
        budget run get a zero-visit / zero-Q point at that checkpoint
        (i.e. "search didn't even touch this move yet").

    Raises:
        ValueError: if ``budgets`` is empty or not strictly ascending.
    """
    budgets_list = list(budgets)
    if not budgets_list:
        raise ValueError("budgets must be non-empty")
    if any(b <= 0 for b in budgets_list):
        raise ValueError("all budgets must be positive")
    if any(b1 >= b2 for b1, b2 in zip(budgets_list, budgets_list[1:])):
        raise ValueError("budgets must be strictly ascending")

    checkpoints: list[MCTSFindings] = []
    for nodes in budgets_list:
        checkpoints.append(extract_mcts(board, engine=engine, nodes=nodes))

    # Use the deepest-budget findings as the base. Its children are the
    # canonical list (some low-budget runs may not have reported moves
    # that the deeper search ultimately surfaces).
    base = checkpoints[-1]

    # Pre-index each checkpoint by UCI for O(1) per-candidate lookup.
    by_uci_per_checkpoint: list[dict[str, MCTSChild]] = [
        {c.uci: c for c in ck.children} for ck in checkpoints
    ]

    enriched_children: list[MCTSChild] = []
    for child in base.children:
        trajectory: list[MCTSTrajectoryPoint] = []
        for ck, by_uci in zip(checkpoints, by_uci_per_checkpoint):
            ck_child = by_uci.get(child.uci)
            if ck_child is None:
                # Move absent from this budget's verbose-move-stats —
                # treat as zero-visit / zero-Q ("search didn't reach
                # this candidate yet").
                trajectory.append(MCTSTrajectoryPoint(
                    nodes_searched=ck.total_nodes_searched,
                    visits=0,
                    q=0.0,
                ))
            else:
                trajectory.append(MCTSTrajectoryPoint(
                    nodes_searched=ck.total_nodes_searched,
                    visits=ck_child.visits,
                    q=ck_child.q,
                ))
        enriched_children.append(child.model_copy(update={"trajectory": trajectory}))

    logger.debug(
        f"mcts_trajectory: budgets={budgets_list} "
        f"top1={base.children[0].san if base.children else None} "
        f"top1_visit_trajectory="
        f"{[p.visits for p in enriched_children[0].trajectory] if enriched_children else []}"
    )

    return base.model_copy(update={"children": enriched_children})
