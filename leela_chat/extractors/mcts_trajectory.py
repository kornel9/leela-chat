"""Single-search MCTS Q-trajectory via Lc0's ``--log-live-stats``.

Runs Lc0 *once* to the deepest requested visit budget with the
``LogLiveStats`` UCI option enabled. Lc0 then emits the full
verbose-move-stats block on every info update during the search — each
block is a complete snapshot of the *same continuous tree* at a
different point in its growth.

Per-candidate trajectories are stitched across snapshots of the same
tree (not across independent searches). For methodology see
``docs/INTERPRETABILITY_NOTES.md`` §4: in particular §4.2 on why
multi-search stitching was rejected, and §4.3-4.4 on milestone
selection.

Companion: :mod:`leela_chat.extractors.layer_emergence` is the
*network-time* half of the dual-timescale methodology (Chunk 6b).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from leela_chat.extractors.mcts import extract_mcts
from leela_chat.extractors.mcts_parser import (
    MCTSChild,
    MCTSFindings,
    MCTSTrajectoryPoint,
    _HEADER_RE,
    parse_verbose_move_stats,
)

if TYPE_CHECKING:
    import chess

    from leela_chat.engine.lc0_engine import Lc0Engine


# Default visit milestones for the trajectory. Logarithmically-spaced
# from "early-search" (where PUCT is still exploring policy priors) to
# "deep-search" (where Q values have propagated). Each is a target;
# milestone-matching is best-effort (closest snapshot wins). See §4.4
# of INTERPRETABILITY_NOTES.md.
DEFAULT_BUDGETS: tuple[int, ...] = (500, 1000, 2500, 5000, 10000)


def _group_live_stats_blocks(info_string_lines: list[str]) -> list[list[str]]:
    """Split a stream of info-string lines into per-emit-cycle blocks.

    With ``LogLiveStats`` on, Lc0 emits the full verbose-move-stats
    block on every info update: a sequence of candidate lines
    terminated by a ``node ...`` summary line. The next emit cycle
    starts with candidate lines again. This function splits at each
    ``node`` summary boundary.

    Lines that don't parse as either candidate or node-summary headers
    (diagnostic info-string chatter) are kept in the block they appear
    in — the parser will filter them downstream.

    Returns:
        A list of blocks, each a list of raw info-string lines. The
        last block may be without a terminating ``node`` line if the
        engine ended mid-cycle; it's still included if it contains any
        candidate lines.
    """
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in info_string_lines:
        current.append(line)
        # The mcts_parser's _HEADER_RE matches "node (idx)" as a
        # candidate-shaped header. We use a simpler prefix check here
        # to avoid double-parsing — the parser later does its own work.
        stripped = line.lstrip()
        if stripped.startswith("node ") or stripped.startswith("node\t"):
            # Verify it's the summary header pattern (not just any
            # word starting with "node") by running _HEADER_RE.
            m = _HEADER_RE.match(line)
            if m is not None and m.group("move") == "node":
                blocks.append(current)
                current = []
    # Tail without a closing `node` summary — keep if it has any
    # candidate-shaped content (otherwise drop trailing noise).
    if current and any(_HEADER_RE.match(line) for line in current):
        blocks.append(current)
    return blocks


def _select_milestone_snapshots(
    findings_per_block: list[MCTSFindings],
    milestones: list[int],
) -> list[MCTSFindings]:
    """For each milestone, pick the snapshot whose actual
    ``total_nodes_searched`` is closest. Duplicates allowed when
    Lc0's emission cadence is coarser than the requested milestones —
    that's an honest representation; the prose downstream uses each
    snapshot's actual nodes_searched, never the requested milestone.

    If ``findings_per_block`` is empty, returns an empty list and the
    caller gets a flat (no-trajectory) result for every candidate.
    """
    if not findings_per_block:
        return []
    selected: list[MCTSFindings] = []
    for m in milestones:
        best = min(
            findings_per_block,
            key=lambda f: abs(f.total_nodes_searched - m),
        )
        selected.append(best)
    return selected


def _zero_point(nodes_searched: int) -> MCTSTrajectoryPoint:
    """A 'search hadn't reached this candidate yet' sentinel."""
    return MCTSTrajectoryPoint(
        nodes_searched=nodes_searched,
        visits=0,
        q=0.0,
    )


def extract_mcts_trajectory(
    board: "chess.Board",
    *,
    engine: "Lc0Engine",
    budgets: tuple[int, ...] | list[int] = DEFAULT_BUDGETS,
) -> MCTSFindings:
    """Run one Lc0 search with live-stats; build a per-candidate
    trajectory of the *same tree* at the requested visit milestones.

    Args:
        board: Position to analyse. Caller-owned; not mutated.
        engine: Live :class:`Lc0Engine` instance. Caller owns lifecycle.
        budgets: Target visit milestones, in strictly-ascending order.
            The deepest value is the actual node budget passed to Lc0;
            shallower values are targets for picking intra-search
            snapshots from the live-stats stream. Default
            ``DEFAULT_BUDGETS``.

    Returns:
        :class:`MCTSFindings` from the *final* search state (deepest
        snapshot), with each child's ``trajectory`` field populated
        with one :class:`MCTSTrajectoryPoint` per requested milestone,
        in milestone-ascending order. Candidates absent from a
        particular snapshot get a zero-fill point at that milestone
        (search hadn't reached them yet). When Lc0's emission cadence
        misses a milestone entirely, the closest-snapshot rule applies
        and the trajectory may have repeated values.

    Raises:
        ValueError: if ``budgets`` is empty, non-positive, or not
            strictly ascending.
    """
    budgets_list = list(budgets)
    if not budgets_list:
        raise ValueError("budgets must be non-empty")
    if any(b <= 0 for b in budgets_list):
        raise ValueError("all budgets must be positive")
    if any(b1 >= b2 for b1, b2 in zip(budgets_list, budgets_list[1:])):
        raise ValueError("budgets must be strictly ascending")

    max_budget = budgets_list[-1]

    # One Lc0 search to the deepest milestone, with live-stats on.
    search_result = engine.analyse_verbose(
        board, nodes=max_budget, log_live_stats=True,
    )

    # Split the streamed info-string lines into per-emit-cycle blocks.
    blocks = _group_live_stats_blocks(search_result.info_string_lines)

    # Parse each block into MCTSFindings. The bestmove is only
    # meaningful on the final block (engine's chosen move at search
    # end); intermediate blocks pass bestmove_uci=None.
    findings_per_block: list[MCTSFindings] = []
    for i, block in enumerate(blocks):
        is_final = (i == len(blocks) - 1)
        findings_per_block.append(parse_verbose_move_stats(
            block,
            board=board,
            bestmove_uci=search_result.bestmove_uci if is_final else None,
            search_complete=is_final and search_result.search_complete,
        ))

    # Fallback: if live-stats produced no parseable blocks (Lc0 didn't
    # emit, or only the end-of-search block came through and that block
    # wasn't well-formed), fall through to a single end-of-search
    # snapshot via the existing extractor.
    if not findings_per_block:
        logger.warning(
            "mcts_trajectory: no live-stats blocks parsed; "
            "falling back to single end-of-search snapshot"
        )
        base = extract_mcts(board, engine=engine, nodes=max_budget)
        enriched_children: list[MCTSChild] = []
        for child in base.children:
            enriched_children.append(child.model_copy(update={
                "trajectory": [
                    MCTSTrajectoryPoint(
                        nodes_searched=base.total_nodes_searched,
                        visits=child.visits,
                        q=child.q,
                    )
                ] * len(budgets_list),
            }))
        return base.model_copy(update={"children": enriched_children})

    selected = _select_milestone_snapshots(findings_per_block, budgets_list)

    # The base findings are the deepest snapshot — that's where
    # downstream callers expect bestmove, children sorted by final
    # visits, etc.
    base = findings_per_block[-1]

    # Pre-index each selected snapshot by UCI for O(1) lookup.
    by_uci_per_snapshot: list[dict[str, MCTSChild]] = [
        {c.uci: c for c in snap.children} for snap in selected
    ]

    enriched_children: list[MCTSChild] = []
    for child in base.children:
        trajectory: list[MCTSTrajectoryPoint] = []
        for snap, by_uci in zip(selected, by_uci_per_snapshot):
            snap_child = by_uci.get(child.uci)
            if snap_child is None:
                trajectory.append(_zero_point(snap.total_nodes_searched))
            else:
                trajectory.append(MCTSTrajectoryPoint(
                    nodes_searched=snap.total_nodes_searched,
                    visits=snap_child.visits,
                    q=snap_child.q,
                ))
        enriched_children.append(child.model_copy(update={"trajectory": trajectory}))

    logger.debug(
        f"mcts_trajectory: max_budget={max_budget} "
        f"emit_blocks={len(findings_per_block)} "
        f"selected_snapshots={len(selected)} "
        f"top1={base.children[0].san if base.children else None} "
        f"top1_visit_trajectory="
        f"{[p.visits for p in enriched_children[0].trajectory] if enriched_children else []}"
    )

    return base.model_copy(update={"children": enriched_children})
