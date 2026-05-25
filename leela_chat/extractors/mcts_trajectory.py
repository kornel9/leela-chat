"""Single-search MCTS Q-trajectory via Lc0's ``--log-live-stats``.

Source of truth for the **search-time methodology** in the dual-
timescale dismissal framework. Cross-cutting framing in
``docs/INTERPRETABILITY_NOTES.md``; prose contract in §3 of that file
applies to every consumer of this module's output.

This module is the *search-time* half of the dual-timescale
methodology. The *network-time* half is
:mod:`leela_chat.extractors.layer_emergence` (zero-ablation
logit-lens; created in Chunk 6b).

================================================================
Methodology
================================================================

**Goal.** For each candidate move in a position, record how its
(visits, Q) evolved within **one continuous MCTS tree** as the search
progressed. The data is a true intra-search history, not a stitched
composite of independent searches.

**Why not stitch independent searches?**

A first-cut implementation ran Lc0 K times at budgets [500, 1 000,
2 500, 5 000, 10 000] and stitched per-candidate (visits, Q) across
the K independent search results. We shipped that as Chunk 6a v0 and
rejected it after reviewer feedback:

- The K trees are not the same tree. MCTS expansion order depends on
  backend stochasticity, exploration-bonus tiebreaks, and network
  noise. Two 10 000-node runs can produce different visit
  distributions; *different-budget* runs produce *different-shape*
  trees that happened to end at different sizes.
- Prose built on stitched data over-claims. A sentence like *"Q
  dropped from +0.74 at 1 000 nodes to +0.53 at 10 000"* implies
  single-run continuity that the data didn't have.

**How we capture single-search trajectories.**

Lc0 v0.32.1 supports the ``--log-live-stats`` flag (UCI option
``LogLiveStats``). Per
``update_research/repos/lc0-master/src/search/classic/params.cc:294``:

    "Do VerboseMoveStats on every info update."

When on, Lc0 emits the full verbose-move-stats block — every
candidate's line plus a ``node`` summary line — on every UCI info
update during the search. The format is identical to end-of-search
verbose-move-stats, so the existing
:func:`leela_chat.extractors.mcts_parser.parse_verbose_move_stats`
reuses cleanly per block.

Block boundary detection: a line whose header parses to the literal
token ``node`` ends the current block. The next non-``node`` candidate
line opens a new block.

**Storage policy (Chunk 6a v2 decision: store everything).**

We retain ALL captured blocks on ``MCTSChild.trajectory_full`` (~20-100
blocks per 10k-node search; a few KB per candidate; cheap). On top we
compute:

- ``trajectory_summary: MCTSTrajectorySummary`` — peak-Q,
  peak-at-visits, trough-Q, trough-at-visits, q-stabilized-at-visits,
  visit-growth-rate, q-crossed-zero, inflection_points (smoothed
  local maxima/minima), block_count.
- ``trajectory_selected: list[MCTSTrajectoryPoint]`` — content-aware
  down-sampled selection (~5-8 points) for prompt use: always
  includes first, last, q-peak, q-trough, q-stabilized; remainder
  fills from inflection points / even spacing.
- ``trajectory: list[MCTSTrajectoryPoint]`` — legacy field equal to
  ``trajectory_selected`` when populated. Preserved for Chunks 1-5
  callers and the 6a v1 contract.

Why content-aware down-sampling beats fixed milestones? The inflection
narrative ("Q peaked at +0.74 around visit 800 then declined steadily
to +0.40") is more useful prose-fodder than five raw snapshots at
[500, 1000, 2500, 5000, 10000]. Same prompt budget, more informative
content.

**Backward compatibility.** A caller passing ``budgets=[500, 1000,
2500, 5000, 10000]`` (the v1 contract) gets the closest-snapshot rule
to populate ``trajectory_selected`` and ``trajectory``. A caller
omitting ``budgets`` (the v2 default) gets content-aware sampling. In
both cases ``trajectory_full`` and ``trajectory_summary`` are populated
identically.

**Limitations.**

- Lc0's live-stats emission cadence is not configurable at fine
  granularity. We get the cadence Lc0 chooses. **Empirical (pod
  2026-05-19, Lc0 v0.32.1):** a 10 000-node search on the Rxd4
  walkthrough produces ~25 blocks; the last emit lands around
  ~5 000 nodes (PUCT stops producing `info` updates once bestmove is
  confident, then plays out the remaining budget silently). So
  ``q_last`` in the summary represents "Q at the deepest live-stats
  emit," not "Q at the actual end-of-search." This is fine for
  methodology — the deliberation period is what the trajectory
  captures. Prose should not refer to absolute end-of-search Q.
- The ``LogLiveStats`` UCI option in Lc0 v0.32.1 is `ProOnly`
  visibility — accessible only via the ``--log-live-stats`` CLI flag,
  NOT via UCI ``setoption``. ``Lc0Engine`` spawns with the flag
  always-on; the per-call ``log_live_stats`` kwarg controls whether
  the wrapper returns the full multi-block stream or filters to the
  final emit cycle for single-snapshot callers.
- Fallback path: if the live-stats stream produces no parseable blocks
  (engine misconfiguration, format change), the extractor falls back
  to a single end-of-search snapshot via the standard
  :func:`extract_mcts` path and emits a flat (single-point-repeated)
  trajectory.
- We do **not** report visit-by-visit precision (one (visits, Q) per
  individual MCTS visit). That would need an Lc0 source patch and is
  not currently planned.

**Prose contract — cross-cutting (§3 of INTERPRETABILITY_NOTES.md).**
Outputs of this module are observational. Don't claim what search
"decided" / "realized" / "discovered." Describe what changed in the
trajectory; search is the source of *trajectory* facts; Stockfish is
the source of *content* facts (which move refutes which).

**Prose contract — Q-trajectory specific (§4.4 of
INTERPRETABILITY_NOTES.md).** Refuted moves show partial-U Q
trajectories — initial Q matches policy-prior optimism, DROPS to a
trough as search finds the refutation, then PARTIAL RECOVERY as PUCT
redirects visits elsewhere and the move stops being expanded.
**The recovery is NOT the move getting better.** It's "search no
longer looking." The refutation signature is the *trough* plus visit
stagnation, not q_last.

Forbidden phrasings about Q trajectory:

- "Q dropped monotonically from X to Y"
- "by visit N, Q had fallen to..." (when N is past the trough)
- "Q recovered to X" framed as positive ("move improved")
- "the move's Q stabilized at Y" when Y is post-trough drift
- "the trajectory ended at Q=Y" (q_last drift is not informative on
  its own; cite q_trough + visit stagnation)

Allowed phrasings about Q trajectory:

- "Q hit a trough of X around visit Y" (the trough is load-bearing)
- "then visits stagnated at ~N as PUCT redirected elsewhere"
- "the post-trough drift to Z is not Q recovering; it's search no
  longer sampling that branch"
- "the refutation signature is trough + visit stagnation, not the
  final Q value"
- "Q reached the trough at visit Y, and remained low while visits
  stagnated"

The 6c v1 prompt enforces this contract; this docstring is the
source-of-truth.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from leela_chat.extractors.mcts import extract_mcts
from leela_chat.extractors.mcts_parser import (
    MCTSChild,
    MCTSFindings,
    MCTSTrajectoryPoint,
    MCTSTrajectorySummary,
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


# === C2: summary, inflection-point detection, content-aware down-sample ====


# Maximum inflection points to retain. More than this indicates noise
# dominating signal; we cap and log.
_INFLECTION_MAX = 5

# Minimum magnitude of Q change between consecutive inflections to
# retain. Tuned against MCTS noise; tighten if false-positives appear in
# integration tests.
_INFLECTION_Q_MIN = 0.05

# Stabilization window: subsequent Q variation ≤ this for the remainder
# of the trajectory means "stabilized."
_STABILIZATION_TOL = 0.05


def _smooth_q(qs: list[float], *, window: int = 3) -> list[float]:
    """Centered moving-average smoothing on Q values.

    Window is clipped at boundaries (the first/last values use a
    shrinking window). Robust to MCTS noise without distorting peak
    locations significantly.
    """
    if not qs:
        return []
    half = window // 2
    smoothed: list[float] = []
    for i in range(len(qs)):
        lo = max(0, i - half)
        hi = min(len(qs), i + half + 1)
        smoothed.append(sum(qs[lo:hi]) / (hi - lo))
    return smoothed


def _detect_inflection_points(
    points: list[MCTSTrajectoryPoint],
    *,
    q_min: float = _INFLECTION_Q_MIN,
    max_count: int = _INFLECTION_MAX,
) -> list[MCTSTrajectoryPoint]:
    """Find smoothed local maxima / minima of Q across the trajectory.

    Returns the inflection points as `MCTSTrajectoryPoint`s drawn from
    the input (not synthetic). Filtered so consecutive inflections
    differ by at least ``q_min`` in magnitude. Capped at ``max_count``.

    A trajectory with fewer than 3 points has no inflections (you need
    a middle point to be a local extremum).
    """
    if len(points) < 3:
        return []
    qs = [p.q for p in points]
    smoothed = _smooth_q(qs)
    candidates: list[tuple[int, MCTSTrajectoryPoint]] = []
    for i in range(1, len(smoothed) - 1):
        prev, cur, nxt = smoothed[i - 1], smoothed[i], smoothed[i + 1]
        is_max = cur > prev and cur > nxt
        is_min = cur < prev and cur < nxt
        if is_max or is_min:
            candidates.append((i, points[i]))

    # Magnitude filter: walk in order, only keep an inflection if its Q
    # differs from the previous *retained* inflection (or the start) by
    # at least q_min.
    retained: list[MCTSTrajectoryPoint] = []
    last_q = points[0].q
    for _, point in candidates:
        if abs(point.q - last_q) >= q_min:
            retained.append(point)
            last_q = point.q
        if len(retained) >= max_count:
            logger.debug(
                f"_detect_inflection_points: capped at {max_count} "
                f"(noise-dominated trajectory likely)"
            )
            break
    return retained


def _find_q_stabilized_visits(
    points: list[MCTSTrajectoryPoint],
    *,
    tol: float = _STABILIZATION_TOL,
) -> int | None:
    """First visit-count after which subsequent Q variation stays within
    ±``tol`` for the remainder of the trajectory.

    "Subsequent" requires at least 2 points after the anchor; a single-
    point suffix is trivially "stable" (equals itself) and isn't
    informative — that's "we ran out of trajectory," not "Q settled."
    Returns None if no anchor has ≥2 stable successors.
    """
    if len(points) < 3:
        return None
    # Anchor must have at least 2 subsequent points so "stayed stable"
    # carries information. Iterate up to len(points) - 2.
    for i in range(len(points) - 2):
        anchor = points[i]
        suffix = points[i:]
        if all(abs(p.q - anchor.q) <= tol for p in suffix):
            return anchor.visits
    return None


def _compute_trajectory_summary(
    points: list[MCTSTrajectoryPoint],
) -> MCTSTrajectorySummary | None:
    """Build an :class:`MCTSTrajectorySummary` from a per-candidate full
    trajectory. Returns None if ``points`` is empty.

    Single pass over the sequence; cheap. Visit-growth-rate is the mean
    of consecutive-visit increments where positive (negative deltas, if
    they appear due to engine bookkeeping, are clamped to 0).
    """
    if not points:
        return None
    qs = [p.q for p in points]
    visits = [p.visits for p in points]

    q_peak_idx = qs.index(max(qs))
    q_trough_idx = qs.index(min(qs))

    deltas = [max(0, visits[i + 1] - visits[i]) for i in range(len(visits) - 1)]
    growth = sum(deltas) / len(deltas) if deltas else 0.0

    crossed_zero = any(
        (qs[i] >= 0) != (qs[i + 1] >= 0) for i in range(len(qs) - 1)
    )

    return MCTSTrajectorySummary(
        q_first=qs[0],
        q_last=qs[-1],
        q_peak=qs[q_peak_idx],
        q_peak_at_visits=visits[q_peak_idx],
        q_trough=qs[q_trough_idx],
        q_trough_at_visits=visits[q_trough_idx],
        q_stabilized_at_visits=_find_q_stabilized_visits(points),
        visit_growth_rate=growth,
        q_crossed_zero=crossed_zero,
        inflection_points=_detect_inflection_points(points),
        block_count=len(points),
    )


def _content_aware_downsample(
    points: list[MCTSTrajectoryPoint],
    *,
    target_n: int = 6,
    summary: MCTSTrajectorySummary | None = None,
) -> list[MCTSTrajectoryPoint]:
    """Select ~``target_n`` snapshots optimised for prose-relevance.

    Priority (dedup-by-``nodes_searched``):
    1. First-visit snapshot.
    2. Last snapshot.
    3. q-peak snapshot.
    4. q-trough snapshot.
    5. q-stabilized snapshot, if any.
    6. Inflection-point snapshots (in order).
    7. Filler points by even visit-domain spacing.

    Returns snapshots in nodes-ascending order.
    """
    if not points:
        return []
    if summary is None:
        summary = _compute_trajectory_summary(points)
        if summary is None:
            return []

    # Index points by nodes_searched (the dedup key).
    by_nodes: dict[int, MCTSTrajectoryPoint] = {p.nodes_searched: p for p in points}
    picks: list[MCTSTrajectoryPoint] = []
    seen: set[int] = set()

    def add(p: MCTSTrajectoryPoint | None) -> None:
        if p is None or p.nodes_searched in seen:
            return
        seen.add(p.nodes_searched)
        picks.append(p)

    # Priority 1-2: endpoints.
    add(points[0])
    add(points[-1])

    # Priorities 3-5: peak / trough / stabilized — find the point whose
    # visit count matches.
    def find_at_visits(v: int) -> MCTSTrajectoryPoint | None:
        for p in points:
            if p.visits == v:
                return p
        return None

    if len(picks) < target_n:
        add(find_at_visits(summary.q_peak_at_visits))
    if len(picks) < target_n:
        add(find_at_visits(summary.q_trough_at_visits))
    if len(picks) < target_n and summary.q_stabilized_at_visits is not None:
        add(find_at_visits(summary.q_stabilized_at_visits))

    # Priority 6: inflection points (already filtered).
    for inflection in summary.inflection_points:
        if len(picks) >= target_n:
            break
        add(inflection)

    # Priority 7: filler by even visit-domain spacing.
    remaining = target_n - len(picks)
    if remaining > 0 and len(points) > len(picks):
        # Pick remaining indices evenly spaced across the trajectory,
        # skipping already-seen.
        for i in range(1, len(points) - 1):
            if len(picks) >= target_n:
                break
            if points[i].nodes_searched not in seen:
                add(points[i])

    # Return sorted by nodes_searched ascending.
    return sorted(picks, key=lambda p: p.nodes_searched)


def _build_full_trajectory_for_candidate(
    uci: str,
    findings_per_block: list[MCTSFindings],
) -> list[MCTSTrajectoryPoint]:
    """Walk every block; build the candidate's full per-block trajectory.

    Skips blocks where the candidate isn't present (instead of zero-
    filling) — for the full trajectory we report only data we actually
    observed. The down-sample / summary handle the "absent at early
    blocks" case via the visit-growth and first-visit logic.
    """
    trajectory: list[MCTSTrajectoryPoint] = []
    for block in findings_per_block:
        by_uci = {c.uci: c for c in block.children}
        child = by_uci.get(uci)
        if child is None:
            continue
        trajectory.append(MCTSTrajectoryPoint(
            nodes_searched=block.total_nodes_searched,
            visits=child.visits,
            q=child.q,
        ))
    return trajectory


def extract_mcts_trajectory(
    board: "chess.Board",
    *,
    engine: "Lc0Engine",
    budgets: tuple[int, ...] | list[int] | None = None,
    max_nodes: int | None = None,
    target_selected_n: int = 6,
    multipv: int = 1,
    root_moves: "list[chess.Move] | None" = None,
) -> MCTSFindings:
    """Run one Lc0 search with live-stats; build a per-candidate dense
    trajectory of the *same tree* plus summary statistics plus a
    content-aware down-sample.

    Args:
        board: Position to analyse. Caller-owned; not mutated.
        engine: Live :class:`Lc0Engine` instance. Caller owns lifecycle.
        budgets: BACK-COMPAT. If provided, must be strictly ascending
            positive milestones — the deepest is the search budget, and
            ``trajectory_selected`` (and the legacy ``trajectory``
            field) is populated by closest-snapshot-to-each-milestone,
            preserving the v1 contract. If None (the v2 default),
            ``trajectory_selected`` is populated by content-aware down-
            sampling and the search runs to ``max_nodes`` or 10 000.
        max_nodes: Override the search-depth node budget. Used only when
            ``budgets`` is None. Defaults to 10 000.
        target_selected_n: Target number of points in
            ``trajectory_selected`` when using content-aware sampling.
            Ignored if ``budgets`` is given. Default 6.
        multipv: MultiPV value (Chunk 7b). ``1`` (default) = single-PV;
            byte-identical with pre-7b. ``>1`` = N principal variations
            captured; ``pv_san`` populated on matching root candidates
            via the same :func:`extract_mcts._populate_pv_san` helper.
            Per-handler-per-mode default: quick=1, deep=5.
        root_moves: Optional list of moves to constrain the root-level
            search (Chunk 7c). When set, Lc0's MCTS spends its entire
            budget on the named moves; forbidden root candidates stay
            at 0 visits. Use case: focused MCTS on a single candidate
            for ``did_you_consider``'s deep-dive prose. The single
            allowed move's ``pv_san`` will be populated (with a deeper
            line than a normal pass would produce since all budget
            concentrated there). Threaded through to
            :meth:`Lc0Engine.analyse_verbose`.

    Returns:
        :class:`MCTSFindings` from the *final* search state (deepest
        snapshot), with each child carrying ``trajectory_full`` (every
        live-stats block this candidate was in), ``trajectory_summary``
        (peak / trough / inflections / etc.), ``trajectory_selected``
        (prompt-facing down-sample), and the legacy ``trajectory``
        field set to ``trajectory_selected``.

    Raises:
        ValueError: if ``budgets`` is provided and is empty, non-
            positive, or not strictly ascending.
    """
    # --- argument validation + search-budget resolution -----------------
    budgets_list: list[int] | None = None
    if budgets is not None:
        budgets_list = list(budgets)
        if not budgets_list:
            raise ValueError("budgets must be non-empty when provided")
        if any(b <= 0 for b in budgets_list):
            raise ValueError("all budgets must be positive")
        if any(b1 >= b2 for b1, b2 in zip(budgets_list, budgets_list[1:])):
            raise ValueError("budgets must be strictly ascending")
        search_budget = budgets_list[-1]
    else:
        search_budget = max_nodes if max_nodes is not None else 10_000
        if search_budget <= 0:
            raise ValueError("max_nodes must be positive")

    # --- one Lc0 search; live-stats stream -------------------------------
    search_result = engine.analyse_verbose(
        board, nodes=search_budget, log_live_stats=True,
        multipv=multipv, root_moves=root_moves,
    )

    blocks = _group_live_stats_blocks(search_result.info_string_lines)

    findings_per_block: list[MCTSFindings] = []
    for i, block in enumerate(blocks):
        is_final = (i == len(blocks) - 1)
        findings_per_block.append(parse_verbose_move_stats(
            block,
            board=board,
            bestmove_uci=search_result.bestmove_uci if is_final else None,
            search_complete=is_final and search_result.search_complete,
        ))

    # --- fallback: no parseable blocks ----------------------------------
    if not findings_per_block:
        logger.warning(
            "mcts_trajectory: no live-stats blocks parsed; "
            "falling back to single end-of-search snapshot"
        )
        base = extract_mcts(board, engine=engine, nodes=search_budget)
        # If we got at least one Lc0Engine call's stats, synthesize a
        # one-point trajectory for each child.
        n_repeat = len(budgets_list) if budgets_list else max(1, target_selected_n)
        enriched_children: list[MCTSChild] = []
        for child in base.children:
            single_point = MCTSTrajectoryPoint(
                nodes_searched=base.total_nodes_searched,
                visits=child.visits,
                q=child.q,
            )
            full = [single_point]
            selected = [single_point] * n_repeat
            summary = _compute_trajectory_summary(full)
            enriched_children.append(child.model_copy(update={
                "trajectory_full": full,
                "trajectory_selected": selected,
                "trajectory_summary": summary,
                "trajectory": selected,
            }))
        return base.model_copy(update={"children": enriched_children})

    # --- normal path: dense capture + summary + selected ---------------
    base = findings_per_block[-1]

    enriched_children: list[MCTSChild] = []
    for child in base.children:
        full = _build_full_trajectory_for_candidate(child.uci, findings_per_block)
        if not full:
            # Candidate only appears in the final block? Synthesize a
            # one-point trajectory from the base data.
            full = [MCTSTrajectoryPoint(
                nodes_searched=base.total_nodes_searched,
                visits=child.visits,
                q=child.q,
            )]
        summary = _compute_trajectory_summary(full)

        if budgets_list is not None:
            # v1 back-compat: closest-snapshot per milestone.
            milestone_snaps = _select_milestone_snapshots(findings_per_block, budgets_list)
            selected: list[MCTSTrajectoryPoint] = []
            for snap in milestone_snaps:
                snap_child = next(
                    (c for c in snap.children if c.uci == child.uci), None,
                )
                if snap_child is None:
                    selected.append(_zero_point(snap.total_nodes_searched))
                else:
                    selected.append(MCTSTrajectoryPoint(
                        nodes_searched=snap.total_nodes_searched,
                        visits=snap_child.visits,
                        q=snap_child.q,
                    ))
        else:
            # v2 default: content-aware down-sample.
            selected = _content_aware_downsample(
                full, target_n=target_selected_n, summary=summary,
            )

        enriched_children.append(child.model_copy(update={
            "trajectory_full": full,
            "trajectory_selected": selected,
            "trajectory_summary": summary,
            "trajectory": selected,
        }))

    enriched_findings = base.model_copy(update={"children": enriched_children})

    # Chunk 7b: populate pv_san on matching root candidates when Lc0
    # reported principal variations. Gate RELAXED in 7c — was previously
    # multipv > 1; now fires whenever principal_variations is non-empty,
    # so focused passes (multipv=1 + root_moves) also get pv_san populated
    # on the constrained move. Reuse the helper from extract_mcts to keep
    # the SAN-conversion logic in one place.
    if search_result.principal_variations:
        from leela_chat.extractors.mcts import _populate_pv_san  # local import; avoid cycles
        enriched_findings = _populate_pv_san(
            enriched_findings, search_result.principal_variations, board,
        )

    logger.debug(
        f"mcts_trajectory: search_budget={search_budget} multipv={multipv} "
        f"emit_blocks={len(findings_per_block)} "
        f"top1={base.children[0].san if base.children else None} "
        f"top1_full_len="
        f"{len(enriched_children[0].trajectory_full) if enriched_children else 0} "
        f"top1_selected_len="
        f"{len(enriched_children[0].trajectory_selected) if enriched_children else 0} "
        f"pvs_populated={sum(1 for c in enriched_findings.children if c.pv_san)}"
    )

    return enriched_findings


def extract_mcts_trajectory_focused(
    board: "chess.Board",
    *,
    engine: "Lc0Engine",
    focal_move: "chess.Move",
    max_nodes: int = 10000,
    target_selected_n: int = 6,
) -> MCTSFindings:
    """Run focused MCTS constrained to a single root move (Chunk 7c).

    Thin wrapper around :func:`extract_mcts_trajectory` that pins
    ``multipv=1`` (only one root move is searchable so MultiPv > 1 is
    meaningless) and ``root_moves=[focal_move]`` (the focused-search
    constraint). All search budget concentrates below the focal move,
    yielding a deeper PV than a normal pass would produce + a cleaner
    Q-trajectory specifically for the focal move's subtree.

    Use case: ``did_you_consider`` handler's deep-dive prose section —
    "Leela's focused dive into X reached N visits with Q +Y; the
    intended continuation runs X A B C D E F".

    Per the 7c pod probe (``lc0_searchmoves_probe.py``): Lc0's verbose-
    move-stats output is IDENTICAL between focused and normal passes
    (parser doesn't need changes); forbidden root candidates emit their
    stats but stay at 0 visits; the deep PV lives in ``info["pv"]`` and
    gets attached to the focal child via the relaxed
    :func:`extract_mcts._populate_pv_san` gate (was multipv > 1; now
    any-PV-present).

    Args:
        board: Position to analyse. Caller-owned; not mutated.
        engine: Live :class:`Lc0Engine` instance. Caller owns lifecycle.
        focal_move: The single root move the search is constrained to.
        max_nodes: Search-budget node cap (default 10000, same as the
            normal pass). Lc0's SmartPruning may stop earlier if the
            evaluation stabilises.
        target_selected_n: Target down-sampled trajectory points for
            the focal child's ``trajectory_selected``.

    Returns:
        :class:`MCTSFindings` from the focused search. Caller typically
        looks up the focal child via UCI match
        (``next(c for c in findings.children if c.uci == focal_move.uci())``)
        and uses its ``visits``, ``q``, ``pv_san``, ``trajectory_summary``
        for the prompt's deep-dive section.
    """
    return extract_mcts_trajectory(
        board, engine=engine,
        max_nodes=max_nodes,
        target_selected_n=target_selected_n,
        multipv=1,
        root_moves=[focal_move],
    )
