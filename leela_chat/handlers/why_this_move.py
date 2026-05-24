"""Handler: why_this_move.

Takes a position + a specific move and assembles findings for the LLM to
explain *why that move is best* (or why it isn't, when the focal move
isn't the engine's pick).

Same three-signal pipeline as ``what_is_leela_seeing`` (policy network +
Stockfish + MCTS) but with **focal-move callouts** pre-extracted: the
asked-about move's rank and stats from each source. The v1 prompt is then
free to write justification prose without scanning the candidate lists.

Per CLAUDE.md §3 architecture invariants:
- Handlers orchestrate extractors and assemble JSON for the LLM.
- Handlers do not contain chess logic. (The extractors do.)
- Handlers do not call the LLM directly. (The translation layer does.)

Design choices for Chunk 5 (Phase 2):
- Findings model owns the focal-move callout fields directly (no shared
  module across why_this_move + did_you_consider yet — premature
  abstraction; Chunk 6 will refactor if reuse is real).
- MCTS top-N is widened to 10 (vs 5 for Handler 1) so the focal move is
  more likely to appear in ``mcts.children`` even when it's not in the top
  few by visits. The pre-extracted ``focal_mcts_child`` field also
  guarantees the focal move's stats are available regardless.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Literal

import chess
from pydantic import BaseModel, ConfigDict, Field

from leela_chat.extractors.layer_emergence import (
    CandidateLayerTrajectory,
    NetworkDismissalTag,
    extract_layer_emergence,
)
from leela_chat.extractors.mcts import extract_mcts
from leela_chat.extractors.mcts_parser import (
    MCTSChild,
    MCTSFindings,
    MCTSTrajectorySummary,
)
from leela_chat.extractors.mcts_trajectory import extract_mcts_trajectory
from leela_chat.extractors.network_inference import (
    MoveCandidate,
    WDL,
    extract_network_inference,
)
from leela_chat.extractors.stockfish import (
    StockfishMoveCandidate,
    StockfishResult,
    extract_stockfish_eval,
)
from leela_chat.handlers import Mode
from leela_chat.handlers._dismissal import (
    SearchDismissalTag,
    classify_search_dismissal,
)

if TYPE_CHECKING:
    import chess.engine
    from transformer_lens import HookedTransformer

    from leela_chat.engine.lc0_engine import Lc0Engine

# Points at leela_chat/llm/prompts/why_this_move/v1.md.
# Per CLAUDE.md §3: increment, never delete old prompt versions.
# v1 -> v2 (2026-05-20): Chunk 6.5 --detailed mode adds dual-timescale
#   focal-move signals — the focal MCTS trajectory_summary (Q + visits
#   dynamics within one search) and the focal CandidateLayerTrajectory
#   (per-layer rank emergence across BT4). Both feed pre-computed
#   network_dismissal + search_dismissal tags that the v2 prompt
#   narrates with the trough-prose contract. Default mode (no
#   --detailed) keeps v1 byte-identical.
PROMPT_VERSION = 3
# v4 → v5 (2026-05-24, Chunk 7b.1): deep-mode prompt now cites Leela's
# MCTS pv_san (populated for top-5 candidates by Chunk 7b's MultiPv=5
# default). Strict-citation Guard 1 extended to mcts.children[*].pv_san;
# rule 5 inverted (pv_san is no longer "always []"); Branch A item 6 +
# Branch B item 6 updated to prefer Leela's PV over Stockfish's when
# available. Per CLAUDE.md §3: v4.md retained.
PROMPT_VERSION_DETAILED = 5

HANDLER_NAME = "why_this_move"

# Wider than Handler 1's top-5 — the focal move is more likely to land in
# ``mcts.children`` if we surface more rows. The pre-extracted
# ``focal_mcts_child`` field is the authoritative source for the focal
# move's stats regardless of whether it makes the top-N list.
MCTS_TOP_N_FOR_LLM = 10


class WhyThisMoveFindings(BaseModel):
    """Findings JSON for ``why_this_move``, focused on a specific candidate.

    Position-level fields mirror ``WhatIsLeelaSeeingFindings``. Focal-move
    callouts (``focal_*``) are pre-extracted from the pre-truncation
    candidate lists so the prompt doesn't need to scan to find the asked-
    about move. ``focal_*_rank`` fields are 1-based; None when the move
    isn't present in that source's reported set.
    """

    # === Position-level (mirror of WhatIsLeelaSeeingFindings) ===
    fen: str = Field(..., description="The position, in FEN.")
    side_to_move: Literal["white", "black"]
    eval_cp: int = Field(
        ...,
        description=(
            "Leela's policy-network centipawn evaluation. Positive = White better."
        ),
    )
    wdl: WDL
    moves_left_estimate: float = Field(..., ge=0.0)
    top_moves: list[MoveCandidate] = Field(
        ...,
        description="Leela's policy network top-N moves by policy weight, desc.",
    )
    stockfish: StockfishResult = Field(
        ...,
        description="Stockfish classical-search analysis, depth ~15.",
    )
    mcts: MCTSFindings | None = Field(
        default=None,
        description=(
            "Leela MCTS findings. Children truncated to the top "
            "``MCTS_TOP_N_FOR_LLM`` by visits for prompt brevity; the "
            "focal move's stats live in ``focal_mcts_child`` regardless "
            "of whether the move makes that top-N cut."
        ),
    )

    # === Focal move callouts ===
    focal_move_uci: str = Field(
        ..., description="The move the user asked about, in UCI.",
    )
    focal_move_san: str = Field(
        ..., description="The same move in SAN, computed on the root board.",
    )
    focal_is_engine_pick: bool = Field(
        ...,
        description=(
            "True iff the focal move is the engine's committed pick — "
            "matches ``mcts.bestmove_uci`` when MCTS is present, else "
            "matches ``top_moves[0].uci``."
        ),
    )

    focal_policy_pct: float | None = Field(
        ...,
        description=(
            "Focal move's policy_pct from the network's first-impression. "
            "None if the move is outside the policy top-N reported in "
            "``top_moves``."
        ),
    )
    focal_policy_rank: int | None = Field(
        ...,
        description=(
            "1-based rank of the focal move within the policy top-N, or "
            "None if not in the reported list."
        ),
    )
    focal_mcts_child: MCTSChild | None = Field(
        ...,
        description=(
            "Focal move's MCTS stats (visits, q, v, prior, etc.). None "
            "when MCTS not run, or when Lc0 didn't expand this move at "
            "all within the search budget."
        ),
    )
    focal_mcts_rank_by_visits: int | None = Field(
        ...,
        description=(
            "1-based rank of the focal move in MCTS children sorted by "
            "visits desc, or None if MCTS not run / move not reported."
        ),
    )
    focal_stockfish_candidate: StockfishMoveCandidate | None = Field(
        ...,
        description=(
            "Focal move's Stockfish stats (eval_cp / mate_in, pv_san), or "
            "None if not in Stockfish's reported top-N."
        ),
    )
    focal_stockfish_rank: int | None = Field(
        ...,
        description=(
            "1-based rank of the focal move in Stockfish's top moves, or None."
        ),
    )

    # === Chunk 6.5: --detailed-mode dual-timescale signals ===
    # All four default to None in v1 (default) mode; populated when
    # ``handle_why_this_move(..., detailed=True)``. The v2 prompt reads
    # these fields; v1 ignores them (default-mode byte-identity).
    focal_layer_trajectory: CandidateLayerTrajectory | None = Field(
        default=None,
        description=(
            "Network-time: per-layer rank trajectory for the focal move "
            "across BT4's 15 layers (--detailed mode only). None when "
            "the layer-emergence extractor wasn't run."
        ),
    )
    focal_mcts_trajectory_summary: MCTSTrajectorySummary | None = Field(
        default=None,
        description=(
            "Search-time: one-shot summary of the focal move's Q + "
            "visit evolution within one MCTS tree (--detailed mode "
            "only). ``q_trough`` is the load-bearing refutation signal "
            "per the trough-prose contract; NOT ``q_last``. None when "
            "MCTS-trajectory wasn't run or focal not visited."
        ),
    )
    network_dismissal: NetworkDismissalTag | None = Field(
        default=None,
        description=(
            "Pre-computed from ``focal_layer_trajectory.network_dismissal``. "
            "One of ``rose_and_stayed`` / ``rose_and_fell`` / "
            "``never_rose`` / ``unknown``. None in default mode or when "
            "the focal move wasn't in the layer-emergence top-N."
        ),
    )
    search_dismissal: SearchDismissalTag | None = Field(
        default=None,
        description=(
            "Pre-computed from ``focal_mcts_trajectory_summary`` + "
            "``focal_mcts_child.visits`` + root visits via "
            "``classify_search_dismissal``. None in default mode."
        ),
    )

    # frozen=True + extra="allow": see WhatIsLeelaSeeingFindings.model_config
    # for the rationale (Chunk 7a.3 extension-friendliness per ADR-0013).
    model_config = ConfigDict(frozen=True, extra="allow")


def _find_focal_in_policy(
    top_moves: list[MoveCandidate], focal_uci: str,
) -> tuple[float | None, int | None]:
    """Locate the focal move in the policy top-N; return (policy_pct, rank)."""
    for i, m in enumerate(top_moves, start=1):
        if m.uci == focal_uci:
            return m.policy_pct, i
    return None, None


def _find_focal_in_mcts(
    children: list[MCTSChild] | None, focal_uci: str,
) -> tuple[MCTSChild | None, int | None]:
    """Locate the focal move in the MCTS children (already sorted by visits)."""
    if not children:
        return None, None
    for i, c in enumerate(children, start=1):
        if c.uci == focal_uci:
            return c, i
    return None, None


def _find_focal_in_stockfish(
    sf_top: list[StockfishMoveCandidate], focal_uci: str,
) -> tuple[StockfishMoveCandidate | None, int | None]:
    """Locate the focal move in Stockfish's top-N; return (candidate, rank)."""
    for i, c in enumerate(sf_top, start=1):
        if c.uci == focal_uci:
            return c, i
    return None, None


def handle_why_this_move(
    board: chess.Board,
    move: chess.Move,
    *,
    model: "HookedTransformer",
    stockfish_engine: "chess.engine.SimpleEngine",
    lc0_engine: "Lc0Engine | None" = None,
    top_n: int = 5,
    stockfish_depth: int = 15,
    stockfish_top_n: int = 3,
    mcts_nodes: int = 10000,
    mode: Mode = "quick",
) -> WhyThisMoveFindings:
    """Build the findings JSON for ``why_this_move`` on the given position+move.

    Args:
        board: Position to assess.
        move: The move the user is asking about. Must be legal on ``board``;
            raises ``ValueError`` otherwise.
        model: BT4 HookedTransformer (caller-managed).
        stockfish_engine: Live Stockfish SimpleEngine.
        lc0_engine: Optional live Lc0Engine for MCTS. None → handler still
            produces findings but ``mcts``, ``focal_mcts_*`` fields are None;
            the v3 prompt degrades gracefully (policy + Stockfish only).
        top_n: Policy top-N to surface to the prompt.
        stockfish_depth: Stockfish search depth.
        stockfish_top_n: Stockfish multipv count.
        mcts_nodes: Lc0 search budget.
        mode: Per-handler latency mode (per ADR-0012). ``"quick"`` uses
            :func:`extract_mcts` snapshot and skips layer_emergence — the v3
            prompt reads the quick-mode subset. ``"deep"`` swaps in
            :func:`extract_mcts_trajectory` so the focal child carries a
            ``trajectory_summary``, AND calls :func:`extract_layer_emergence`
            with ``focal_uci=move.uci()`` to populate the focal layer
            trajectory. Also derives ``network_dismissal`` (from the layer
            trajectory) and ``search_dismissal`` (via
            :func:`classify_search_dismissal`). The v4 prompt
            (PROMPT_VERSION_DETAILED) reads the dual-timescale signals.
            Default ``"quick"`` preserves the v3 byte-identical contract
            (what the old ``detailed=False`` did).

    Returns:
        :class:`WhyThisMoveFindings`.
    """
    if move not in board.legal_moves:
        raise ValueError(
            f"Move {move.uci()} is not legal on FEN {board.fen()}. "
            "Caller must validate move legality before calling the handler."
        )

    focal_uci = move.uci()
    focal_san = board.san(move)

    # Per Chunk 7a.2 (extractor parallelisation): spawn one thread per
    # ENGINE so the slowest engine sets the handler floor. Three engines:
    # - BT4 thread: extract_network_inference + (deep mode) extract_layer_emergence
    #   in sequence (both touch HookedTransformer).
    # - Stockfish thread: extract_stockfish_eval.
    # - Lc0 thread: extract_mcts or extract_mcts_trajectory.
    # The layer-emergence call needs focal_uci which is known here (set above
    # from move.uci()), so this differs from Handler 1's _bt4_thread.
    def _bt4_thread() -> tuple:
        leela_inner = extract_network_inference(board, model=model, top_n=top_n)
        layer_em_inner = None
        if mode == "deep":
            layer_em_inner = extract_layer_emergence(
                board, model=model, focal_uci=focal_uci, top_n_per_layer=top_n,
            )
        return leela_inner, layer_em_inner

    with ThreadPoolExecutor(max_workers=3) as ex:
        bt4_future = ex.submit(_bt4_thread)
        sf_future = ex.submit(
            extract_stockfish_eval, board,
            engine=stockfish_engine,
            depth=stockfish_depth,
            top_n=stockfish_top_n,
        )
        if lc0_engine is not None:
            if mode == "deep":
                # Chunk 7b: deep mode requests MultiPv=5 — see
                # what_is_leela_seeing for the rationale.
                mcts_future = ex.submit(
                    extract_mcts_trajectory, board,
                    engine=lc0_engine, max_nodes=mcts_nodes, multipv=5,
                )
            else:
                mcts_future = ex.submit(
                    extract_mcts, board,
                    engine=lc0_engine, nodes=mcts_nodes,
                )

    leela, layer_emergence = bt4_future.result()
    sf = sf_future.result()
    full_mcts: MCTSFindings | None = None
    mcts_truncated: MCTSFindings | None = None
    if lc0_engine is not None:
        full_mcts = mcts_future.result()
        # Truncate children for the prompt; full list still available for
        # focal-move lookup via the pre-extracted focal_mcts_child below.
        mcts_truncated = full_mcts.model_copy(
            update={"children": list(full_mcts.children[:MCTS_TOP_N_FOR_LLM])},
        )

    focal_policy_pct, focal_policy_rank = _find_focal_in_policy(
        leela.top_moves, focal_uci,
    )
    focal_mcts_child, focal_mcts_rank_by_visits = _find_focal_in_mcts(
        full_mcts.children if full_mcts is not None else None,
        focal_uci,
    )
    focal_stockfish_candidate, focal_stockfish_rank = _find_focal_in_stockfish(
        sf.top_moves, focal_uci,
    )

    # Determine engine pick: MCTS bestmove when MCTS ran, else policy top-1.
    if full_mcts is not None and full_mcts.bestmove_uci is not None:
        engine_pick_uci = full_mcts.bestmove_uci
    elif leela.top_moves:
        engine_pick_uci = leela.top_moves[0].uci
    else:
        engine_pick_uci = None  # terminal position; shouldn't happen if move legal
    focal_is_engine_pick = (engine_pick_uci is not None
                            and engine_pick_uci == focal_uci)

    # === Chunk 6.5 (Chunk 7 ADR-0012-renamed): deep-mode dual-timescale signals ===
    # Note (Chunk 7a.2): layer_emergence is now pre-fetched in _bt4_thread
    # above (parallel with Stockfish + Lc0), so this block just consumes it
    # instead of running an extract_layer_emergence call inline.
    focal_layer_trajectory: CandidateLayerTrajectory | None = None
    network_dismissal: NetworkDismissalTag | None = None
    focal_mcts_summary: MCTSTrajectorySummary | None = None
    search_dismissal: SearchDismissalTag | None = None
    if mode == "deep" and layer_emergence is not None:
        focal_layer_trajectory = next(
            (c for c in layer_emergence.candidates if c.uci == focal_uci),
            None,
        )
        if focal_layer_trajectory is not None:
            network_dismissal = focal_layer_trajectory.network_dismissal
        if focal_mcts_child is not None:
            focal_mcts_summary = focal_mcts_child.trajectory_summary
            root_visits = (
                full_mcts.root_stats.visits
                if full_mcts is not None and full_mcts.root_stats is not None
                else 0
            )
            search_dismissal = classify_search_dismissal(
                focal_mcts_child, focal_mcts_summary, root_visits,
            )
        elif full_mcts is not None:
            # Focal move not present in deepest snapshot → search never reached.
            search_dismissal = "never_explored"

    return WhyThisMoveFindings(
        fen=leela.fen,
        side_to_move=leela.side_to_move,
        eval_cp=leela.eval_cp,
        wdl=leela.wdl,
        moves_left_estimate=leela.moves_left_estimate,
        top_moves=leela.top_moves,
        stockfish=sf,
        mcts=mcts_truncated,
        focal_move_uci=focal_uci,
        focal_move_san=focal_san,
        focal_is_engine_pick=focal_is_engine_pick,
        focal_policy_pct=focal_policy_pct,
        focal_policy_rank=focal_policy_rank,
        focal_mcts_child=focal_mcts_child,
        focal_mcts_rank_by_visits=focal_mcts_rank_by_visits,
        focal_stockfish_candidate=focal_stockfish_candidate,
        focal_stockfish_rank=focal_stockfish_rank,
        focal_layer_trajectory=focal_layer_trajectory,
        focal_mcts_trajectory_summary=focal_mcts_summary,
        network_dismissal=network_dismissal,
        search_dismissal=search_dismissal,
    )


def templated_fallback(findings: WhyThisMoveFindings) -> str:
    """Mechanical rendering for when LLM validation fails twice.

    Per CLAUDE.md §2 Rule 2: never show an unvalidated response. This is
    grounded by construction — every number traces directly to the
    findings model. Reads robotically; that's acceptable for the
    fallback path.
    """
    eval_pawns = findings.eval_cp / 100.0
    eval_sign = "+" if findings.eval_cp >= 0 else ""
    wdl = findings.wdl

    # Focal move headline.
    pick_phrase = (
        "This IS the engine's committed pick."
        if findings.focal_is_engine_pick
        else "This is NOT the engine's pick — see below."
    )

    # Focal move per-source recap.
    focal_lines: list[str] = []
    if findings.focal_mcts_child is not None:
        c = findings.focal_mcts_child
        focal_lines.append(
            f"MCTS: rank {findings.focal_mcts_rank_by_visits} by visits, "
            f"N={c.visits} (+{c.visits_in_flight}), Q={c.q:+.2f}, "
            f"prior {c.prior_pct:.1f}%"
        )
    elif findings.mcts is not None:
        focal_lines.append("MCTS: search did not expand this move.")
    else:
        focal_lines.append("MCTS: not available (no Lc0 in this run).")

    if findings.focal_policy_rank is not None:
        focal_lines.append(
            f"Policy: rank {findings.focal_policy_rank} at "
            f"{findings.focal_policy_pct:.1f}%"  # type: ignore[union-attr]
        )
    else:
        focal_lines.append("Policy: outside the reported top moves.")

    if findings.focal_stockfish_candidate is not None:
        c = findings.focal_stockfish_candidate
        if c.mate_in is not None:
            sf_eval = f"mate in {abs(c.mate_in)} plies"
        elif c.eval_cp is not None:
            sf_sign = "+" if c.eval_cp >= 0 else ""
            sf_eval = f"{sf_sign}{c.eval_cp} cp"
        else:
            sf_eval = "unavailable"
        focal_lines.append(
            f"Stockfish: rank {findings.focal_stockfish_rank}, eval after this move {sf_eval}"
        )
    else:
        focal_lines.append("Stockfish: outside its reported top moves.")

    focal_summary = "; ".join(focal_lines)

    # Engine actual pick (only mentioned when focal isn't the engine pick).
    engine_pick_part = ""
    if not findings.focal_is_engine_pick:
        if findings.mcts is not None and findings.mcts.bestmove_san:
            engine_pick_part = (
                f" Engine's actual pick: {findings.mcts.bestmove_san}."
            )
        elif findings.top_moves:
            engine_pick_part = (
                f" Engine's actual pick (policy only): {findings.top_moves[0].san}."
            )

    return (
        f"Asked about: {findings.focal_move_san} (uci {findings.focal_move_uci}). "
        f"{pick_phrase}{engine_pick_part} "
        f"Position eval (policy net): {eval_sign}{findings.eval_cp} cp "
        f"({eval_sign}{eval_pawns:.2f}); side to move: {findings.side_to_move}; "
        f"WDL {wdl.white_wins * 100:.0f}% W / {wdl.draw * 100:.0f}% D / "
        f"{wdl.black_wins * 100:.0f}% B. "
        f"Focal move stats — {focal_summary}."
    )
