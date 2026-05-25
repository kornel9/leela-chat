"""Handler: did_you_consider.

Dual-timescale dismissal analysis for one focal move. Takes a position
+ a specific move and assembles findings spanning:

- Position-level signals (policy + WDL + Stockfish + MCTS) — mirroring
  ``why_this_move``.
- **Network-time emergence** of the focal move (per-layer rank
  trajectory across BT4's 15 layers — from
  ``extract_layer_emergence``).
- **Search-time refutation/confirmation** of the focal move (per-block
  Q + visit trajectory from a single Lc0 ``--log-live-stats`` call —
  from ``extract_mcts_trajectory``).
- Pre-computed dismissal tags (``network_dismissal``,
  ``search_dismissal``) so the LLM's prose-design job is "narrate the
  pre-classified case" rather than "derive the case from raw data."

Methodology source-of-truth lives in the module docstrings of
``leela_chat/extractors/layer_emergence.py`` (network-time) and
``leela_chat/extractors/mcts_trajectory.py`` (search-time). The
trough-prose contract in the latter is binding on the v1 prompt.

Per CLAUDE.md §3 architecture invariants:
- Handlers orchestrate extractors and assemble JSON for the LLM.
- Handlers do not contain chess logic.
- Handlers do not call the LLM directly. (The translation layer does.)
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Literal

import chess
from pydantic import BaseModel, ConfigDict, Field

from leela_chat.extractors.layer_emergence import (
    CandidateLayerTrajectory,
    LayerEmergenceResult,
    extract_layer_emergence,
)
from leela_chat.extractors.mcts_parser import (
    MCTSChild,
    MCTSFindings,
    MCTSTrajectorySummary,
)
from leela_chat.extractors.mcts_trajectory import (
    extract_mcts_trajectory,
    extract_mcts_trajectory_focused,
)
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
from leela_chat.handlers._dismissal import (
    SearchDismissalTag,
    classify_search_dismissal,
)
from leela_chat.handlers.why_this_move import (
    _find_focal_in_mcts,
    _find_focal_in_policy,
    _find_focal_in_stockfish,
)

# Backward-compat alias for tests that imported the underscored name from
# this module before the shared-helper refactor.
_classify_search_dismissal = classify_search_dismissal

if TYPE_CHECKING:
    import chess.engine
    from transformer_lens import HookedTransformer

    from leela_chat.engine.lc0_engine import Lc0Engine

# v2 → v3 (2026-05-24, Chunk 7b.1): handler-3 always-deep prompt now
# cites Leela's MCTS pv_san (populated for top-5 by Chunk 7b's MultiPv=5
# default, which this always-deep handler inherits). Strict-citation
# Guard 1 extended to mcts.children[*].pv_san; output structure item 6
# updated to prefer Leela's intended continuation for the engine pick.
# Per CLAUDE.md §3: v2.md retained.
# v3 → v4 (2026-05-24, Chunk 7c): focused-pass MCTS data added — five
# new fields (focused_total_nodes, focused_visits, focused_q,
# focused_pv_san, focused_trajectory_summary) surfacing a second MCTS
# pass with searchmoves constrained to the focal move. v4 prompt adds
# step 6 "Focused-pass deep dive" + extends strict-citation Guard 1 to
# focused_pv_san. Per CLAUDE.md §3: v3.md retained.
# Per the CLAUDE.md §8 bundling rule (Chunk 7b.1): this prompt-version
# bump ships in the SAME commit chain as the new findings fields it
# cites — NOT deferred to 7m. First behavioral test of the rule.
PROMPT_VERSION = 4
HANDLER_NAME = "did_you_consider"

# MCTS top-N to surface to the LLM (mirrors why_this_move).
MCTS_TOP_N_FOR_LLM = 10


class DidYouConsiderFindings(BaseModel):
    """Findings JSON for ``did_you_consider``.

    Combines position-level data (mirrored from ``WhyThisMoveFindings``)
    with two trajectory blocks for the focal move: **network-time**
    (per-layer rank evolution across BT4 layers) and **search-time**
    (per-block visits + Q evolution within one MCTS tree). Plus two
    pre-computed dismissal tags that the v1 prompt uses to short-
    circuit case-detection.
    """

    # === Position-level (mirror of WhyThisMoveFindings) ===
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
        description="Leela's policy-network top-N moves by policy weight.",
    )
    stockfish: StockfishResult = Field(
        ...,
        description="Stockfish classical-search analysis.",
    )
    mcts: MCTSFindings | None = Field(
        default=None,
        description=(
            "Leela MCTS findings — children truncated to "
            "``MCTS_TOP_N_FOR_LLM`` for prompt brevity. Each child also "
            "carries ``trajectory_selected`` / ``trajectory_summary``."
        ),
    )

    # === Focal-move callouts (mirror of WhyThisMoveFindings) ===
    focal_move_uci: str
    focal_move_san: str
    focal_is_engine_pick: bool

    focal_policy_pct: float | None
    focal_policy_rank: int | None
    focal_mcts_child: MCTSChild | None = Field(
        default=None,
        description=(
            "Focal move's MCTS stats from the deepest live-stats emit. "
            "The ``trajectory_*`` fields on this child carry the full "
            "intra-search history. None when Lc0 didn't run."
        ),
    )
    focal_mcts_rank_by_visits: int | None
    focal_stockfish_candidate: StockfishMoveCandidate | None = None
    focal_stockfish_rank: int | None

    # === Dual-timescale dismissal data ===
    focal_layer_trajectory: CandidateLayerTrajectory | None = Field(
        default=None,
        description=(
            "Network-time: per-layer policy rank trajectory for the "
            "focal move across BT4's layers. ``network_dismissal`` tag "
            "comes from this. None when the layer-emergence extractor "
            "was skipped (e.g. no BT4 model available)."
        ),
    )
    focal_mcts_trajectory_summary: MCTSTrajectorySummary | None = Field(
        default=None,
        description=(
            "Search-time: one-shot summary of the focal move's Q + "
            "visit evolution within one MCTS tree. q_trough is the "
            "load-bearing refutation signal (NOT q_last). None when "
            "MCTS skipped or focal not visited."
        ),
    )

    # === Pre-computed dismissal tags ===
    network_dismissal: (
        Literal["rose_and_stayed", "rose_and_fell", "never_rose", "unknown"]
        | None
    ) = Field(
        default=None,
        description=(
            "Pre-computed from ``focal_layer_trajectory``. None when the "
            "layer-emergence extractor was skipped."
        ),
    )
    search_dismissal: SearchDismissalTag | None = Field(
        default=None,
        description=(
            "Pre-computed from ``focal_mcts_trajectory_summary`` + "
            "``focal_mcts_child.visits`` + root visits. None when "
            "MCTS skipped."
        ),
    )

    # === Chunk 7c: focused-pass MCTS data ===
    # A second MCTS pass with searchmoves constrained to the focal move
    # only — the entire search budget concentrates below the focal move,
    # producing a deeper PV and a cleaner Q-trajectory within that subtree
    # than the normal pass (where the focal move shares budget with other
    # candidates). Used by the v4 prompt for the "deep dive on focal move"
    # prose section. All None when lc0_engine is None OR the focused pass
    # didn't return a matching child.
    focused_total_nodes: int | None = Field(
        default=None,
        description=(
            "Chunk 7c — total nodes searched in the focused pass. "
            "Useful for the prompt to frame 'with N focused nodes...'. "
            "Typically close to the normal-pass total (Lc0's SmartPruning "
            "may stop earlier when the constrained search stabilises). "
            "None when focused pass skipped."
        ),
    )
    focused_visits: int | None = Field(
        default=None,
        description=(
            "Chunk 7c — visits on the focal move in the focused tree. "
            "Typically 90%+ of focused_total_nodes since the focal move "
            "is the only allowed root candidate. Differs from "
            "``focal_mcts_child.visits`` in that THIS reflects the "
            "concentrated-budget search, not the shared normal pass. "
            "None when focused pass skipped or focal child not found "
            "in the focused-pass children."
        ),
    )
    focused_q: float | None = Field(
        default=None,
        description=(
            "Chunk 7c — final Q on the focal move from the focused tree. "
            "May refine the normal-pass Q with extra visits ('the focused "
            "dive confirms Q at +X with more budget'). None when focused "
            "pass skipped."
        ),
    )
    focused_pv_san: list[str] = Field(
        default_factory=list,
        description=(
            "Chunk 7c — Leela's intended continuation from the focal move "
            "under the focused (concentrated-budget) search. Typically "
            "DEEPER than the normal pass's per-PV pv_san for the same "
            "candidate (~15 plies vs ~10) because all budget went below "
            "the focal move. Empty when focused pass skipped, focal child "
            "missing, or PV walk truncated. All moves validator-trusted "
            "via the recursive _collect_trusted_moves walk."
        ),
    )
    focused_trajectory_summary: MCTSTrajectorySummary | None = Field(
        default=None,
        description=(
            "Chunk 7c — Q + visit trajectory summary of the focal move "
            "WITHIN the focused tree. Shows how Q evolved when search "
            "had to commit to this move only (no distractions from other "
            "candidates). Same MCTSTrajectorySummary shape as "
            "``focal_mcts_trajectory_summary`` (the normal-pass version); "
            "the two together let prose compare 'Q under normal search "
            "vs Q under focused concentration'. None when focused pass "
            "skipped or no trajectory data."
        ),
    )

    # frozen=True + extra="allow": see WhatIsLeelaSeeingFindings.model_config
    # for the rationale (Chunk 7a.3 extension-friendliness per ADR-0013).
    model_config = ConfigDict(frozen=True, extra="allow")


def handle_did_you_consider(
    board: chess.Board,
    move: chess.Move,
    *,
    model: "HookedTransformer",
    stockfish_engine: "chess.engine.SimpleEngine",
    lc0_engine: "Lc0Engine | None" = None,
    top_n: int = 5,
    stockfish_depth: int = 15,
    stockfish_top_n: int = 3,
    mcts_max_nodes: int = 10000,
    skip_layer_emergence: bool = False,
) -> DidYouConsiderFindings:
    """Build the findings JSON for ``did_you_consider`` on (board, move).

    **API asymmetry note (Chunk 7a.1):** unlike :func:`handle_what_is_leela_seeing`
    and :func:`handle_why_this_move`, this handler does NOT take a
    ``mode: Mode`` parameter. ``did_you_consider`` is always-deep by
    construction — its prompt contract requires the dismissal-tag /
    dual-timescale data the quick-mode config wouldn't produce. The
    Mode auto-upgrade happens in :func:`leela_chat.cli.__main__._answer_one`
    (logs INFO and proceeds with deep behavior for a ``mode="quick"`` request).
    If a future Chunk-8+ feature needs per-mode dispatch on this handler
    (e.g., a per-mode chess-content extractor set), ADD a ``mode: Mode``
    parameter here AND remove the auto-upgrade branch in ``_answer_one``
    — the contract has an exception today; new per-mode work undoes it.
    See ``leela_chat.handlers.Mode`` docstring for the always-deep
    upgrade semantics.

    Args:
        board: Position to assess.
        move: The focal move. Must be legal on ``board``.
        model: BT4 HookedTransformer (caller-managed).
        stockfish_engine: Live Stockfish SimpleEngine.
        lc0_engine: Optional live Lc0Engine. None → no MCTS trajectory
            (search-time half of dual-timescale gracefully degrades).
        top_n: Policy top-N to surface to the prompt.
        stockfish_depth: Stockfish search depth.
        stockfish_top_n: Stockfish multipv count.
        mcts_max_nodes: Lc0 search budget for the live-stats trajectory.
        skip_layer_emergence: Set True to skip the N+1 forward passes
            (useful for unit tests + diagnostic paths that don't need
            network-time data).

    Returns:
        :class:`DidYouConsiderFindings`.
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
    # - BT4 thread: network_inference + (unless skip_layer_emergence)
    #   extract_layer_emergence with focal_uci (sequential within thread).
    # - Stockfish thread.
    # - Lc0 thread: extract_mcts_trajectory (handler 3 is always-deep so
    #   trajectory is mandatory).
    def _bt4_thread() -> tuple:
        leela_inner = extract_network_inference(board, model=model, top_n=top_n)
        layer_em_inner: LayerEmergenceResult | None = None
        if not skip_layer_emergence:
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
            # Chunk 7b: handler 3 is always-deep; request MultiPv=5 so
            # the top-5 candidates carry their PV move sequences.
            mcts_future = ex.submit(
                extract_mcts_trajectory, board,
                engine=lc0_engine, max_nodes=mcts_max_nodes, multipv=5,
            )

    leela, layer_emergence = bt4_future.result()
    sf = sf_future.result()
    full_mcts: MCTSFindings | None = None
    mcts_truncated: MCTSFindings | None = None
    if lc0_engine is not None:
        full_mcts = mcts_future.result()
        mcts_truncated = full_mcts.model_copy(
            update={"children": list(full_mcts.children[:MCTS_TOP_N_FOR_LLM])},
        )

    # === Chunk 7c: focused MCTS pass (sequential after normal pass) ===
    # Run a second MCTS pass with searchmoves constrained to the focal
    # move only. SEQUENTIAL within the Lc0 thread (not parallel) — two
    # parallel calls on the same Lc0 engine would race; multiple engine
    # instances is overhead we don't need here. Per the 7c plan: parallel
    # Lc0 contention is 7d's experiment, not 7c's.
    focused_findings: MCTSFindings | None = None
    if lc0_engine is not None:
        focused_findings = extract_mcts_trajectory_focused(
            board, engine=lc0_engine, focal_move=move, max_nodes=mcts_max_nodes,
        )

    # Pre-extract focal-move-specific data from the focused pass for
    # convenient prompt citation (v4 cites focused_visits, focused_q,
    # focused_pv_san, focused_trajectory_summary directly).
    focused_total_nodes: int | None = None
    focused_visits: int | None = None
    focused_q: float | None = None
    focused_pv_san: list[str] = []
    focused_trajectory_summary: MCTSTrajectorySummary | None = None
    if focused_findings is not None:
        focused_total_nodes = focused_findings.total_nodes_searched
        focused_focal_child = next(
            (c for c in focused_findings.children if c.uci == focal_uci),
            None,
        )
        if focused_focal_child is not None:
            focused_visits = focused_focal_child.visits
            focused_q = focused_focal_child.q
            focused_pv_san = list(focused_focal_child.pv_san)
            focused_trajectory_summary = focused_focal_child.trajectory_summary

    # Focal-move callouts (reuse the helpers from why_this_move).
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

    # Engine pick: MCTS bestmove when MCTS ran, else policy top-1.
    if full_mcts is not None and full_mcts.bestmove_uci is not None:
        engine_pick_uci = full_mcts.bestmove_uci
    elif leela.top_moves:
        engine_pick_uci = leela.top_moves[0].uci
    else:
        engine_pick_uci = None
    focal_is_engine_pick = (engine_pick_uci is not None
                            and engine_pick_uci == focal_uci)

    # Pull the focal trajectory from layer_emergence's per-candidate list.
    focal_layer_trajectory: CandidateLayerTrajectory | None = None
    network_dismissal: str | None = None
    if layer_emergence is not None:
        focal_layer_trajectory = next(
            (c for c in layer_emergence.candidates if c.uci == focal_uci),
            None,
        )
        if focal_layer_trajectory is not None:
            network_dismissal = focal_layer_trajectory.network_dismissal

    # Pull the focal trajectory_summary from the MCTS child.
    focal_mcts_summary: MCTSTrajectorySummary | None = None
    search_dismissal: SearchDismissalTag | None = None
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

    return DidYouConsiderFindings(
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
        network_dismissal=network_dismissal,  # type: ignore[arg-type]
        search_dismissal=search_dismissal,
        # Chunk 7c focused-pass data.
        focused_total_nodes=focused_total_nodes,
        focused_visits=focused_visits,
        focused_q=focused_q,
        focused_pv_san=focused_pv_san,
        focused_trajectory_summary=focused_trajectory_summary,
    )


def templated_fallback(findings: DidYouConsiderFindings) -> str:
    """Mechanical rendering for when LLM validation fails twice.

    Grounded by construction — every number traces directly to the
    findings model. Respects the trough-prose contract (no "monotonic
    drop", no "Q recovered = move improved"; trough framing only).
    """
    eval_pawns = findings.eval_cp / 100.0
    eval_sign = "+" if findings.eval_cp >= 0 else ""

    # Focal headline.
    pick_phrase = (
        "This IS the engine's committed pick."
        if findings.focal_is_engine_pick
        else "This is NOT the engine's pick — see below."
    )

    # Engine actual pick (only when focal isn't the pick).
    engine_pick_part = ""
    if not findings.focal_is_engine_pick:
        if findings.mcts is not None and findings.mcts.bestmove_san:
            engine_pick_part = (
                f" Engine's actual pick: {findings.mcts.bestmove_san}."
            )
        elif findings.top_moves:
            engine_pick_part = (
                f" Engine's actual pick (policy only): "
                f"{findings.top_moves[0].san}."
            )

    # Network-time recap.
    network_part = "Network-time: unavailable."
    if findings.focal_layer_trajectory is not None:
        t = findings.focal_layer_trajectory
        network_part = (
            f"Network-time ({findings.network_dismissal}): "
            f"peak rank {t.peak_rank} at L={t.peak_layer}; "
            f"final-layer rank "
            f"{t.layer_trajectory[-1].rank if t.layer_trajectory else 'n/a'}."
        )

    # Search-time recap. Respect trough-prose contract: cite q_trough,
    # not q_last, when refutation signature is present.
    search_part = "Search-time: unavailable."
    if findings.focal_mcts_trajectory_summary is not None:
        s = findings.focal_mcts_trajectory_summary
        if findings.search_dismissal == "explored_and_refuted":
            search_part = (
                f"Search-time (explored_and_refuted): "
                f"Q hit a trough of {s.q_trough:+.2f} at v={s.q_trough_at_visits}; "
                f"visits stagnated near {findings.focal_mcts_child.visits if findings.focal_mcts_child else 'n/a'}."
            )
        elif findings.search_dismissal == "explored_and_confirmed":
            search_part = (
                f"Search-time (explored_and_confirmed): "
                f"Q peaked at {s.q_peak:+.2f} (v={s.q_peak_at_visits}); "
                f"visits accumulated to "
                f"{findings.focal_mcts_child.visits if findings.focal_mcts_child else 'n/a'}."
            )
        else:
            search_part = (
                f"Search-time ({findings.search_dismissal}): "
                f"q_first={s.q_first:+.2f} q_trough={s.q_trough:+.2f} "
                f"q_peak={s.q_peak:+.2f}; blocks={s.block_count}."
            )
    elif findings.search_dismissal == "never_explored":
        search_part = (
            "Search-time (never_explored): the search did not reach "
            "this candidate within its budget."
        )

    return (
        f"Asked about: {findings.focal_move_san} (uci {findings.focal_move_uci}). "
        f"{pick_phrase}{engine_pick_part} "
        f"Position eval (policy net): {eval_sign}{findings.eval_cp} cp "
        f"({eval_sign}{eval_pawns:.2f}). "
        f"{network_part} {search_part}"
    )
