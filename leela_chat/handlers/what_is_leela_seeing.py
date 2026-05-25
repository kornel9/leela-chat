"""Handler: what_is_leela_seeing.

Orchestrates the network-inference and Stockfish extractors and assembles the
findings JSON that the translation layer will pass to the LLM.

Per CLAUDE.md §3 architecture invariants:
- Handlers orchestrate extractors and assemble JSON for the LLM.
- Handlers do not contain chess logic. (The extractors do.)
- Handlers do not call the LLM directly. (The translation layer does.)
  They return the findings JSON; the translation layer calls the LLM.

Stockfish was pulled in alongside Leela's policy network per ADR-0009 after
the 2026-05-18 audit found policy-only output missing moves classical search
catches (position 4: Rxd4). The two extractors run independently; the LLM
prompt explicitly contrasts them.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Literal

import chess
from pydantic import BaseModel, ConfigDict, Field

from leela_chat.extractors.adaptive_multipv import (
    compute_adaptive_multipv,
    policy_entropy_normalized,
)
from leela_chat.extractors.layer_emergence import (
    LayerEmergenceResult,
    extract_layer_emergence,
)
from leela_chat.extractors.mcts import extract_mcts
from leela_chat.extractors.mcts_parser import MCTSFindings
from leela_chat.extractors.mcts_trajectory import extract_mcts_trajectory
from leela_chat.extractors.network_inference import (
    MoveCandidate,
    WDL,
    extract_network_inference,
)
from leela_chat.extractors.stockfish import (
    StockfishResult,
    extract_stockfish_eval,
)
from leela_chat.handlers import Mode

if TYPE_CHECKING:
    import chess.engine
    from transformer_lens import HookedTransformer

    from leela_chat.engine.lc0_engine import Lc0Engine

# Points at leela_chat/llm/prompts/what_is_leela_seeing/v4.md.
# Per CLAUDE.md §3: increment, never delete old prompt versions.
# v1 -> v2 (2026-05-18): tightened the precision rule for percentages
#   after smoke testing found Sonnet rounding `1.2%` to `1%`, which the
#   validator (correctly) flagged as ungrounded. v2 explicitly requires
#   one decimal place for `policy_pct` and bans threshold phrases like
#   `under 1%`.
# v2 -> v3 (2026-05-18): Stockfish comparison added to findings (ADR-0009).
#   v3 reframes "Leela" as "Leela's policy network" (no MCTS yet), introduces
#   Stockfish as an independent classical-search reference, requires comparing
#   the two and explicitly noting disagreements (>0.5 cp eval gap or different
#   top-move).
# v3 -> v4 (2026-05-19): MCTS findings added (Phase 2 Chunk 4). The prompt
#   now teaches a THREE-signal framing — policy first-impression, MCTS
#   committed-judgment via visit shares, and Stockfish — with the visit
#   share as the load-bearing metric for "what Leela actually plays."
#   Worked example pinned on the Rxd4 walkthrough case (the canonical
#   policy-vs-search disagreement).
# v4 -> v5 (2026-05-20): Chunk 6.5 --detailed mode adds two trajectory
#   signals — the MCTS Q + visit live-stats stream (per-child
#   trajectory_summary) and the BT4 per-layer rank trajectories
#   (layer_emergence). Default mode (no --detailed) keeps v4 unchanged.
PROMPT_VERSION = 6
# v7 → v8 (2026-05-24, Chunk 7b.1): deep-mode prompts learn to cite
# Leela's MCTS principal variations (`mcts.children[i].pv_san`) populated
# for top-5 candidates by Chunk 7b's MultiPv=5 deep-mode default. New
# section "When to cite Leela's MCTS PV vs Stockfish's"; both worked
# examples updated with pv_san data; four new anti-patterns. Validator
# unchanged (recursive walk already trusts pv_san). Per CLAUDE.md §3:
# v7.md retained.
# v8 → v9 (2026-05-25, Chunk 7c.8): adaptive MultiPv per ADR-0015.
# Replaces v8's fixed MultiPv=5 with policy-character-driven choice
# (dominant: 2, moderate: 5, contested/flat: 8). New top-of-prompt
# section "v9 addition: adaptive MultiPv citation" + the "PLUS the v8
# addition" line reframed for v9. Per the bundling rule (CLAUDE.md §8):
# this prompt bump ships in the same commit chain as the new
# `mcts.multipv_used` findings field. Per CLAUDE.md §3: v8.md retained.
PROMPT_VERSION_DETAILED = 9

# How many top-by-visits MCTS children to surface to the LLM in this
# handler's findings. The extractor returns all root candidates (up to ~50);
# the LLM only needs the leading few. Phase-6-relevant raw data is still
# computed in extract_mcts but not all of it flows into Handler 1's prompt.
MCTS_TOP_N_FOR_LLM = 5

HANDLER_NAME = "what_is_leela_seeing"


class WhatIsLeelaSeeingFindings(BaseModel):
    """Findings JSON for ``what_is_leela_seeing``, passed to the translation layer.

    Rule 1 (CLAUDE.md §2): the LLM is allowed to mention exactly the contents
    of this object — nothing more. The validator (Rule 2) checks the LLM's
    output against these fields and the underlying position.

    Two analysis sources live alongside each other: Leela's policy network
    (single forward pass, no MCTS) and Stockfish (classical search, depth-N).
    The prompt template instructs the LLM to compare them and surface
    disagreements. See ADR-0009.
    """

    fen: str = Field(..., description="The position, in FEN.")
    side_to_move: Literal["white", "black"]
    eval_cp: int = Field(
        ...,
        description=(
            "Leela's policy-network centipawn evaluation (one forward pass, "
            "no MCTS). Positive = White better."
        ),
    )
    wdl: WDL
    moves_left_estimate: float = Field(..., ge=0.0)
    top_moves: list[MoveCandidate] = Field(
        ...,
        description=(
            "Leela's POLICY NETWORK top-N legal moves by policy weight, "
            "sorted descending. This is Leela's prior — what MCTS would "
            "search first — not a search-refined posterior."
        ),
    )
    stockfish: StockfishResult = Field(
        ...,
        description=(
            "Stockfish's classical-search analysis of the same position, "
            "depth ~15. Independent reference to compare against Leela's "
            "policy. See ADR-0009 for why this lives in Phase 1."
        ),
    )
    mcts: MCTSFindings | None = Field(
        default=None,
        description=(
            "Leela's MCTS search findings — the engine's committed "
            "judgment after expanding a search tree (visit-weighted), as "
            "opposed to the policy network's first-impression in "
            "``top_moves``. None when this handler is invoked without an "
            "Lc0 engine (e.g. test paths). When present, children are "
            "truncated to the top ``MCTS_TOP_N_FOR_LLM`` by visits; the "
            "extractor's full output stays available to Handler 3 "
            "(`did_you_consider`) which needs the long tail. In "
            "--detailed mode each child also carries ``trajectory_*`` "
            "fields populated from the live-stats stream."
        ),
    )
    layer_emergence: LayerEmergenceResult | None = Field(
        default=None,
        description=(
            "Network-time per-layer rank trajectories for the policy "
            "top-N (Chunk 6.5 --detailed mode only). None in default "
            "mode. Each candidate's ``layer_trajectory`` shows how the "
            "move's rank evolves across BT4's 15 layers; "
            "``network_dismissal`` tags the qualitative shape "
            "(``rose_and_stayed`` / ``rose_and_fell`` / ``never_rose`` / "
            "``unknown``)."
        ),
    )

    # frozen=True: model is immutable post-construction (the v1 contract;
    # findings flow forward; nothing mutates them in-place).
    # extra="allow" (added in Chunk 7a.3 for extension-friendliness per
    # ADR-0013): Tier-2 / Chunk-8+ extractors can attach new findings
    # fields here without modifying this class. The strict per-extractor
    # schema lives at the EXTRACTOR-level Pydantic models (e.g.,
    # MCTSFindings, StockfishResult); this top-level wrapper is permissive.
    model_config = ConfigDict(frozen=True, extra="allow")


def handle_what_is_leela_seeing(
    board: chess.Board,
    *,
    model: "HookedTransformer",
    stockfish_engine: "chess.engine.SimpleEngine",
    lc0_engine: "Lc0Engine | None" = None,
    top_n: int = 5,
    stockfish_depth: int = 15,
    stockfish_top_n: int = 3,
    mcts_nodes: int = 10000,
    mode: Mode = "quick",
) -> WhatIsLeelaSeeingFindings:
    """Build the findings JSON for ``what_is_leela_seeing`` on the given position.

    Runs Leela's policy-network forward pass, Stockfish analysis, and
    (when an Lc0 engine is supplied) Leela MCTS search side by side, then
    assembles a combined findings model. Per CLAUDE.md §3, the handler does
    no chess interpretation — it just calls extractors and packs their output.

    Args:
        board: Position to assess.
        model: Loaded ``HookedTransformer`` BT4 from
            :func:`leela_chat.network.load_bt4`, already on the target device.
        stockfish_engine: Live ``chess.engine.SimpleEngine``. Caller-managed
            (spawn once at process start; close on exit).
        lc0_engine: Optional live :class:`Lc0Engine` for MCTS analysis. When
            ``None``, the handler skips MCTS entirely (``findings.mcts =
            None``) and the v4 prompt + templated fallback degrade
            gracefully. Used for legacy / test paths that don't want to
            spawn Lc0; production CLI always supplies one.
        top_n: How many Leela-policy candidate moves to surface to the LLM.
        stockfish_depth: Stockfish search depth. 15 is the default; raise for
            sharper positions or critical eval, lower for sub-second latency.
        stockfish_top_n: How many Stockfish candidate lines to include (multipv).
        mcts_nodes: Node budget for Lc0 MCTS. 10000 is a reasonable trade-off
            between depth and pod-warm latency (~1-3 s). Increase for the
            rare position where the user explicitly asks for deeper analysis.
        mode: Per-handler latency mode (per ADR-0012). ``"quick"`` uses
            :func:`extract_mcts` (snapshot) and skips layer_emergence — fits
            the <30s quick-mode gate. ``"deep"`` swaps in
            :func:`extract_mcts_trajectory` (free populates per-child
            ``trajectory_*`` fields from the live-stats stream) AND calls
            :func:`extract_layer_emergence` to populate ``layer_emergence``
            (~750 ms for N+1 forward passes on RTX 3090). The v7 prompt
            template (PROMPT_VERSION_DETAILED) reads the extra fields; the
            v6 prompt (PROMPT_VERSION) reads the quick-mode subset. Default
            ``"quick"`` preserves the byte-identical v4 contract for legacy
            and non-coach paths (this is what the old ``detailed=False`` did).

    Returns:
        Findings model for the translation layer.
    """
    # Per Chunk 7a.2 (extractor parallelisation, Chunk 6.7 entry criterion #1):
    # spawn one thread per ENGINE so the slowest engine sets the handler floor
    # instead of the sum.
    #
    # Chunk 7c.8 wire-up: extract_network_inference now runs FIRST (sequential
    # ~0.3s on the BT4 GPU) so its policy-distribution character can inform
    # the MCTS multipv choice (compute_adaptive_multipv per ADR-0015). After
    # that, the slow layer_emergence (~3-4s) runs in a thread alongside SF +
    # Lc0. Net latency vs the pre-7c.8 arrangement is ~neutral (old: bt4_thread
    # did nw + layer_em sequentially while SF/Lc0 ran in parallel; new:
    # 0.3s of nw before parallel block, then max(layer_em, SF, Lc0) running
    # in parallel = roughly the same total).
    leela = extract_network_inference(board, model=model, top_n=top_n)
    if mode == "deep" and lc0_engine is not None:
        # Compute adaptive multipv from the policy distribution character.
        # Per ADR-0015 + 7c.8 calibration: dominant top → 2, moderate → 5
        # (the 7b default), contested → 10 (caps to 8 in deep), flat → 15
        # (caps to 8 in deep). Quick mode caps to 3 — but quick uses
        # extract_mcts not extract_mcts_trajectory so adaptive doesn't
        # apply to quick (quick stays multipv=1 implicitly via extract_mcts).
        policy_pcts = [m.policy_pct for m in leela.top_moves]
        adaptive_mpv = compute_adaptive_multipv(
            top_prob=(leela.top_moves[0].policy_pct / 100.0) if leela.top_moves else 0.0,
            policy_entropy=policy_entropy_normalized(policy_pcts),
            mode=mode,
        )
    else:
        adaptive_mpv = 1  # quick mode / no lc0_engine

    def _bt4_layer_em_thread() -> LayerEmergenceResult | None:
        if mode == "deep":
            return extract_layer_emergence(
                board, model=model, focal_uci=None, top_n_per_layer=top_n,
            )
        return None

    with ThreadPoolExecutor(max_workers=3) as ex:
        bt4_layer_future = ex.submit(_bt4_layer_em_thread)
        sf_future = ex.submit(
            extract_stockfish_eval, board,
            engine=stockfish_engine,
            depth=stockfish_depth,
            top_n=stockfish_top_n,
        )
        if lc0_engine is not None:
            if mode == "deep":
                # Chunk 7c.8 adaptive multipv — was hardcoded multipv=5 in
                # 7b. See compute_adaptive_multipv above for the policy
                # character → multipv mapping (ADR-0015).
                mcts_future = ex.submit(
                    extract_mcts_trajectory, board,
                    engine=lc0_engine, max_nodes=mcts_nodes, multipv=adaptive_mpv,
                )
            else:
                mcts_future = ex.submit(
                    extract_mcts, board,
                    engine=lc0_engine, nodes=mcts_nodes,
                )

    layer_emergence = bt4_layer_future.result()
    sf = sf_future.result()
    mcts_findings: MCTSFindings | None = None
    if lc0_engine is not None:
        full_mcts = mcts_future.result()
        # Truncate children for the LLM-facing handler projection. The
        # extractor's full output (all root candidates incl. zero-visit
        # tail) is what `did_you_consider` (Handler 3, Chunk 6) will
        # consume; Handler 1 only needs the leading few by visit share.
        mcts_findings = full_mcts.model_copy(
            update={"children": list(full_mcts.children[:MCTS_TOP_N_FOR_LLM])},
        )

    return WhatIsLeelaSeeingFindings(
        fen=leela.fen,
        side_to_move=leela.side_to_move,
        eval_cp=leela.eval_cp,
        wdl=leela.wdl,
        moves_left_estimate=leela.moves_left_estimate,
        top_moves=leela.top_moves,
        stockfish=sf,
        mcts=mcts_findings,
        layer_emergence=layer_emergence,
    )


def templated_fallback(findings: WhatIsLeelaSeeingFindings) -> str:
    """Mechanical rendering of findings, used when LLM validation fails twice.

    Per CLAUDE.md §2 Rule 2: never show an unvalidated response to the user.
    When the LLM hallucinates on both attempts, this is the safe last resort —
    every number traces directly back to the findings model, so it is
    grounded by construction. Reads robotically; that is acceptable for a
    fallback path. v3 added Stockfish (ADR-0009); v4 adds MCTS — the
    MCTS section is rendered only when ``findings.mcts`` is present.
    """
    eval_pawns = findings.eval_cp / 100.0
    eval_sign = "+" if findings.eval_cp >= 0 else ""
    wdl = findings.wdl
    wdl_str = (
        f"{wdl.white_wins * 100:.0f}% White / "
        f"{wdl.draw * 100:.0f}% Draw / "
        f"{wdl.black_wins * 100:.0f}% Black"
    )
    if findings.top_moves:
        moves_str = ", ".join(
            f"{m.san} ({m.policy_pct:.1f}%)" for m in findings.top_moves[:5]
        )
        leela_moves_part = f" Leela's policy top moves: {moves_str}."
    else:
        leela_moves_part = " No legal moves (terminal position)."

    sf = findings.stockfish
    if sf.root_mate_in is not None:
        sf_eval_str = f"mate in {abs(sf.root_mate_in)} plies for {'White' if sf.root_mate_in > 0 else 'Black'}"
    elif sf.root_eval_cp is not None:
        sf_sign = "+" if sf.root_eval_cp >= 0 else ""
        sf_eval_str = f"{sf_sign}{sf.root_eval_cp} cp ({sf_sign}{sf.root_eval_cp / 100.0:.2f})"
    else:
        sf_eval_str = "unavailable"
    if sf.top_moves:
        sf_top_str = ", ".join(m.san for m in sf.top_moves[:3])
        sf_part = (
            f" Stockfish (depth {sf.depth}): root eval {sf_eval_str}; "
            f"top candidates {sf_top_str}."
        )
    else:
        sf_part = f" Stockfish (depth {sf.depth}): no candidates returned."

    mcts = findings.mcts
    if mcts is None or not mcts.children:
        mcts_part = ""
    else:
        top = mcts.children[0]
        total = mcts.total_nodes_searched or 1  # guard /0 on degenerate cases
        share_pct = 100.0 * top.visits / total
        top_str = ", ".join(
            f"{c.san} (N={c.visits}, Q={c.q:+.2f})" for c in mcts.children[:3]
        )
        mcts_part = (
            f" Leela MCTS (committed: {mcts.bestmove_san}, {top.visits} of "
            f"{total} visits = {share_pct:.0f}%): top by visits {top_str}."
        )

    return (
        f"Leela's policy network evaluates the position at "
        f"{eval_sign}{findings.eval_cp} cp ({eval_sign}{eval_pawns:.2f}). "
        f"Game-result probabilities: {wdl_str}. "
        f"Side to move: {findings.side_to_move}."
        f"{leela_moves_part}"
        f"{sf_part}"
        f"{mcts_part}"
    )
