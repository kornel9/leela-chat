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

from typing import TYPE_CHECKING, Literal

import chess
from pydantic import BaseModel, ConfigDict, Field

from leela_chat.extractors.network_inference import (
    MoveCandidate,
    WDL,
    extract_network_inference,
)
from leela_chat.extractors.stockfish import (
    StockfishResult,
    extract_stockfish_eval,
)

if TYPE_CHECKING:
    import chess.engine
    from transformer_lens import HookedTransformer

# Points at leela_chat/llm/prompts/what_is_leela_seeing/v3.md.
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
PROMPT_VERSION = 3

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

    model_config = ConfigDict(frozen=True)


def handle_what_is_leela_seeing(
    board: chess.Board,
    *,
    model: "HookedTransformer",
    stockfish_engine: "chess.engine.SimpleEngine",
    top_n: int = 5,
    stockfish_depth: int = 15,
    stockfish_top_n: int = 3,
) -> WhatIsLeelaSeeingFindings:
    """Build the findings JSON for ``what_is_leela_seeing`` on the given position.

    Runs Leela's policy-network forward pass and Stockfish analysis side by
    side, then assembles a combined findings model. Per CLAUDE.md §3, the
    handler does no chess interpretation — it just calls extractors and packs
    their output.

    Args:
        board: Position to assess.
        model: Loaded ``HookedTransformer`` BT4 from
            :func:`leela_chat.network.load_bt4`, already on the target device.
        stockfish_engine: Live ``chess.engine.SimpleEngine``. Caller-managed
            (spawn once at process start; close on exit).
        top_n: How many Leela-policy candidate moves to surface to the LLM.
        stockfish_depth: Stockfish search depth. 15 is the default; raise for
            sharper positions or critical eval, lower for sub-second latency.
        stockfish_top_n: How many Stockfish candidate lines to include (multipv).

    Returns:
        Findings model for the translation layer.
    """
    leela = extract_network_inference(board, model=model, top_n=top_n)
    sf = extract_stockfish_eval(
        board,
        engine=stockfish_engine,
        depth=stockfish_depth,
        top_n=stockfish_top_n,
    )
    return WhatIsLeelaSeeingFindings(
        fen=leela.fen,
        side_to_move=leela.side_to_move,
        eval_cp=leela.eval_cp,
        wdl=leela.wdl,
        moves_left_estimate=leela.moves_left_estimate,
        top_moves=leela.top_moves,
        stockfish=sf,
    )


def templated_fallback(findings: WhatIsLeelaSeeingFindings) -> str:
    """Mechanical rendering of findings, used when LLM validation fails twice.

    Per CLAUDE.md §2 Rule 2: never show an unvalidated response to the user.
    When the LLM hallucinates on both attempts, this is the safe last resort —
    every number traces directly back to the findings model, so it is
    grounded by construction. Reads robotically; that is acceptable for a
    fallback path. Now includes Stockfish since v3 (ADR-0009).
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

    return (
        f"Leela's policy network evaluates the position at "
        f"{eval_sign}{findings.eval_cp} cp ({eval_sign}{eval_pawns:.2f}). "
        f"Game-result probabilities: {wdl_str}. "
        f"Side to move: {findings.side_to_move}."
        f"{leela_moves_part}"
        f"{sf_part}"
    )
