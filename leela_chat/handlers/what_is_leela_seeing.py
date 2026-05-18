"""Handler: what_is_leela_seeing.

Orchestrates the network-inference extractor and assembles the findings JSON
that the translation layer will pass to the LLM.

Per CLAUDE.md §3 architecture invariants:
- Handlers orchestrate extractors and assemble JSON for the LLM.
- Handlers do not contain chess logic. (The extractor does.)
- Handlers do not call the LLM directly. (The translation layer does.)
  They return the findings JSON; the translation layer calls the LLM.
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

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer

# Points at leela_chat/llm/prompts/what_is_leela_seeing/v2.md.
# Per CLAUDE.md §3: increment, never delete old prompt versions.
# v1 -> v2 (2026-05-18): tightened the precision rule for percentages
#   after smoke testing found Sonnet rounding `1.2%` to `1%`, which the
#   validator (correctly) flagged as ungrounded. v2 explicitly requires
#   one decimal place for `policy_pct` and bans threshold phrases like
#   `under 1%`.
PROMPT_VERSION = 2

HANDLER_NAME = "what_is_leela_seeing"


class WhatIsLeelaSeeingFindings(BaseModel):
    """Findings JSON for ``what_is_leela_seeing``, passed to the translation layer.

    Rule 1 (CLAUDE.md §2): the LLM is allowed to mention exactly the contents
    of this object — nothing more. The validator (Rule 2) checks the LLM's
    output against these fields and the underlying position.
    """

    fen: str = Field(..., description="The position, in FEN.")
    side_to_move: Literal["white", "black"]
    eval_cp: int = Field(
        ..., description="Centipawn-equivalent evaluation. Positive = White better.",
    )
    wdl: WDL
    moves_left_estimate: float = Field(..., ge=0.0)
    top_moves: list[MoveCandidate] = Field(
        ..., description="Top-N legal moves by policy weight, sorted descending.",
    )

    model_config = ConfigDict(frozen=True)


def handle_what_is_leela_seeing(
    board: chess.Board,
    *,
    model: "HookedTransformer",
    top_n: int = 5,
) -> WhatIsLeelaSeeingFindings:
    """Build the findings JSON for ``what_is_leela_seeing`` on the given position.

    Args:
        board: Position to assess.
        model: Loaded ``HookedTransformer`` BT4 from
            :func:`leela_chat.network.load_bt4`. Already on the target device.
        top_n: How many candidate moves to surface to the LLM.

    Returns:
        Findings model for the translation layer.
    """
    result = extract_network_inference(board, model=model, top_n=top_n)
    return WhatIsLeelaSeeingFindings(
        fen=result.fen,
        side_to_move=result.side_to_move,
        eval_cp=result.eval_cp,
        wdl=result.wdl,
        moves_left_estimate=result.moves_left_estimate,
        top_moves=result.top_moves,
    )


def templated_fallback(findings: WhatIsLeelaSeeingFindings) -> str:
    """Mechanical rendering of findings, used when LLM validation fails twice.

    Per CLAUDE.md §2 Rule 2: never show an unvalidated response to the user.
    When the LLM hallucinates on both attempts, this is the safe last resort —
    every number traces directly back to the findings model, so it is
    grounded by construction. Reads robotically; that is acceptable for a
    fallback path.
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
        moves_part = f" Top candidate moves: {moves_str}."
    else:
        moves_part = " No legal moves (terminal position)."
    return (
        f"Leela evaluates the position at {eval_sign}{findings.eval_cp} cp "
        f"({eval_sign}{eval_pawns:.2f}). Game-result probabilities: {wdl_str}. "
        f"Side to move: {findings.side_to_move}.{moves_part}"
    )
