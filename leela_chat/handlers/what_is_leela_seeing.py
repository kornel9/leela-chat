"""Handler: what_is_leela_seeing.

Orchestrates the network-inference extractor and assembles the findings JSON
that the translation layer will pass to the LLM.

Per CLAUDE.md §3 architecture invariants:
- Handlers orchestrate extractors and assemble JSON for the LLM.
- Handlers do not contain chess logic. (The extractor does.)
- Handlers do not call the LLM directly. (The translation layer does.)
  They return the findings JSON; the translation layer calls the LLM.

Implementation is deferred to Phase 1 Chunk 5.
"""
from __future__ import annotations

from typing import Literal

import chess
from pydantic import BaseModel, ConfigDict, Field

from leela_chat.extractors.network_inference import MoveCandidate, WDL

# Points at leela_chat/llm/prompts/what_is_leela_seeing/v1.md.
# Per CLAUDE.md §3: increment, never delete old prompt versions.
PROMPT_VERSION = 1


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
    model: object,  # LczeroModel — passed in by the conversation manager
    top_n: int = 5,
) -> WhatIsLeelaSeeingFindings:
    """Build the findings JSON for ``what_is_leela_seeing`` on the given position.

    Args:
        board: Position to assess.
        model: Loaded ``lczerolens.LczeroModel`` on the target device.
        top_n: How many candidate moves to surface to the LLM.

    Returns:
        Findings JSON for the translation layer to feed to the LLM.

    Raises:
        NotImplementedError: implementation is in Phase 1 Chunk 5.
    """
    raise NotImplementedError("Implemented in Phase 1 Chunk 5 (first handler).")
