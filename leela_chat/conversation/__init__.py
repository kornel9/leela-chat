"""In-memory conversation state for the REPL.

Phase 1 scope: record-keeping only. The REPL appends one ``Turn`` per
question/response exchange and offers a ``:history`` command to display them.
The LLM translation layer does NOT yet consume conversation context — the v3
``what_is_leela_seeing`` prompt is per-position and stateless. Multi-turn
context feeding lands when a handler with conversational follow-up makes that
useful (likely a future Phase 1+ revision once handlers 2/3 ship after Phase
2).

These models are deliberately NOT frozen — ``ConversationState.turns`` and
``current_fen`` mutate across a REPL session.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Turn(BaseModel):
    """One Q&A exchange in a REPL session."""

    fen: str = Field(..., description="The position the question was asked about.")
    question: str = Field(..., description="The user's literal question.")
    response: str = Field(..., description="The validated final response shown to the user.")
    source: Literal["llm", "llm_retry", "templated"] = Field(
        ...,
        description=(
            "Which path produced the response, mirroring "
            "``ValidatedTranslation.source``. Useful for `:history` display."
        ),
    )


class ConversationState(BaseModel):
    """Session-scoped state owned by the REPL loop.

    Lifecycle: created once per REPL invocation; mutated by the loop;
    discarded at exit. Not persisted across sessions.
    """

    current_fen: str | None = Field(
        default=None,
        description=(
            "The position currently in focus. ``None`` before the first valid "
            "FEN is supplied; set via the initial prompt or `:fen <FEN>`."
        ),
    )
    turns: list[Turn] = Field(
        default_factory=list,
        description="Past Q&A turns in chronological order. Used by `:history`.",
    )
