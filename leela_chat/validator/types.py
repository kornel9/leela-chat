"""Validator types — Pydantic models for validation results.

Per CLAUDE.md §2 Rule 2: every LLM response is validated before reaching the
user. The validator (implemented in Phase 1 Chunk 3) checks:

1. **Legality**: every move-like string in the response is legal on the
   position (verified via python-chess).
2. **JSON grounding**: every eval number in the response appears in the
   findings JSON within tolerance.
3. **Feature grounding** (Phase 4+): every feature label mentioned is in
   the firing list.

This module just defines the result types. Implementation is in
``leela_chat/validator/__init__.py`` (function signatures) and Chunk 3
(actual checking logic).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ErrorKind = Literal[
    "illegal_move",         # SAN/UCI in response is not legal on the position
    "ungrounded_eval",      # eval number not present in findings JSON
    "unknown_feature",      # feature label not in firings list (Phase 4+)
    "missing_findings",     # response references a field absent from findings JSON
    "format_violation",     # response violates structural rules (e.g., lists when prose required)
]


class ValidationError(BaseModel):
    """One validation failure within a response."""

    kind: ErrorKind = Field(..., description="Category of validation failure.")
    detail: str = Field(..., description="Human-readable description of the violation.")
    span: str = Field(
        ...,
        description="The substring of the LLM response that triggered the failure.",
    )

    model_config = ConfigDict(frozen=True)


class ValidationResult(BaseModel):
    """Outcome of validating one LLM response.

    ``ok=True`` means the response passed all checks and can be shown to the
    user. ``ok=False`` means at least one error was found; the conversation
    manager should retry with a stricter prompt or fall back to a templated
    response built from the findings JSON directly (CLAUDE.md §2 Rule 2).
    """

    ok: bool = Field(..., description="True iff zero errors.")
    errors: list[ValidationError] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)
