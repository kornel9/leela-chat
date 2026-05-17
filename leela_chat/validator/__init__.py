"""Validator: gate-keeps every LLM response before it reaches the user.

Per CLAUDE.md §2 Rule 2:

> The output validator runs on every LLM response. It extracts every move-like
> string and verifies legality via python-chess. It extracts every eval number
> and checks it appears in the JSON within tolerance. It extracts every feature
> label and checks it's in the firing list. If validation fails:
> 1. Retry with stricter prompt
> 2. If retry fails, fall back to templated response built from the JSON directly
> 3. Log the failure
> 4. Never show an unvalidated response to the user

Per ADR-0005 G1.3: this validator is **strict mechanical only** — legality and
JSON-grounding. Semantic plausibility checks ("does this argument hold?") live
in the eval grader's LLM-as-judge, not on the live path.

Implementation is deferred to Phase 1 Chunk 3. This module defines the
function signature and re-exports the result types.
"""
from __future__ import annotations

from typing import Any

from leela_chat.validator.types import ErrorKind, ValidationError, ValidationResult

__all__ = [
    "ErrorKind",
    "ValidationError",
    "ValidationResult",
    "validate_response",
]


def validate_response(
    response_text: str,
    *,
    fen: str,
    findings: dict[str, Any],
) -> ValidationResult:
    """Validate an LLM response for legality + JSON-grounding.

    Args:
        response_text: The raw prose from the LLM.
        fen: The position the LLM was answering about. Used to legality-check
            any SAN/UCI moves mentioned.
        findings: The findings JSON the handler passed to the translation
            layer (i.e. the dict serialization of the handler's Pydantic
            findings model). Used to ground-check any numbers in the response.

    Returns:
        Validation result with ``ok`` and (if not ok) a list of errors.

    Raises:
        NotImplementedError: implementation is in Phase 1 Chunk 3.
    """
    raise NotImplementedError("Implemented in Phase 1 Chunk 3.")
