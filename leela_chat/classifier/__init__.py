"""Question-to-handler classifier.

Phase 1 scope: trivial routing. Only one production handler
(``what_is_leela_seeing``) exists, so everything maps to it. Signature is
final — Phase 2+ will replace the body with real routing once handlers 2
and 3 ship (probably keyword matching first, then a small LLM call for
ambiguous cases).
"""
from __future__ import annotations


def classify(question: str, current_handlers: list[str] | None = None) -> str:
    """Pick which handler should answer ``question``.

    Args:
        question: User's literal natural-language question. Currently unused
            (Phase 1 routes everything to a single handler); the parameter
            stays in the signature so callers don't change when real routing
            lands.
        current_handlers: Optional list of handler names available in this
            build. If supplied and ``what_is_leela_seeing`` isn't in it,
            raises — defends against silently degrading. ``None`` (default)
            skips the check.

    Returns:
        Handler name. Phase 1: always ``"what_is_leela_seeing"``.
    """
    handler = "what_is_leela_seeing"
    if current_handlers is not None and handler not in current_handlers:
        raise ValueError(
            f"Classifier wants to route to {handler!r}, but it is not in "
            f"current_handlers={current_handlers!r}."
        )
    return handler
