"""Unit tests for the smoke-battery aggregator (Chunk 6.8.1 hardening).

Specifically tests the three-rate classification + top-api-error-kinds
banner + circuit-breaker sentinel detection added in Chunk 6.8.1.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from leela_chat.scripts.aggregate_smoke_battery import (
    _classify_api_error_subkind,
    _detect_circuit_breaker_sentinel,
    _render,
)


# === _classify_api_error_subkind ==========================================


def test_classify_credit_exhausted():
    detail = "LLM API call failed (BadRequestError): Your credit balance is too low to access the Anthropic API."
    assert _classify_api_error_subkind(detail) == "credit_exhausted"


def test_classify_rate_limit():
    detail = "LLM API error (RateLimitError): Error code: 429 - {...}"
    assert _classify_api_error_subkind(detail) == "rate_limit"


def test_classify_overloaded():
    detail = "LLM API error (OverloadedError): Error code: 529 - {'type': 'error', 'error': {'type': 'overloaded_error'}}"
    assert _classify_api_error_subkind(detail) == "overloaded"


def test_classify_timeout():
    detail = "Connection timed out after 60s"
    assert _classify_api_error_subkind(detail) == "timeout"


def test_classify_other():
    detail = "Some unknown error condition"
    assert _classify_api_error_subkind(detail) == "other"


def test_classify_empty():
    assert _classify_api_error_subkind("") == "other"
    assert _classify_api_error_subkind(None) == "other"


# === _detect_circuit_breaker_sentinel =====================================


def test_sentinel_detection_present():
    rows = [
        {"position_id": "p1", "source": "templated"},
        {"_circuit_breaker": True, "consecutive_api_errors": 3, "last_error_kind": "api_error", "last_error_detail": "credit too low"},
    ]
    sentinel = _detect_circuit_breaker_sentinel(rows)
    assert sentinel is not None
    assert sentinel["consecutive_api_errors"] == 3


def test_sentinel_detection_absent():
    rows = [{"position_id": "p1", "source": "llm", "validation_ok": True}]
    assert _detect_circuit_breaker_sentinel(rows) is None


# === _render: three-rate computation ======================================


def _row(
    *, source: str = "llm", validation_ok: bool = True,
    llm_calls: int = 1, errors: list[dict] | None = None,
    category: str = "tactical_puzzle",
    handler: str = "what_is_leela_seeing", mode: str = "default",
    position_id: str = "p_test", themes: list[str] | None = None,
    rating: int = 1900, prose: str = "Some prose.",
    handler_s: float = 5.0, translate_s: float = 10.0,
    focal_uci: str | None = None, mcts_bestmove_uci: str | None = None,
) -> dict:
    return {
        "position_id": position_id,
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "category": category,
        "stratum": category,
        "themes": themes or ["fork"],
        "rating": rating,
        "handler": handler,
        "mode": mode,
        "focal_uci": focal_uci,
        "focal_source": "puzzle_solution",
        "mcts_bestmove_uci": mcts_bestmove_uci,
        "focal_matches_mcts_pick": (
            focal_uci is not None and focal_uci == mcts_bestmove_uci
        ),
        "source": source,
        "validation_ok": validation_ok,
        "validation_errors": errors or [],
        "prompt_version": 4,
        "handler_s": handler_s,
        "translate_s": translate_s,
        "llm_calls": llm_calls,
        "input_tokens": 1000 if llm_calls > 0 else 0,
        "output_tokens": 200 if llm_calls > 0 else 0,
        "prose": prose,
    }


def test_render_clean_run_no_banner():
    """All llm_success → no LLM-health warning banner."""
    rows = [_row(source="llm", validation_ok=True) for _ in range(20)]
    md = _render(rows, Path("test.jsonl"))
    assert "LLM-success rate: 100.0%" in md or "LLM-success rate:** 100.0%" in md
    # Banner only appears when below 95%; here it's 100%, no banner.
    assert "⚠️ LLM-success rate" not in md


def test_render_credit_empty_run_banner_fires():
    """Every call api_error → llm_success_rate=0% → banner with credit_exhausted."""
    err = {"kind": "api_error", "detail": "credit balance is too low", "span": ""}
    rows = [_row(source="templated", validation_ok=False, llm_calls=2, errors=[err]) for _ in range(20)]
    md = _render(rows, Path("test.jsonl"))
    # Banner present
    assert "⚠️ LLM-success rate" in md
    assert "LLM-success rate is 0.0%" in md
    # Top api_error sub-kind surfaced
    assert "credit_exhausted" in md
    # Validation-pass numbers flagged as unreliable
    assert "unreliable" in md


def test_render_mixed_failure_kinds_surfaces_top_3():
    """Run with mixed api_error kinds → top 3 in banner."""
    err_credit = {"kind": "api_error", "detail": "credit balance is too low", "span": ""}
    err_rate = {"kind": "api_error", "detail": "Error code: 429 rate limit", "span": ""}
    err_overload = {"kind": "api_error", "detail": "Error code: 529 overloaded", "span": ""}
    rows = (
        [_row(source="templated", validation_ok=False, llm_calls=2, errors=[err_credit])] * 5
        + [_row(source="templated", validation_ok=False, llm_calls=2, errors=[err_rate])] * 3
        + [_row(source="templated", validation_ok=False, llm_calls=2, errors=[err_overload])] * 2
    )
    md = _render(rows, Path("test.jsonl"))
    assert "credit_exhausted" in md
    assert "rate_limit" in md
    assert "overloaded" in md
    # Counts should reflect the per-kind counts.
    assert "(5)" in md
    assert "(3)" in md
    assert "(2)" in md


def test_render_circuit_breaker_sentinel_banner():
    """If a sentinel row is in the JSONL, the summary surfaces it
    prominently before the headline aggregate."""
    sentinel = {
        "_circuit_breaker": True,
        "consecutive_api_errors": 3,
        "last_error_kind": "api_error",
        "last_error_detail": "credit balance is too low to access the API",
        "ts": "2026-05-21T18:00:00Z",
    }
    rows = (
        [_row(source="templated", validation_ok=False, llm_calls=2,
              errors=[{"kind": "api_error", "detail": "credit too low", "span": ""}])
         for _ in range(3)]
        + [sentinel]
    )
    md = _render(rows, Path("test.jsonl"))
    assert "CIRCUIT BREAKER ENGAGED" in md
    assert "3 consecutive" in md
    # Sentinel row itself excluded from per-rate computation.
    assert "Runs:** 3" in md  # NOT 4 (sentinel excluded)


def test_render_validation_pass_among_llm_success_is_separated():
    """When some llm calls succeed and some templated, val_pass_rate
    should be computed ONLY among the llm-success calls."""
    rows = (
        [_row(source="llm", validation_ok=True) for _ in range(15)]
        + [_row(source="llm", validation_ok=False) for _ in range(5)]  # llm but bad prose
        + [_row(source="templated", validation_ok=False, llm_calls=2,
                errors=[{"kind": "api_error", "detail": "rate", "span": ""}])] * 5
    )
    md = _render(rows, Path("test.jsonl"))
    # llm_success: 20 (15+5), templated: 5 → llm_success_rate = 20/25 = 80%
    assert "LLM-success rate:** 80.0%" in md
    # Validation-pass AMONG llm-success: 15/20 = 75%
    assert "Validation-pass among LLM-success:** 75.0%" in md
    # Banner fires because llm_success < 95%
    assert "⚠️ LLM-success rate" in md
