"""Integration test for the what_is_leela_seeing translation pipeline.

Mocks the LLM (the only external dependency at this layer) and exercises the
full path: prompt loading -> LLMClient.translate -> validator -> retry / fallback.

Per CLAUDE.md §6: every handler has at least one integration test that mocks
the LLM. This test stands in for that requirement for ``what_is_leela_seeing``.

The handler itself (``handle_what_is_leela_seeing``) is intentionally NOT
exercised here — it's a thin shim over ``extract_network_inference``, which
requires loading BT4 (~1.3 GB, GPU). The handler is smoke-tested on the pod
via ``leela_chat/scripts/smoke_test_handler.py`` with real FENs.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from leela_chat.extractors.network_inference import MoveCandidate, WDL
from leela_chat.handlers.what_is_leela_seeing import (
    HANDLER_NAME,
    PROMPT_VERSION,
    WhatIsLeelaSeeingFindings,
    templated_fallback,
)
from leela_chat.llm.client import LLMResponse
from leela_chat.llm.translate import translate_and_validate


# A fixed findings object used across tests. Starting position, slight White
# edge, two top-policy moves. Keeps numbers small and ungrounded-failures easy
# to construct.
FIXED_FINDINGS = WhatIsLeelaSeeingFindings(
    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    side_to_move="white",
    eval_cp=12,
    wdl=WDL(white_wins=0.19, draw=0.64, black_wins=0.17),
    moves_left_estimate=86.4,
    top_moves=[
        MoveCandidate(uci="e2e4", san="e4", policy_pct=14.2),
        MoveCandidate(uci="d2d4", san="d4", policy_pct=12.8),
    ],
)


def _mock_llm_response(text: str) -> LLMResponse:
    """Build an LLMResponse with the given text and zero-token telemetry."""
    return LLMResponse(
        text=text,
        model="claude-test-mock",
        input_tokens=100,
        output_tokens=50,
        stop_reason="end_turn",
    )


def _api_error_response() -> LLMResponse:
    """Build the sentinel LLMResponse the client returns on transient API failure."""
    return LLMResponse(
        text="",
        model="claude-test-mock",
        input_tokens=0,
        output_tokens=0,
        stop_reason="api_error",
    )


def _mock_client(responses: list[str]) -> MagicMock:
    """Build a mocked LLMClient that returns the given responses in order."""
    client = MagicMock()
    client.translate = MagicMock(
        side_effect=[_mock_llm_response(t) for t in responses]
    )
    return client


def _mock_client_mixed(responses: list[LLMResponse]) -> MagicMock:
    """Build a mocked LLMClient returning pre-built LLMResponses in order.

    Use this when you need to mix normal text responses with api_error sentinels.
    """
    client = MagicMock()
    client.translate = MagicMock(side_effect=responses)
    return client


def test_first_attempt_valid_returns_llm_source() -> None:
    """Happy path: LLM returns a valid response on attempt 1."""
    # All numbers in this response trace to FIXED_FINDINGS.
    response = (
        "Leela evaluates the position at +12 cp with probabilities 19% White, "
        "64% Draw, 17% Black. The top moves are e4 (14.2%) and d4 (12.8%)."
    )
    client = _mock_client([response])

    result = translate_and_validate(
        findings=FIXED_FINDINGS,
        question="What is Leela seeing?",
        handler_name=HANDLER_NAME,
        templated_fallback=templated_fallback,
        prompt_version=PROMPT_VERSION,
        client=client,
    )

    assert result.source == "llm"
    assert result.validation.ok is True
    assert result.text == response
    assert len(result.llm_responses) == 1
    assert client.translate.call_count == 1


def test_first_invalid_retry_valid_returns_llm_retry_source() -> None:
    """LLM hallucinates on attempt 1, fixes itself on retry."""
    bad_response = (
        # Mentions Nf6, which is illegal in the starting position (knight on g8,
        # not c6/d7).  Also cites +50 which isn't in findings.
        "Leela sees a balanced position. The top reply is Nf6 (+50 cp)."
    )
    good_response = (
        "Leela evaluates the position at +12 cp; W/D/L is 19% / 64% / 17%. "
        "Leela prefers e4 (14.2%) over d4 (12.8%)."
    )
    client = _mock_client([bad_response, good_response])

    result = translate_and_validate(
        findings=FIXED_FINDINGS,
        question="What is Leela seeing?",
        handler_name=HANDLER_NAME,
        templated_fallback=templated_fallback,
        prompt_version=PROMPT_VERSION,
        client=client,
    )

    assert result.source == "llm_retry"
    assert result.validation.ok is True
    assert result.text == good_response
    assert len(result.llm_responses) == 2
    assert client.translate.call_count == 2

    # The second call must have received a stricter system prompt — verify by
    # inspecting kwargs. The stricter prompt contains "## Retry context".
    second_call_kwargs = client.translate.call_args_list[1].kwargs
    assert "## Retry context" in second_call_kwargs["system_prompt"]
    assert "illegal_move" in second_call_kwargs["system_prompt"]


def test_both_attempts_invalid_falls_back_to_templated() -> None:
    """LLM hallucinates twice; orchestrator drops to templated fallback."""
    bad1 = "Top move is Nf6 with eval +50."  # illegal move + ungrounded number
    bad2 = "Black's best is Qxh1 with eval -99.9."  # illegal move + ungrounded
    client = _mock_client([bad1, bad2])

    result = translate_and_validate(
        findings=FIXED_FINDINGS,
        question="What is Leela seeing?",
        handler_name=HANDLER_NAME,
        templated_fallback=templated_fallback,
        prompt_version=PROMPT_VERSION,
        client=client,
    )

    assert result.source == "templated"
    # validation field carries the LAST LLM-attempt validation, which failed.
    assert result.validation.ok is False
    # Templated text should be the mechanical render — contains the literal cp.
    assert "+12 cp" in result.text
    assert "19% White" in result.text
    assert "e4 (14.2%)" in result.text
    assert len(result.llm_responses) == 2
    assert client.translate.call_count == 2


def test_first_api_error_retry_valid_returns_llm_retry_source() -> None:
    """LLM API call fails on attempt 1, succeeds on retry. Retry uses base prompt."""
    good_response = (
        "Leela evaluates the position at +12 cp; W/D/L is 19% / 64% / 17%. "
        "Leela prefers e4 (14.2%) over d4 (12.8%)."
    )
    client = _mock_client_mixed([
        _api_error_response(),
        _mock_llm_response(good_response),
    ])

    result = translate_and_validate(
        findings=FIXED_FINDINGS,
        question="What is Leela seeing?",
        handler_name=HANDLER_NAME,
        templated_fallback=templated_fallback,
        prompt_version=PROMPT_VERSION,
        client=client,
    )

    assert result.source == "llm_retry"
    assert result.validation.ok is True
    assert result.text == good_response
    assert client.translate.call_count == 2

    # API error on attempt 1 means the retry has no content to critique;
    # it must use the BASE prompt unmodified, not the stricter retry prompt.
    first_call_kwargs = client.translate.call_args_list[0].kwargs
    second_call_kwargs = client.translate.call_args_list[1].kwargs
    assert "## Retry context" not in second_call_kwargs["system_prompt"]
    assert first_call_kwargs["system_prompt"] == second_call_kwargs["system_prompt"]


def test_both_attempts_api_error_falls_back_to_templated() -> None:
    """Both LLM calls fail with API error; orchestrator drops to templated."""
    client = _mock_client_mixed([
        _api_error_response(),
        _api_error_response(),
    ])

    result = translate_and_validate(
        findings=FIXED_FINDINGS,
        question="What is Leela seeing?",
        handler_name=HANDLER_NAME,
        templated_fallback=templated_fallback,
        prompt_version=PROMPT_VERSION,
        client=client,
    )

    assert result.source == "templated"
    assert result.validation.ok is False
    # The last-attempt validation carries the api_error sentinel kind.
    assert any(e.kind == "api_error" for e in result.validation.errors)
    # Templated text is the mechanical render — verify the key fields landed.
    assert "+12 cp" in result.text
    assert "19% White" in result.text
    assert "e4 (14.2%)" in result.text
    assert client.translate.call_count == 2


def test_content_failure_then_api_error_falls_back_to_templated() -> None:
    """Attempt 1 fails validation, attempt 2 API-errors; templated kicks in."""
    bad_response = "Top move is Nf6 with eval +50."  # illegal move + ungrounded
    client = _mock_client_mixed([
        _mock_llm_response(bad_response),
        _api_error_response(),
    ])

    result = translate_and_validate(
        findings=FIXED_FINDINGS,
        question="What is Leela seeing?",
        handler_name=HANDLER_NAME,
        templated_fallback=templated_fallback,
        prompt_version=PROMPT_VERSION,
        client=client,
    )

    assert result.source == "templated"
    assert result.validation.ok is False
    assert any(e.kind == "api_error" for e in result.validation.errors)
    # Attempt 1 was a content failure, so the retry SHOULD have used the
    # stricter prompt (with "## Retry context") — even though attempt 2 then
    # api-errored before producing anything.
    second_call_kwargs = client.translate.call_args_list[1].kwargs
    assert "## Retry context" in second_call_kwargs["system_prompt"]


def test_empty_response_treated_as_failure() -> None:
    """LLM returns empty text (e.g. refused); orchestrator retries."""
    empty_response = LLMResponse(
        text="",
        model="claude-test-mock",
        input_tokens=50,
        output_tokens=0,
        stop_reason="end_turn",
    )
    good_response = (
        "Leela sees +12 cp, with 19% White / 64% Draw / 17% Black. Top move e4 (14.2%)."
    )
    client = _mock_client_mixed([empty_response, _mock_llm_response(good_response)])

    result = translate_and_validate(
        findings=FIXED_FINDINGS,
        question="What is Leela seeing?",
        handler_name=HANDLER_NAME,
        templated_fallback=templated_fallback,
        prompt_version=PROMPT_VERSION,
        client=client,
    )

    assert result.source == "llm_retry"
    assert result.validation.ok is True
    # Empty response should retry with BASE prompt (nothing to critique).
    second_call_kwargs = client.translate.call_args_list[1].kwargs
    assert "## Retry context" not in second_call_kwargs["system_prompt"]


def test_unknown_handler_name_raises() -> None:
    """If the handler's prompt directory doesn't exist, fail loudly."""
    client = _mock_client(["dummy"])
    with pytest.raises(FileNotFoundError, match="Prompt not found"):
        translate_and_validate(
            findings=FIXED_FINDINGS,
            question="Q?",
            handler_name="does_not_exist",
            templated_fallback=templated_fallback,
            client=client,
        )


def test_unknown_prompt_version_raises() -> None:
    """If the prompt version doesn't exist, fail loudly."""
    client = _mock_client(["dummy"])
    with pytest.raises(FileNotFoundError, match="Prompt not found"):
        translate_and_validate(
            findings=FIXED_FINDINGS,
            question="Q?",
            handler_name=HANDLER_NAME,
            templated_fallback=templated_fallback,
            prompt_version=999,
            client=client,
        )
