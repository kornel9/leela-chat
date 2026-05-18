"""Translation orchestrator: findings -> LLM -> validator -> validated prose.

Per CLAUDE.md §3, this layer is the only place the LLM is called. It owns:

- Loading the per-handler prompt template (versioned files under
  ``leela_chat/llm/prompts/{handler}/v{N}.md``).
- Calling :class:`LLMClient` with the cached system prompt.
- Running :func:`validate_response` on the result.
- The retry / fallback chain mandated by Rule 2 (CLAUDE.md §2):

  1. Attempt 1: normal prompt.
  2. On validation failure: Attempt 2 with the validator's errors appended
     to the system prompt as a "retry context" block.
  3. On second failure: drop to the handler's ``templated_fallback`` —
     mechanical rendering of the findings, grounded by construction.

The orchestrator never raises on validation failure; it always returns a
:class:`ValidatedTranslation`. The ``source`` field tells you which path
produced the final text.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from leela_chat.llm.client import LLMClient, LLMMode, LLMResponse
from leela_chat.validator import ValidationResult, validate_response
from leela_chat.validator.types import ValidationError


_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _validate_attempt(
    resp: LLMResponse, *, fen: str, findings: dict
) -> ValidationResult:
    """Wrapper around ``validate_response`` that also handles non-content failures.

    Three failure modes the orchestrator's fallback chain must treat
    identically: validation failure, an empty text response, and an API-level
    error (overload, rate limit, network). The validator alone can't see the
    latter two — empty text passes legality and grounding trivially because
    there's nothing to check. This helper synthesizes the appropriate
    ``ValidationResult`` so the same retry/templated logic catches all three.
    """
    if resp.stop_reason == "api_error":
        return ValidationResult(
            ok=False,
            errors=[ValidationError(
                kind="api_error",
                detail=(
                    "LLM API call failed (overload, rate limit, or transient "
                    "network error). See client.py warning log for the exact "
                    "exception type."
                ),
                span="",
            )],
        )
    if not resp.text:
        return ValidationResult(
            ok=False,
            errors=[ValidationError(
                kind="empty_response",
                detail="LLM returned an empty response.",
                span="",
            )],
        )
    return validate_response(resp.text, fen=fen, findings=findings)


class ValidatedTranslation(BaseModel):
    """Final result of the translation pipeline."""

    text: str = Field(..., description="User-facing prose. Always validated or templated.")
    source: Literal["llm", "llm_retry", "templated"] = Field(
        ...,
        description=(
            "Which path produced ``text``: ``llm`` = first attempt validated; "
            "``llm_retry`` = second attempt validated; ``templated`` = mechanical "
            "fallback after both LLM attempts failed validation."
        ),
    )
    validation: ValidationResult = Field(
        ...,
        description=(
            "Validation result for ``text``. For ``source=llm``/``llm_retry`` this "
            "passed (``ok=True``). For ``source=templated`` this is the *last* "
            "LLM-attempt validation result (always ``ok=False``); the templated "
            "text itself is grounded by construction."
        ),
    )
    llm_responses: list[LLMResponse] = Field(
        default_factory=list,
        description="All LLM calls made (1 or 2). Empty list never happens in practice.",
    )
    handler_name: str
    prompt_version: int

    model_config = ConfigDict(frozen=True)


def translate_and_validate(
    findings: BaseModel,
    question: str,
    *,
    handler_name: str,
    templated_fallback: Callable[[Any], str],
    prompt_version: int = 1,
    mode: LLMMode = LLMMode.PROD,
    client: LLMClient | None = None,
) -> ValidatedTranslation:
    """Run the translate-and-validate pipeline for one handler call.

    Args:
        findings: Pydantic findings model from the handler. Must expose a
            ``fen: str`` attribute and serialize cleanly via ``model_dump()``.
            This contract is convention-enforced until handler #2 lands.
        question: User's literal question.
        handler_name: Which handler's prompts directory to read from
            (``leela_chat/llm/prompts/{handler_name}/v{prompt_version}.md``).
        templated_fallback: Handler-supplied function that renders ``findings``
            as mechanical prose. Called only if both LLM attempts fail
            validation. Receives the same ``findings`` model.
        prompt_version: Which version of the handler's prompt to load.
        mode: ``LLMMode.PROD`` (Opus 4.7, eval/prod) or ``LLMMode.ITER``
            (Sonnet 4.6, dev iteration). Per ADR-0005.
        client: Pre-built :class:`LLMClient`. If ``None``, one is created
            for this call.

    Returns:
        :class:`ValidatedTranslation` with the final user-facing text and
        provenance metadata.
    """
    if client is None:
        client = LLMClient(mode=mode)

    system_prompt = _load_prompt(handler_name, prompt_version)
    findings_dict = findings.model_dump()
    findings_json = findings.model_dump_json(indent=2)
    fen = findings.fen  # contract: every findings model has a .fen

    # --- Attempt 1: normal prompt ---
    resp1 = client.translate(
        system_prompt=system_prompt,
        findings_json=findings_json,
        question=question,
    )
    val1 = _validate_attempt(resp1, fen=fen, findings=findings_dict)
    if val1.ok:
        return ValidatedTranslation(
            text=resp1.text,
            source="llm",
            validation=val1,
            llm_responses=[resp1],
            handler_name=handler_name,
            prompt_version=prompt_version,
        )

    logger.warning(
        f"translate: attempt 1 failed ({_failure_summary(val1)}); retrying"
    )

    # --- Attempt 2: retry ---
    # On content failures (illegal_move / ungrounded_eval / ...), inject the
    # validator's errors into a "Retry context" addendum so the LLM can
    # self-correct. On API/empty failures there's nothing to tell the LLM
    # about its previous output — retry with the base prompt unchanged so
    # we don't pollute the cache key with addenda that don't apply.
    if _is_call_failure(val1):
        retry_prompt = system_prompt
    else:
        retry_prompt = _stricter_retry_prompt(system_prompt, val1)
    resp2 = client.translate(
        system_prompt=retry_prompt,
        findings_json=findings_json,
        question=question,
    )
    val2 = _validate_attempt(resp2, fen=fen, findings=findings_dict)
    if val2.ok:
        return ValidatedTranslation(
            text=resp2.text,
            source="llm_retry",
            validation=val2,
            llm_responses=[resp1, resp2],
            handler_name=handler_name,
            prompt_version=prompt_version,
        )

    logger.warning(
        f"translate: attempt 2 also failed ({_failure_summary(val2)}); "
        f"falling back to templated response"
    )

    # --- Fallback: templated ---
    templated_text = templated_fallback(findings)
    return ValidatedTranslation(
        text=templated_text,
        source="templated",
        validation=val2,  # last LLM-attempt validation result
        llm_responses=[resp1, resp2],
        handler_name=handler_name,
        prompt_version=prompt_version,
    )


def _is_call_failure(val: ValidationResult) -> bool:
    """True if any error in ``val`` indicates a call-level failure.

    "Call-level" = api_error or empty_response — failure modes where retrying
    with a stricter content-correction prompt makes no sense because the
    previous attempt produced no usable content to critique.
    """
    return any(e.kind in ("api_error", "empty_response") for e in val.errors)


def _failure_summary(val: ValidationResult) -> str:
    """Short telemetry-friendly string describing ``val.errors``."""
    kinds = sorted({e.kind for e in val.errors})
    return f"{len(val.errors)} error(s), kinds={kinds}"


def _load_prompt(handler_name: str, version: int) -> str:
    path = _PROMPTS_DIR / handler_name / f"v{version}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt not found for handler '{handler_name}' v{version}: {path}"
        )
    return path.read_text()


def _stricter_retry_prompt(base_prompt: str, prior_validation: ValidationResult) -> str:
    """Append a retry-context block to the system prompt.

    The validator's errors are listed verbatim so the LLM can self-correct.
    We deliberately do NOT rewrite the rules — we point at the specific
    failures and ask for a fresh attempt mentioning only items in the JSON.
    """
    error_lines = "\n".join(
        f"- {e.kind}: {e.detail}" for e in prior_validation.errors
    )
    addendum = (
        "\n\n## Retry context\n\n"
        "Your previous attempt at this position failed automated validation. "
        "Specific issues:\n"
        f"{error_lines}\n\n"
        "Generate a new response from scratch. Mention only moves, evaluations, "
        "and probabilities that appear in the Findings JSON. Do not reuse phrasing "
        "from your previous attempt — start over."
    )
    return base_prompt + addendum
