"""Anthropic LLM client for the translation layer.

Wraps the official Anthropic SDK with:

- **Mode flag** (``iter`` / ``prod``) per ADR-0005 G1.2 — Sonnet 4.6 for
  prompt-design iteration, Opus 4.7 for eval runs and live use.
- **Prompt caching** (``cache_control: ephemeral``) on the stable system
  block per ADR-0005 G1.4. Note: cache only engages above the model's
  minimum cacheable prefix (Opus 4.7: 4096 tokens, Sonnet 4.6: 2048).
  The v1 prompt is ~1000 tokens — too small to cache today. The wiring is
  forward-looking; caching kicks in as prompts grow (more handlers,
  few-shot examples, JSON schemas).
- **Telemetry**: every call returns input/output token counts plus the
  cache-write and cache-read totals so we can verify hit rate over time.

Per CLAUDE.md §3, the translation layer is the only place the LLM is
called. It receives findings JSON from the handler plus the user's
question, returns prose. Validation is downstream
(``leela_chat/validator/__init__.py``).
"""
from __future__ import annotations

import enum

import anthropic
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field


class LLMMode(str, enum.Enum):
    """Per ADR-0005 G1.2 — which model tier to target.

    ``ITER`` is the cheap/fast Sonnet path for prompt-design loops; ``PROD``
    is the Opus path for eval runs and live user-facing inference.
    """

    ITER = "iter"
    PROD = "prod"


# Exact model strings per ADR-0005 (and the claude-api skill: use the
# published alias, never append date suffixes).
MODE_MODEL: dict[LLMMode, str] = {
    LLMMode.ITER: "claude-sonnet-4-6",
    LLMMode.PROD: "claude-opus-4-7",
}


class LLMResponse(BaseModel):
    """Wrapped LLM response with usage telemetry.

    All token counts are integers. ``cache_creation_input_tokens`` and
    ``cache_read_input_tokens`` are zero when caching didn't engage (e.g.
    the cached prefix was smaller than the model's minimum, or the first
    call of a 5-minute TTL window).
    """

    text: str = Field(..., description="The prose response from the LLM.")
    model: str = Field(..., description="The exact model ID that served the request.")
    input_tokens: int = Field(..., ge=0, description="Uncached input tokens billed at full rate.")
    output_tokens: int = Field(..., ge=0)
    cache_creation_input_tokens: int = Field(
        0, ge=0, description="Tokens written to the cache this call (~1.25x rate).",
    )
    cache_read_input_tokens: int = Field(
        0, ge=0, description="Tokens served from the cache this call (~0.1x rate).",
    )
    stop_reason: str | None = None

    model_config = ConfigDict(frozen=True)

    @property
    def cache_hit(self) -> bool:
        """True iff at least one token was served from cache on this call."""
        return self.cache_read_input_tokens > 0


class LLMClient:
    """Translation-layer client wrapping ``anthropic.Anthropic()``."""

    def __init__(self, *, mode: LLMMode = LLMMode.PROD, max_tokens: int = 1024) -> None:
        """Initialise.

        Args:
            mode: Model tier per ADR-0005. Default ``PROD`` (Opus 4.7) —
                callers explicitly opt into ``ITER`` for prompt iteration.
            max_tokens: Response cap. Translation outputs are typically 2-4
                sentences (~200-500 tokens). 1024 leaves comfortable headroom
                without inviting runaway generation.
        """
        # Reads ANTHROPIC_API_KEY from os.environ; leela_chat.llm.env.load_env
        # must have been called first if the key lives in .env.
        self._client = anthropic.Anthropic()
        self.mode = mode
        self.model = MODE_MODEL[mode]
        self.max_tokens = max_tokens
        logger.info(f"LLMClient ready: mode={mode.value} model={self.model}")

    def translate(
        self,
        *,
        system_prompt: str,
        findings_json: str,
        question: str,
    ) -> LLMResponse:
        """Generate prose translation of findings JSON.

        Args:
            system_prompt: Handler prompt template content. Cached (ephemeral)
                because it's stable across calls for the same handler version.
            findings_json: Handler findings as a JSON string. Varies per
                position, so it goes in the user turn (uncached).
            question: User's literal question. Varies per call.

        Returns:
            LLMResponse with prose text, model used, and cache telemetry.
        """
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{
                "type": "text",
                "text": system_prompt,
                # Stable across calls for the same handler version → cache it.
                # Below the model minimum (Opus 4.7: 4096, Sonnet 4.6: 2048)
                # this silently no-ops. That's fine — the wiring is forward-
                # looking; caching engages naturally as prompts grow.
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Findings JSON:\n```json\n{findings_json}\n```"
                ),
            }],
        )

        # First text block is the response prose. There should be exactly
        # one for a non-thinking, non-tool-use call.
        text = next((b.text for b in response.content if b.type == "text"), "")

        usage = response.usage
        result = LLMResponse(
            text=text,
            model=response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            stop_reason=response.stop_reason,
        )

        logger.info(
            f"LLM call: model={result.model} "
            f"in={result.input_tokens} "
            f"cache_write={result.cache_creation_input_tokens} "
            f"cache_read={result.cache_read_input_tokens} "
            f"out={result.output_tokens} "
            f"stop={result.stop_reason}"
        )
        return result
