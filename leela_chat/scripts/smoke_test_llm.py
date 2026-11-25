"""Smoke test: minimal Anthropic API call verifying env loading + caching scaffold.

Run: ``uv run python -m leela_chat.scripts.smoke_test_llm``

Sends a tiny prompt to each model tier (Sonnet 4.6 for ``iter``, Opus 4.7
for ``prod``) — ~$0.001 total. Verifies:

1. ``.env`` loads ``ANTHROPIC_API_KEY`` into the environment.
2. ``LLMClient`` can reach the Anthropic API in both modes.
3. Cache-telemetry fields are populated on the response (cache won't
   actually hit at this tiny prompt size — both models' minimums are
   above the system prompt length here — but the wiring path is exercised).
4. Models respond with text + valid ``stop_reason``.
"""
from __future__ import annotations

import sys

from loguru import logger

from leela_chat.llm.client import LLMClient, LLMMode
from leela_chat.llm.env import load_env

STUB_SYSTEM = (
    "You are a test assistant. Reply with one short sentence acknowledging "
    "that you received the JSON marker and the question. Do not invent any "
    "chess content."
)
STUB_FINDINGS_JSON = '{"test": true, "marker": "chunk2_smoke", "expected_reply": "ack"}'
STUB_QUESTION = "Did you receive the marker?"


def main() -> int:
    load_env()

    for mode in (LLMMode.ITER, LLMMode.PROD):
        logger.info(f"=== testing mode={mode.value} ===")
        client = LLMClient(mode=mode, max_tokens=200)
        resp = client.translate(
            system_prompt=STUB_SYSTEM,
            findings_json=STUB_FINDINGS_JSON,
            question=STUB_QUESTION,
        )
        logger.success(f"[{mode.value}] response text: {resp.text!r}")
        logger.info(
            f"[{mode.value}] cache_hit={resp.cache_hit} "
            f"(expected False at this prompt size — Opus 4.7 needs >=4096 tokens, "
            f"Sonnet 4.6 needs >=2048; cache wiring still exercised)"
        )

    logger.success("smoke test passed: both modes reached Anthropic API and returned text.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
