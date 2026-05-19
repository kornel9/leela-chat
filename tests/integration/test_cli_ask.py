"""Integration test for the CLI's one-shot ``ask`` subcommand.

Mocks the heavy resources (``_build_context`` and ``_answer_one``) so the
test exercises argparse parsing, dispatch, stdout/stderr handling, and exit
codes — not the actual BT4 / Stockfish / LLM stack. That stack is smoke-
tested separately on the pod.

REPL is harder to unit-test cleanly (would require simulating stdin); the
shared ``_answer_one`` / ``_build_context`` / ``_close_context`` core that
both subcommands use is covered here. REPL UX is verified manually on the
pod per the plan's verification section.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from leela_chat.cli import __main__ as cli_main
from leela_chat.llm.client import LLMMode, LLMResponse
from leela_chat.llm.translate import ValidatedTranslation
from leela_chat.validator import ValidationResult


# Standard starting position; valid FEN, used across happy-path tests.
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _fake_ctx() -> MagicMock:
    """Build a placeholder CliContext-shaped mock for tests."""
    ctx = MagicMock(spec=cli_main.CliContext)
    ctx.verbose = False
    return ctx


def _fake_result(text: str = "Mock prose response.", source: str = "llm") -> ValidatedTranslation:
    """Build a valid ValidatedTranslation for the CLI to print."""
    return ValidatedTranslation(
        text=text,
        source=source,
        validation=ValidationResult(ok=True, errors=[]),
        llm_responses=[
            LLMResponse(
                text=text,
                model="claude-test-mock",
                input_tokens=100,
                output_tokens=50,
                stop_reason="end_turn",
            ),
        ],
        handler_name="what_is_leela_seeing",
        prompt_version=3,
    )


def _patch_heavy(monkeypatch: pytest.MonkeyPatch, *, answer_result=None) -> dict:
    """Replace the resource-loading + answering helpers with mocks.

    Returns a dict of captured side effects (mode passed to _build_context,
    handler_override passed to _answer_one) for assertions.
    """
    captured: dict = {}

    def fake_build(mode: LLMMode, verbose: bool) -> MagicMock:
        captured["mode"] = mode
        captured["verbose"] = verbose
        return _fake_ctx()

    def fake_answer(ctx, fen, question, *, handler_override=None):
        captured["fen"] = fen
        captured["question"] = question
        captured["handler_override"] = handler_override
        return answer_result or _fake_result(), 0.10, 0.20

    monkeypatch.setattr(cli_main, "_build_context", fake_build)
    monkeypatch.setattr(cli_main, "_close_context", lambda ctx: None)
    monkeypatch.setattr(cli_main, "_answer_one", fake_answer)
    return captured


def test_ask_happy_path(capsys, monkeypatch):
    """Valid FEN + question → response on stdout, nothing on stderr, exit 0."""
    captured = _patch_heavy(monkeypatch)
    code = cli_main.main(["ask", "--fen", START_FEN, "--question", "What is Leela seeing?"])
    assert code == 0
    out = capsys.readouterr()
    assert "Mock prose response." in out.out
    # No verbose flag → no telemetry on stderr.
    assert "source=" not in out.err
    assert captured["fen"] == START_FEN
    assert captured["question"] == "What is Leela seeing?"


def test_ask_verbose_emits_telemetry(capsys, monkeypatch):
    """With -v, a single telemetry line goes to stderr."""
    _patch_heavy(monkeypatch)
    code = cli_main.main([
        "ask", "--fen", START_FEN, "--question", "?", "-v",
    ])
    assert code == 0
    out = capsys.readouterr()
    assert "source=llm" in out.err
    assert "validation_ok=True" in out.err
    # Stdout still just the prose, no telemetry there.
    assert "source=" not in out.out


def test_ask_invalid_fen_exits_2(capsys, monkeypatch):
    """Invalid FEN should exit 2 BEFORE loading BT4 / Stockfish."""
    # No need to patch _build_context — FEN validation runs first and short-
    # circuits. Tracking that explicitly:
    build_called = []
    monkeypatch.setattr(
        cli_main, "_build_context",
        lambda mode, verbose: build_called.append(True) or _fake_ctx(),
    )
    monkeypatch.setattr(cli_main, "_close_context", lambda ctx: None)
    monkeypatch.setattr(cli_main, "_answer_one", lambda *a, **kw: (_fake_result(), 0.0, 0.0))

    code = cli_main.main(["ask", "--fen", "not a fen", "--question", "?"])
    assert code == 2
    out = capsys.readouterr()
    assert "invalid fen" in out.err.lower()
    assert build_called == [], "BT4 / Stockfish should not be loaded for invalid FEN"


def test_ask_mode_iter_and_prod(monkeypatch):
    """The --mode flag is parsed and threaded into _build_context."""
    captured = _patch_heavy(monkeypatch)

    cli_main.main(["ask", "--fen", START_FEN, "--question", "?", "--mode", "iter"])
    assert captured["mode"] == LLMMode.ITER

    cli_main.main(["ask", "--fen", START_FEN, "--question", "?", "--mode", "prod"])
    assert captured["mode"] == LLMMode.PROD

    # Default (no flag) should be PROD per ADR-0005.
    cli_main.main(["ask", "--fen", START_FEN, "--question", "?"])
    assert captured["mode"] == LLMMode.PROD


def test_ask_handler_override_passed_through(monkeypatch):
    """The --handler flag flows to _answer_one as handler_override."""
    captured = _patch_heavy(monkeypatch)
    cli_main.main([
        "ask", "--fen", START_FEN, "--question", "?",
        "--handler", "what_is_leela_seeing",
    ])
    assert captured["handler_override"] == "what_is_leela_seeing"

    # Without --handler, override is None (classifier picks).
    cli_main.main(["ask", "--fen", START_FEN, "--question", "?"])
    assert captured["handler_override"] is None


def test_ask_unknown_subcommand_returns_2(capsys):
    """A bogus subcommand should error out via argparse (exits via SystemExit)."""
    with pytest.raises(SystemExit):
        cli_main.main(["bogus-cmd"])
