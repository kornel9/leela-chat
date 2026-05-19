"""leela-chat CLI: REPL (default) + one-shot ``ask`` subcommand.

Per ADR-0005 G1.5: REPL is for human interactive use; the one-shot ``ask``
subcommand is the entry point Chunk 7's eval grader would have called
(skipped per ADR-0010, but the contract stays — Phase 3+ may want it back).

Per ADR-0005 G1.2: ``--mode`` selects the LLM tier — ``iter`` (Sonnet,
prompt-design iteration) vs ``prod`` (Opus, eval-set runs and live use).
Default is ``prod``.

Run with:

    uv run python -m leela_chat.cli                                    # REPL
    uv run python -m leela_chat.cli ask --fen ... --question ...       # one-shot
    uv run python -m leela_chat.cli ask --fen ... --question ... -v    # + telemetry

Resource lifecycle:
- Both subcommands load BT4 once (~30-50 s), start one Stockfish UCI
  subprocess, and — when the Lc0 binary + weights are present — one
  Lc0 subprocess for MCTS. Both engines are closed in a ``finally`` block.
- Lc0 paths default to the pod conventions; override with ``LC0_PATH``
  and ``LC0_WEIGHTS`` env vars. If either is missing, the CLI degrades
  gracefully to policy-only mode (``findings.mcts = None``) and the v4
  prompt template handles the no-MCTS case. Useful for local dev boxes
  without a Lc0 build.
- The CLI is intentionally not optimised for repeat one-shot calls — the
  REPL is the right mode for interactive personal use. One-shot exists
  for scripting and for the eval-grader contract.
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import chess
import chess.engine
from loguru import logger

from leela_chat.classifier import classify
from leela_chat.conversation import ConversationState, Turn
from leela_chat.handlers.what_is_leela_seeing import (
    HANDLER_NAME,
    PROMPT_VERSION,
    handle_what_is_leela_seeing,
    templated_fallback,
)
from leela_chat.llm.client import LLMClient, LLMMode
from leela_chat.llm.env import load_env
from leela_chat.llm.translate import ValidatedTranslation, translate_and_validate
from leela_chat.network import load_bt4

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer

    from leela_chat.engine.lc0_engine import Lc0Engine


# Pod-default paths for Lc0 binary + BT4 weights (Phase-0 build conventions).
# Override via LC0_PATH / LC0_WEIGHTS env vars on a local dev box.
_DEFAULT_LC0_BINARY = Path("/workspace/bin/lc0")
_DEFAULT_LC0_WEIGHTS = Path(
    "/workspace/weights/BT4-1024x15x32h-swa-6147500/model.pb.gz"
)


# Mapping from --handler choices to the handler implementations. Phase 1
# has only one; this dict is the extension point for handlers 2/3 once they
# land after Phase 2 per the architecture-of-efficiency decision.
HANDLERS_AVAILABLE: list[str] = [HANDLER_NAME]


# === Shared session context =================================================


@dataclass
class CliContext:
    """Resources held for the lifetime of a CLI invocation.

    Built once per ``leela-chat`` invocation by :func:`_build_context`. The
    REPL holds it for the whole interactive session; the one-shot ``ask``
    holds it for one Q&A. Either way, :func:`_close_context` must run on
    exit (handled by the ``finally`` block in each command).

    ``lc0`` is optional: when the binary or weights are missing (typical
    on a local dev box), the field is ``None`` and Handler 1 skips MCTS.
    Pod runs always have a real :class:`Lc0Engine`.
    """

    model: "HookedTransformer"
    stockfish: chess.engine.SimpleEngine
    client: LLMClient
    verbose: bool
    lc0: "Lc0Engine | None" = None


def _build_context(mode: LLMMode, verbose: bool) -> CliContext:
    """Load env, BT4, Stockfish, optional Lc0, and LLM client.

    Lc0 is best-effort: if the binary or weights are missing at the
    configured paths, log a warning and proceed in policy-only mode.
    The handler + v4 prompt + templated fallback all handle
    ``findings.mcts = None`` gracefully.
    """
    load_env()
    stockfish_path = os.environ.get("STOCKFISH_PATH", "/usr/games/stockfish")
    lc0_path = Path(os.environ.get("LC0_PATH", str(_DEFAULT_LC0_BINARY)))
    lc0_weights = Path(os.environ.get("LC0_WEIGHTS", str(_DEFAULT_LC0_WEIGHTS)))

    logger.info("Loading BT4 (this can take ~30-50 s on first call)...")
    t0 = time.perf_counter()
    model = load_bt4()
    logger.info(f"BT4 loaded in {time.perf_counter() - t0:.1f}s")

    logger.info(f"Starting Stockfish from {stockfish_path}...")
    stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    logger.info(f"Stockfish ready: id_name={stockfish.id.get('name', '?')}")

    lc0: "Lc0Engine | None" = None
    if lc0_path.exists() and lc0_weights.exists():
        from leela_chat.engine.lc0_engine import Lc0Engine
        lc0 = Lc0Engine(binary=lc0_path, weights=lc0_weights)
    else:
        missing = []
        if not lc0_path.exists():
            missing.append(f"binary at {lc0_path}")
        if not lc0_weights.exists():
            missing.append(f"weights at {lc0_weights}")
        logger.warning(
            "Lc0 missing ({}); proceeding without MCTS — handler will use "
            "policy + Stockfish only.".format(", ".join(missing)),
        )

    client = LLMClient(mode=mode)
    return CliContext(
        model=model,
        stockfish=stockfish,
        client=client,
        verbose=verbose,
        lc0=lc0,
    )


def _close_context(ctx: CliContext) -> None:
    """Cleanly close the engine subprocesses. Idempotent in practice."""
    try:
        ctx.stockfish.close()
    except Exception as exc:  # pragma: no cover — defensive cleanup
        logger.warning(f"Stockfish close failed: {type(exc).__name__}: {exc}")
    if ctx.lc0 is not None:
        try:
            ctx.lc0.close()
        except Exception as exc:  # pragma: no cover — defensive cleanup
            logger.warning(f"Lc0 close failed: {type(exc).__name__}: {exc}")


# === Per-question core ======================================================


def _answer_one(
    ctx: CliContext,
    fen: str,
    question: str,
    *,
    handler_override: str | None = None,
) -> tuple[ValidatedTranslation, float, float]:
    """Run handler + translate for one (fen, question) pair.

    Args:
        ctx: Active CLI context.
        fen: Position to analyze. Caller is responsible for FEN validity —
            this function does NOT catch ``ValueError`` from ``chess.Board``.
        question: User's literal question.
        handler_override: If supplied, force this handler instead of asking
            the classifier. Must be one of ``HANDLERS_AVAILABLE``.

    Returns:
        Tuple of (validated translation, handler latency in seconds,
        translate latency in seconds).
    """
    board = chess.Board(fen)
    handler_name = handler_override or classify(
        question, current_handlers=HANDLERS_AVAILABLE
    )
    if handler_name not in HANDLERS_AVAILABLE:
        raise ValueError(
            f"Handler {handler_name!r} not implemented in this build "
            f"(available: {HANDLERS_AVAILABLE})."
        )

    # Phase 1: only one handler exists. When handlers 2/3 land we'll dispatch
    # on handler_name. Keeping this explicit so the future extension is
    # mechanical, not a refactor.
    t0 = time.perf_counter()
    findings = handle_what_is_leela_seeing(
        board,
        model=ctx.model,
        stockfish_engine=ctx.stockfish,
        lc0_engine=ctx.lc0,
    )
    t1 = time.perf_counter()
    result = translate_and_validate(
        findings=findings,
        question=question,
        handler_name=HANDLER_NAME,
        templated_fallback=templated_fallback,
        prompt_version=PROMPT_VERSION,
        client=ctx.client,
    )
    t2 = time.perf_counter()
    return result, t1 - t0, t2 - t1


def _emit_telemetry(
    result: ValidatedTranslation, handler_s: float, translate_s: float
) -> None:
    """Print one-line telemetry footer to stderr."""
    total_in = sum(r.input_tokens for r in result.llm_responses)
    total_out = sum(r.output_tokens for r in result.llm_responses)
    print(
        f"[source={result.source} validation_ok={result.validation.ok} "
        f"handler={handler_s:.2f}s translate={translate_s:.2f}s "
        f"llm_calls={len(result.llm_responses)} "
        f"tokens=in:{total_in}/out:{total_out}]",
        file=sys.stderr,
    )


# === cmd_ask: one-shot ======================================================


def cmd_ask(args: argparse.Namespace) -> int:
    """One-shot: load context, ask one question, print response, exit."""
    try:
        chess.Board(args.fen)
    except ValueError as exc:
        print(f"error: invalid FEN: {exc}", file=sys.stderr)
        return 2

    mode = LLMMode(args.mode)
    ctx = _build_context(mode=mode, verbose=args.verbose)
    try:
        try:
            result, handler_s, translate_s = _answer_one(
                ctx, args.fen, args.question, handler_override=args.handler
            )
        except ValueError as exc:
            # Most likely: --handler points at a not-yet-implemented handler.
            # FEN validity was already checked above.
            print(f"error: {exc}", file=sys.stderr)
            return 2

        print(result.text)
        if args.verbose:
            _emit_telemetry(result, handler_s, translate_s)
        return 0
    finally:
        _close_context(ctx)


# === cmd_repl: interactive ==================================================


REPL_HELP = """\
Commands:
  :fen <FEN>    Switch to a new position (replaces the current one).
  :history      Show the Q&A history of this session.
  :help         Show this message.
  :quit         Exit the REPL.

Anything else is treated as a question about the current position.
"""


def _prompt_initial_fen() -> str | None:
    """Loop until a valid FEN is entered, or return None on quit/EOF."""
    while True:
        try:
            raw = input("Position FEN (or :quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()  # newline so the next shell prompt looks clean
            return None
        if raw in (":quit", ":q"):
            return None
        if not raw:
            continue
        try:
            chess.Board(raw)
        except ValueError as exc:
            print(f"  invalid FEN: {exc}", file=sys.stderr)
            continue
        return raw


def _print_history(state: ConversationState) -> None:
    if not state.turns:
        print("(no turns yet)")
        return
    for i, turn in enumerate(state.turns, start=1):
        print(f"--- turn {i} [source={turn.source}] ---")
        print(f"FEN: {turn.fen}")
        print(f"Q:   {turn.question}")
        print(f"A:   {turn.response}")


def cmd_repl(args: argparse.Namespace) -> int:
    """Interactive REPL. Returns process exit code."""
    mode = LLMMode(args.mode)
    ctx = _build_context(mode=mode, verbose=args.verbose)
    state = ConversationState()
    try:
        initial = _prompt_initial_fen()
        if initial is None:
            return 0
        state.current_fen = initial
        print(REPL_HELP)

        while True:
            try:
                raw = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not raw:
                continue

            # Dispatch on commands. Anything not starting with ":" is a
            # question against state.current_fen.
            if raw in (":quit", ":q"):
                return 0
            if raw in (":help", ":h", ":?"):
                print(REPL_HELP)
                continue
            if raw == ":history":
                _print_history(state)
                continue
            if raw.startswith(":fen"):
                parts = shlex.split(raw)
                if len(parts) < 2:
                    print("usage: :fen <FEN>", file=sys.stderr)
                    continue
                # Join in case the user pasted unquoted FEN with spaces.
                new_fen = " ".join(parts[1:])
                try:
                    chess.Board(new_fen)
                except ValueError as exc:
                    print(f"  invalid FEN: {exc}", file=sys.stderr)
                    continue
                state.current_fen = new_fen
                print(f"(position changed)")
                continue
            if raw.startswith(":"):
                print(f"unknown command: {raw}. Type :help for commands.", file=sys.stderr)
                continue

            # Treat as question.
            assert state.current_fen is not None  # set by _prompt_initial_fen
            try:
                result, handler_s, translate_s = _answer_one(
                    ctx, state.current_fen, raw, handler_override=args.handler
                )
            except ValueError as exc:
                # Shouldn't happen for a question (FEN already validated;
                # handler_override is whatever the user passed at startup).
                print(f"error: {exc}", file=sys.stderr)
                continue
            except Exception as exc:  # noqa: BLE001 — REPL should keep going
                logger.exception("Unhandled error during answer:")
                print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue

            print(result.text)
            state.turns.append(Turn(
                fen=state.current_fen,
                question=raw,
                response=result.text,
                source=result.source,
            ))
            if args.verbose:
                _emit_telemetry(result, handler_s, translate_s)
    finally:
        _close_context(ctx)


# === argparse ===============================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leela-chat",
        description="Conversational interface to Leela Chess Zero.",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_repl = sub.add_parser("repl", help="Interactive REPL (default).")
    _add_common_flags(p_repl)

    p_ask = sub.add_parser("ask", help="One-shot question against a single position.")
    p_ask.add_argument(
        "--fen", required=True,
        help="Position to analyze, in FEN notation.",
    )
    p_ask.add_argument(
        "--question", required=True,
        help="Natural-language question.",
    )
    _add_common_flags(p_ask)

    # Repl is the default when no subcommand — flags must also be accepted
    # at the top level so `leela-chat --mode iter` works without `repl`.
    _add_common_flags(parser)

    return parser


def _add_common_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--handler",
        choices=HANDLERS_AVAILABLE,
        help=(
            "Force a specific handler. Phase 1 has only one; the classifier "
            "routes to it by default. Pre-wired for handlers 2/3."
        ),
    )
    p.add_argument(
        "--mode",
        choices=["iter", "prod"],
        default="prod",
        help="LLM tier (ADR-0005): 'iter' = Sonnet, 'prod' = Opus.",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help=(
            "Emit a one-line telemetry footer to stderr after each response "
            "(source, validation status, latency, token counts). Off by default."
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Default to REPL if no subcommand was given.
    if args.cmd is None or args.cmd == "repl":
        return cmd_repl(args)
    if args.cmd == "ask":
        return cmd_ask(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
