"""leela-chat CLI: REPL (default) + one-shot ``ask`` subcommand.

Per ADR-0005 G1.5: REPL is for human interactive use; the one-shot ``ask``
subcommand is what the eval grader (Phase 1 Chunk 7) calls.

Per ADR-0005 G1.2: ``--mode`` selects the LLM tier — ``iter`` (Sonnet,
prompt-design iteration) vs ``prod`` (Opus, eval-set runs and live use).
Default is ``prod``.

This is a sketch — argparse structure + subcommand dispatch. The bodies
raise NotImplementedError; real implementations land in Phase 1 Chunk 6.

Run with:
    uv run python -m leela_chat.cli              # REPL
    uv run python -m leela_chat.cli ask --fen ... --question ...
"""
from __future__ import annotations

import argparse


def cmd_repl(args: argparse.Namespace) -> int:
    """Interactive REPL: prompt for FEN, then loop on questions."""
    raise NotImplementedError("REPL implemented in Phase 1 Chunk 6.")


def cmd_ask(args: argparse.Namespace) -> int:
    """One-shot: load FEN, ask one question, print the response and exit."""
    raise NotImplementedError("`ask` subcommand implemented in Phase 1 Chunk 6.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leela-chat",
        description="Conversational interface to Leela Chess Zero.",
    )
    sub = parser.add_subparsers(dest="cmd")

    # `repl` is the default if no subcommand
    sub.add_parser("repl", help="Interactive REPL (default).")

    p_ask = sub.add_parser("ask", help="One-shot question against a single position.")
    p_ask.add_argument(
        "--fen", required=True,
        help="Position to analyze, in FEN notation.",
    )
    p_ask.add_argument(
        "--question", required=True,
        help="Natural-language question.",
    )
    p_ask.add_argument(
        "--handler",
        choices=["what_is_leela_seeing", "why_this_move", "did_you_consider"],
        help="Force a specific handler. If omitted, the classifier picks one.",
    )
    p_ask.add_argument(
        "--mode",
        choices=["iter", "prod"],
        default="prod",
        help="LLM tier (ADR-0005): 'iter' = Sonnet, 'prod' = Opus.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd is None or args.cmd == "repl":
        return cmd_repl(args)
    if args.cmd == "ask":
        return cmd_ask(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
