"""Chunk 5 smoke test: run what_is_leela_seeing end-to-end on dev fixtures.

NOT eval data. These FENs are throwaway development fixtures, not the
canonical eval set (``tests/eval/eval_positions.yaml`` — friends authoring
per the G0.2 override).

Loads BT4 once, iterates ``FIXTURES`` (FEN + question), runs handler then
translator, prints findings + final response + telemetry per position.

Run on the pod (BT4 weights + GPU required):

    uv run python -m leela_chat.scripts.smoke_test_handler

Defaults to ITER mode (Sonnet 4.6) for cheap iteration. Pass ``--prod`` to
use Opus 4.7 (PROD mode).
"""
from __future__ import annotations

import argparse
import json
import time

import chess
from loguru import logger

from leela_chat.handlers.what_is_leela_seeing import (
    HANDLER_NAME,
    PROMPT_VERSION,
    handle_what_is_leela_seeing,
    templated_fallback,
)
from leela_chat.llm.client import LLMClient, LLMMode
from leela_chat.llm.env import load_env
from leela_chat.llm.translate import translate_and_validate
from leela_chat.network import load_bt4


# Dev fixture FENs, supplied by the user 2026-05-18. Format: (fen, question,
# comment). These are NOT eval entries (friends authoring per G0.2 override) —
# throwaway smoke positions for Chunk 5 pipeline verification.
FIXTURES: list[tuple[str, str, str]] = [
    ("8/3r2k1/1b4p1/1P3p1p/8/1RN2KP1/1P6/8 b - - 0 1",
     "What is Leela seeing here?",
     "endgame: R+B+N+2P each side, locked pawn structure"),
    ("1k3r2/1pb4p/p1q5/4PN2/5Q2/2P5/PP4bP/R1B3K1 w - - 0 1",
     "What is Leela seeing here?",
     "middlegame: both queens active, dynamic position"),
    ("8/8/1P6/1K4k1/r5P1/8/p7/R7 b - - 0 1",
     "What is Leela seeing here?",
     "rook+pawn endgame: both sides racing to promote"),
    ("1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1",
     "What is Leela seeing here?",
     "middlegame: attack vs attack with queens"),
    ("8/5P1b/4k3/1K5B/2p5/2Bn4/6p1/8 w - - 0 1",
     "What is Leela seeing here?",
     "minor-piece endgame with passed pawns near promotion"),
    ("r1bqkbnr/pp1p1ppp/3P2n1/4pP2/4P3/2P5/6PP/RNBQKBNR b KQkq - 0 1",
     "What is Leela seeing here?",
     "irregular opening: White pawn on d6, Black knight on g6"),
    ("rnbqkbnr/pppp1ppp/8/8/2B1P3/2p5/PP3PPP/RNBQK1NR w KQkq - 0 1",
     "What is Leela seeing here?",
     "opening: Black has advanced pawn to c3 (gambit-style)"),
    ("rnbqk2r/pp1p1ppp/4pn2/2p5/1bPP4/5N2/PP1BPPPP/RN1QKB1R w KQkq - 0 1",
     "What is Leela seeing here?",
     "opening: Nimzo-style with Bb4 pin"),
    ("3r4/8/8/5p2/2K2k2/8/8/4R3 w - - 0 1",
     "What is Leela seeing here?",
     "rook endgame: K+R+P vs K+R, kings facing"),
    ("r1b1kbnr/pppp1ppp/8/4N1q1/2BnP3/8/PPPP1PPP/RNBQK2R w KQkq - 0 1",
     "What is Leela seeing here?",
     "wild opening melee: Qg5, knight on e5, Nd4 hanging"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Use Opus 4.7 (PROD mode). Default is Sonnet 4.6 (ITER, cheap).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="How many top-policy moves the handler exposes to the LLM (default 5).",
    )
    args = parser.parse_args()

    if not FIXTURES:
        logger.error(
            "FIXTURES is empty. Edit "
            "leela_chat/scripts/smoke_test_handler.py and add (FEN, question, "
            "comment) tuples to FIXTURES before running."
        )
        return

    load_env()
    mode = LLMMode.PROD if args.prod else LLMMode.ITER

    logger.info(f"Loading BT4 (this can take ~20s on first call)...")
    t0 = time.perf_counter()
    model = load_bt4()
    logger.info(f"BT4 loaded in {time.perf_counter() - t0:.1f}s")

    client = LLMClient(mode=mode)

    for i, (fen, question, comment) in enumerate(FIXTURES, start=1):
        print(f"\n{'=' * 72}")
        print(f"[{i}/{len(FIXTURES)}] {comment}")
        print(f"FEN: {fen}")
        print(f"Q:   {question}")
        print("-" * 72)

        try:
            board = chess.Board(fen)
        except ValueError as e:
            logger.error(f"Invalid FEN, skipping: {e}")
            continue

        t_handler = time.perf_counter()
        findings = handle_what_is_leela_seeing(board, model=model, top_n=args.top_n)
        t_translate = time.perf_counter()

        result = translate_and_validate(
            findings=findings,
            question=question,
            handler_name=HANDLER_NAME,
            prompt_version=PROMPT_VERSION,
            templated_fallback=templated_fallback,
            mode=mode,
            client=client,
        )
        t_done = time.perf_counter()

        # Findings (truncated dump for readability).
        print("Findings:")
        print(json.dumps(findings.model_dump(), indent=2))
        print()
        print(f"Response (source={result.source}, validation_ok={result.validation.ok}):")
        print(result.text)
        if not result.validation.ok:
            print()
            print(f"Last-attempt validation errors ({len(result.validation.errors)}):")
            for e in result.validation.errors:
                print(f"  - {e.kind}: {e.detail}")

        # Telemetry.
        print()
        print(
            f"Latency: handler={t_translate - t_handler:.2f}s "
            f"translate={t_done - t_translate:.2f}s "
            f"total={t_done - t_handler:.2f}s "
            f"llm_calls={len(result.llm_responses)}"
        )
        total_in = sum(r.input_tokens for r in result.llm_responses)
        total_out = sum(r.output_tokens for r in result.llm_responses)
        print(f"Tokens: in={total_in} out={total_out}")


if __name__ == "__main__":
    main()
