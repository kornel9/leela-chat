"""Capture one walkthrough_rxd4 row in smoke-battery JSONL format.

Per Chunk 7c.7 sub-fix 1b (real-row regression-test fixture for the
validator's incomplete_trough_citation check) AND sub-fix 1d (apples-
to-apples baseline-delta for analyzer's walkthrough-only section).
Both sub-fixes need a JSONL row whose schema matches the smoke
battery's output (including the post-7c.7 trajectory_summaries and
cache_creation_input_tokens / cache_read_input_tokens fields).

The smoke battery has no per-position filter; this script builds the
exact position dict + handler/mode triple needed and reuses smoke
battery's ``_run_one`` to write the row.

**Findings-dict sidecar (Chunk 7c.7 sub-fix 1b enhancement):** the
smoke battery JSONL row only persists ``findings_summary`` (lossy)
by design. The regression test for the validator's
incomplete_trough_citation check needs the FULL findings_dict so it
can replay validate_response, not just assert on the snapshot
validation_errors. This script writes a sidecar ``<out>.findings.json``
alongside the JSONL containing the model_dump of the handler's
findings + the LLM prose. The regression test loads the sidecar.

Requires pod (BT4 + Stockfish + Lc0). One LLM call (~$0.05). Output
written to ``logs/capture_walkthrough_<timestamp>.jsonl`` by default;
sidecar at ``logs/capture_walkthrough_<timestamp>.findings.json``.

Usage (on the pod):

    uv run python -m leela_chat.scripts.capture_walkthrough_row

    uv run python -m leela_chat.scripts.capture_walkthrough_row \\
        --out /tmp/walkthrough_row.jsonl
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from loguru import logger

WALKTHROUGH_FEN = "1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1"
WALKTHROUGH_FOCAL_UCI = "d8d4"  # Rxd4 — the canonical focal move


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=None,
        help="JSONL output path (default: logs/capture_walkthrough_<ts>.jsonl)",
    )
    parser.add_argument(
        "--handler", default="did_you_consider",
        choices=("what_is_leela_seeing", "why_this_move", "did_you_consider"),
        help="Handler to run (default: did_you_consider for 7c spot-check parity)",
    )
    parser.add_argument(
        "--mode", default="deep", choices=("quick", "deep"),
        help="Latency mode (default: deep; did_you_consider auto-upgrades quick→deep)",
    )
    args = parser.parse_args()

    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.out is None:
        args.out = Path(f"logs/capture_walkthrough_{timestamp}.jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Late-bind the context build + _run_one import so the script can
    # be inspected off-pod without crashing on the BT4 / Lc0 imports.
    from leela_chat.cli.__main__ import _build_context
    from leela_chat.llm.client import LLMMode
    from leela_chat.scripts.run_smoke_battery import _mcts_bestmove, _run_one

    logger.info(f"output: {args.out}")
    logger.info(f"building LLM context (mode=PROD)")
    ctx = _build_context(mode=LLMMode.PROD, verbose=False)

    # Build the position dict in the schema _run_one expects.
    import chess
    position = {
        "id": "walkthrough_rxd4",
        "fen": WALKTHROUGH_FEN,
        "category": "tactical",
        "stratum": "walkthrough_rxd4_fixture",
        "themes": ["sacrifice", "middlegame"],
        "rating": 0,  # canonical fixture, not from puzzle DB
        "focal_uci": WALKTHROUGH_FOCAL_UCI,
        "focal_source": "fixture",
    }

    # Pre-pass: MCTS bestmove (so focal_matches_mcts_pick is populated).
    logger.info("MCTS bestmove pre-pass (1000 nodes)")
    try:
        board = chess.Board(position["fen"])
        position["_mcts_bestmove_uci"] = _mcts_bestmove(ctx, board)
    except Exception as exc:
        logger.warning(f"MCTS bestmove failed: {exc}")
        position["_mcts_bestmove_uci"] = None

    logger.info(f"running {args.handler}/{args.mode} on walkthrough_rxd4 (single LLM call)")
    # Single-call capture: previously _run_one was called for the row
    # AND _answer_one was called separately for the sidecar findings_dict
    # — TWO LLM calls per script run with independent MCTS + LLM
    # nondeterminism, producing inconsistent row prose vs sidecar
    # findings. The 7c.7 capture revealed the inconsistency: row's
    # prose was clean (validation_ok=True), sidecar's prose was
    # templated-fallback with 7 illegal_move errors. The regression
    # test needs row prose + findings_dict from the SAME call. Refactored
    # to call _answer_one once + build BOTH the row AND the sidecar
    # from the unified result.
    from leela_chat.cli.__main__ import _answer_one
    import time
    question = (
        "Did you consider Rxd4?" if args.handler == "did_you_consider"
        else "Why this move?" if args.handler == "why_this_move"
        else "What is Leela seeing here?"
    )
    t0 = time.perf_counter()
    result, handler_s, translate_s, findings_dict = _answer_one(
        ctx, position["fen"], question,
        handler_override=args.handler,
        move_uci=position.get("focal_uci"),
        mode=args.mode,
    )
    _total_s = time.perf_counter() - t0

    # Build smoke-battery-shaped row from the single call. Mirrors
    # _run_one's row construction logic (field-for-field). Importing
    # the validator helper to populate trajectory_summaries — same
    # pattern as run_smoke_battery.py per 7c.7 sub-fix 1a-struct.
    from leela_chat.validator import _collect_trajectory_summaries
    from leela_chat.scripts.run_smoke_battery import (
        _extract_findings_summary, _extract_interpretive_claims,
    )
    focal_uci = position["focal_uci"] if args.handler in ("why_this_move", "did_you_consider") else None
    total_in = sum(r.input_tokens for r in result.llm_responses)
    total_out = sum(r.output_tokens for r in result.llm_responses)
    total_cache_creation = sum(r.cache_creation_input_tokens for r in result.llm_responses)
    total_cache_read = sum(r.cache_read_input_tokens for r in result.llm_responses)
    row = {
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "position_id": position["id"],
        "fen": position["fen"],
        "category": position["category"],
        "stratum": position["stratum"],
        "themes": position["themes"],
        "rating": position["rating"],
        "handler": args.handler,
        "mode": args.mode,
        "focal_uci": focal_uci,
        "focal_source": position.get("focal_source"),
        "mcts_bestmove_uci": position.get("_mcts_bestmove_uci"),
        "focal_matches_mcts_pick": (
            focal_uci is not None
            and position.get("_mcts_bestmove_uci") is not None
            and focal_uci == position["_mcts_bestmove_uci"]
        ),
        "source": result.source,
        "validation_ok": result.validation.ok,
        "validation_errors": [
            {"kind": e.kind, "detail": e.detail, "span": e.span}
            for e in result.validation.errors
        ],
        "prompt_version": result.prompt_version,
        "handler_s": handler_s,
        "translate_s": translate_s,
        "llm_calls": len(result.llm_responses),
        "input_tokens": total_in,
        "cache_creation_input_tokens": total_cache_creation,
        "cache_read_input_tokens": total_cache_read,
        "output_tokens": total_out,
        "prose": result.text,
        "llm_attempt_texts": [r.text for r in result.llm_responses],
        "findings_summary": _extract_findings_summary(findings_dict),
        "interpretive_claims": _extract_interpretive_claims(
            result.text, _extract_findings_summary(findings_dict),
        ),
        "trajectory_summaries": (
            _collect_trajectory_summaries(findings_dict) if findings_dict else []
        ),
    }
    with args.out.open("w") as f:
        f.write(json.dumps(row) + "\n")
    logger.info(f"wrote 1 row to {args.out}")

    # Sidecar from the SAME _answer_one result — prose + findings_dict
    # + validation_errors are guaranteed consistent.
    sidecar_path = args.out.with_suffix(".findings.json")
    sidecar = {
        "captured_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "position_id": position["id"],
        "fen": position["fen"],
        "handler": args.handler,
        "mode": args.mode,
        "focal_uci": position.get("focal_uci"),
        "prompt_version": result.prompt_version,
        "prose": result.text,
        "validation_ok": result.validation.ok,
        "validation_errors": [
            {"kind": e.kind, "detail": e.detail, "span": e.span}
            for e in result.validation.errors
        ],
        "findings_dict": findings_dict,
    }
    with sidecar_path.open("w") as f:
        json.dump(sidecar, f, indent=2)
    logger.info(f"wrote sidecar fixture to {sidecar_path}")

    # Surface a quick summary.
    print(f"\n=== Captured row ===")
    print(f"position_id     : {row['position_id']}")
    print(f"handler/mode    : {row['handler']}/{row['mode']}")
    print(f"prompt_version  : {row.get('prompt_version')}")
    print(f"source          : {row['source']}")
    print(f"validation_ok   : {row['validation_ok']}")
    print(f"input_tokens    : {row.get('input_tokens')}")
    print(f"cache_write     : {row.get('cache_creation_input_tokens', 'N/A')}")
    print(f"cache_read      : {row.get('cache_read_input_tokens', 'N/A')}")
    print(f"output_tokens   : {row.get('output_tokens')}")
    print(f"translate_s     : {row.get('translate_s'):.2f}s")
    print(f"handler_s       : {row.get('handler_s'):.2f}s")
    print(f"trajectory_summaries: {len(row.get('trajectory_summaries', []))}")
    print(f"validation_errors: {len(row.get('validation_errors') or [])}")
    if row.get("validation_errors"):
        for e in row["validation_errors"]:
            print(f"  - {e.get('kind')}: {e.get('detail', '')[:120]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
