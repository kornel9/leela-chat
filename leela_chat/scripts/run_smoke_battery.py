"""Chunk 6.8 smoke battery — stratified sampling + 100-call matrix.

Runs all 3 handlers in both modes (where applicable) against 20 positions
sampled stratified from the Lichess Puzzle DB. Captures per-run telemetry
+ prose to JSONL; the aggregate summary is written separately to
``tests/eval/smoke_summaries/<YYYY-MM-DD>.md``.

This is the MEASURE half of an eval set, NOT the grading half. Prose
quality / 2200+-player agreement still needs G0.2-friends.

Run on the pod (BT4 + Stockfish + Lc0 + corpus required):

    uv run python -m leela_chat.scripts.run_smoke_battery
    uv run python -m leela_chat.scripts.run_smoke_battery --positions 1 --dry-run

The `--dry-run` mode skips context-build + LLM calls; useful for off-pod
pipeline-correctness checks. `--positions N` caps the matrix at the
first N positions in the sampled fixture (default: all 20).

Hard gates (evaluated by the summary writer, NOT enforced here):
- 100 JSONL entries
- overall validation_ok ≥ 90%
- 0 unexplained illegal_move errors (per the §8 classification)
- templated-fallback ≤ 10%

Exit codes (Chunk 6.8.1):
- 0: success (battery ran to completion).
- 1: generic error (unhandled exception in the loop body that the CRASH
     path didn't catch).
- 2: argument error (argparse default for malformed CLI args).
- 3: circuit breaker engaged (3+ consecutive api_error responses; the
     JSONL contains a `{"_circuit_breaker": True, ...}` sentinel row
     at the end as the canonical machine-readable abort marker).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import chess
import yaml
from loguru import logger

from leela_chat.eval.lichess_puzzle_db import sample_stratified

# 20-position strata — see Chunk 6.8 plan for the rationale.
STRATA: dict[str, dict] = {
    # Tactical category — 8 puzzles, theme + rating stratified.
    # Themes: 2 fork + 2 pin + 1 sacrifice + 1 discoveredAttack + 1 skewer
    # + 1 mateIn3-or-mateIn4 (per the plan's theme spread).
    # Rating bands: each theme is anchored to a band that makes sense
    # for that pattern's typical difficulty.
    "tactical_fork":      {"themes": "fork",             "n": 2, "rating_range": (1800, 2200)},
    "tactical_pin":       {"themes": "pin",              "n": 2, "rating_range": (2200, 2600)},
    "tactical_sac":       {"themes": "sacrifice",        "n": 1, "rating_range": (2600, 3000)},
    "tactical_disc":      {"themes": "discoveredAttack", "n": 1, "rating_range": (2600, 3000)},
    "tactical_skewer":    {"themes": "skewer",           "n": 1, "rating_range": (3000, 3400),
                           "fallback_to_lower": True},
    "tactical_mateInN":   {"themes": "mateIn3",          "n": 1, "rating_range": (3400, 9999),
                           "fallback_to_lower": True},
    # Quiet middlegames — 5 at 1600-1900, excluding tactical themes so
    # we get genuinely quiet positional content.
    "quiet_middlegame":   {"themes": "middlegame",       "n": 5, "rating_range": (1600, 1900),
                           "exclude_themes": ("fork", "pin", "sacrifice", "discoveredAttack", "skewer", "mateIn1", "mateIn2", "mateIn3", "mateIn4")},
    # Endgames — 4 at 1500-2200, EXCLUDING K+N+N vs K per project memory.
    "endgame":            {"themes": "endgame",          "n": 4, "rating_range": (1500, 2200),
                           "exclude_material_signatures": ("KNN_vs_K",)},
    # Openings — 3 around move 8-15 (Lichess tags `opening` theme).
    "opening":            {"themes": "opening",          "n": 3, "rating_range": (1700, 2000)},
}

# Category aggregation: which strata belong to which user-facing category.
CATEGORY_OF: dict[str, str] = {
    "tactical_fork": "tactical_puzzle",
    "tactical_pin": "tactical_puzzle",
    "tactical_sac": "tactical_puzzle",
    "tactical_disc": "tactical_puzzle",
    "tactical_skewer": "tactical_puzzle",
    "tactical_mateInN": "tactical_puzzle",
    "quiet_middlegame": "quiet_middlegame",
    "endgame": "endgame",
    "opening": "opening",
}


def _build_fixture(seed: int = 0) -> list[dict]:
    """Sample 20 positions per STRATA and return a flat list of fixture dicts.

    Each dict contains the schema documented in tests/eval/smoke_positions.yaml:
    id, fen, source, category, stratum, themes, rating, focal_uci, focal_source.
    """
    sampled = sample_stratified(STRATA, seed=seed)
    fixture: list[dict] = []
    for stratum_name, rows in sampled.items():
        category = CATEGORY_OF[stratum_name]
        for r in rows:
            fixture.append({
                "id": f"puzzle_{r['PuzzleId']}",
                "fen": r["FEN"],
                "source": f"lichess_puzzle:{r['PuzzleId']}",
                "category": category,
                "stratum": stratum_name,
                "themes": list(r["Themes"]),
                "rating": int(r["Rating"]),
                "focal_uci": r["solver_first_uci"],
                "focal_source": "puzzle_solution",
                "material_signature": r["material_signature"],
            })
    return fixture


def _write_fixture(fixture: list[dict], path: Path) -> None:
    """Write the fixture to a YAML file with a clear header."""
    header = (
        "# tests/eval/smoke_positions.yaml — Chunk 6.8 smoke battery fixture.\n"
        "#\n"
        "# Generated by `python -m leela_chat.scripts.run_smoke_battery`\n"
        "# from the Lichess Puzzle DB corpus via sample_stratified(seed=0).\n"
        "#\n"
        "# NOT THE GRADED EVAL SET — that's tests/eval/eval_positions.yaml\n"
        "# (G0.2 friends authoring; quality-grading positions with ground truth).\n"
        "#\n"
        "# This file is the operational/regression/coverage half: 20 positions\n"
        "# sampled from Lichess for validation rate + latency + prose-pattern\n"
        "# generalization measurements. Re-generated deterministically via\n"
        "# seed=0; commit once for reproducibility.\n"
        "#\n"
        "# Per CLAUDE.md §2 Rule 3 the eval set is sacred — that rule applies\n"
        "# to eval_positions.yaml, NOT to this file. The smoke fixture may be\n"
        "# regenerated when sampling logic, strata, or seed change.\n"
        "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(header)
        yaml.safe_dump({"positions": fixture}, f, sort_keys=False, default_flow_style=False)


def _load_fixture(path: Path) -> list[dict]:
    """Load the fixture from disk; mostly used for re-runs without re-sampling."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["positions"]


def _extract_findings_summary(findings_dict: dict | None) -> dict:
    """Capture the trusted-moves arrays from a findings dict.

    Chunk 6.9 sub-item 1. Without this, the smoke battery's diagnostic
    can't detect validator_gap cases (span IS in findings but
    ``_collect_trusted_moves`` missed it); the validator-fix path in
    sub-item 2 would be theatre.

    Walks the dict for: every ``san`` / ``uci`` / ``pv_san`` / ``bestmove_san``
    / ``bestmove_uci`` / ``focal_move_san`` / ``focal_move_uci`` value.
    Output is a flat dict the aggregator can quickly compare flagged
    spans against. ~150-300 bytes per row.

    Args:
        findings_dict: The Pydantic findings model's ``model_dump()`` output,
            or None / {} if findings weren't available.

    Returns:
        Flat dict with lists/strings for each trusted-move category.
    """
    out: dict[str, Any] = {
        "top_moves_san": [],
        "top_moves_uci": [],
        "mcts_children_san": [],
        "mcts_children_uci": [],
        "mcts_bestmove_san": None,
        "mcts_bestmove_uci": None,
        "stockfish_top_san": [],
        "stockfish_top_uci": [],
        "stockfish_pv_sans": [],  # flattened across stockfish.top_moves[*].pv_san
        # Chunk 7b.2.C: added mcts_pv_sans so the interpretive-claim
        # heuristic's exclusion rule can recognise sentences that cite
        # Leela's MCTS PV moves as grounded (small gap inherited from
        # 7b — the validator's recursive walk caught these but
        # _extract_findings_summary didn't surface them in JSONL).
        "mcts_pv_sans": [],  # flattened across mcts.children[*].pv_san
        "focal_move_san": None,
        "focal_move_uci": None,
    }
    if not findings_dict:
        return out

    # Top-level focal callouts (handler 2 + 3).
    out["focal_move_san"] = findings_dict.get("focal_move_san")
    out["focal_move_uci"] = findings_dict.get("focal_move_uci")

    # Top moves (policy net candidates; all handlers).
    for tm in findings_dict.get("top_moves", []) or []:
        if isinstance(tm, dict):
            if tm.get("san"): out["top_moves_san"].append(tm["san"])
            if tm.get("uci"): out["top_moves_uci"].append(tm["uci"])

    # MCTS children + bestmove.
    mcts = findings_dict.get("mcts") or {}
    if isinstance(mcts, dict):
        out["mcts_bestmove_san"] = mcts.get("bestmove_san")
        out["mcts_bestmove_uci"] = mcts.get("bestmove_uci")
        for ch in mcts.get("children", []) or []:
            if isinstance(ch, dict):
                if ch.get("san"): out["mcts_children_san"].append(ch["san"])
                if ch.get("uci"): out["mcts_children_uci"].append(ch["uci"])
                # Chunk 7b.2.C: per-child pv_san (populated by Chunk 7b's
                # MultiPv=5 deep-mode default for top-5 candidates).
                for pv_move in ch.get("pv_san", []) or []:
                    if isinstance(pv_move, str):
                        out["mcts_pv_sans"].append(pv_move)

    # Stockfish top moves + pv_san (flattened).
    sf = findings_dict.get("stockfish") or {}
    if isinstance(sf, dict):
        for sm in sf.get("top_moves", []) or []:
            if isinstance(sm, dict):
                if sm.get("san"): out["stockfish_top_san"].append(sm["san"])
                if sm.get("uci"): out["stockfish_top_uci"].append(sm["uci"])
                for pv_move in sm.get("pv_san", []) or []:
                    if isinstance(pv_move, str):
                        out["stockfish_pv_sans"].append(pv_move)

    # Focal-specific Stockfish PV (handler 2 + 3 detailed mode).
    fsc = findings_dict.get("focal_stockfish_candidate") or {}
    if isinstance(fsc, dict):
        for pv_move in fsc.get("pv_san", []) or []:
            if isinstance(pv_move, str):
                out["stockfish_pv_sans"].append(pv_move)

    return out


# =====================================================================
# Chunk 7b.2.C — interpretive-claim capture (Option 1.5 from 7b.1 retro)
# =====================================================================
#
# The validator catches hallucinated MOVES (move-citation walker) but
# doesn't catch hallucinated INTERPRETATIONS — confident-sounding chess
# interpretations of position/PV mechanics that the findings JSON
# doesn't ground. 7b.1's canonical example: "White's queen has nowhere
# to hide after the d4-knight falls" — validator passed (no illegal
# moves cited) but the claim is interpretive and chess-accuracy is
# rater-judgment territory.
#
# This helper surfaces such interpretive-claim CANDIDATES from prose so
# the 2500+ rater at 7o can scrutinize them explicitly. It does NOT
# auto-validate (that's the multi-month Option 3 from the retro, out
# of scope). False positives are expected and FINE — the point is
# making the empirical signal at 7o extractable rather than relying on
# the rater to spot interpretive claims in unstructured prose.
#
# Two-rule design (per the user's 7b.2 plan refinement):
# - Rule A (wordlist): piece name + interpretive-claim phrase from
#   seed list (~20-30 entries; grows iteratively like the forbidden-
#   language list).
# - Rule B (structural — durable fallback): sentence asserts a
#   chess-positional claim about a piece's RELATIONSHIP to other
#   pieces (attacker/defender, mobility, scope, coordination) without
#   grounding in findings. Catches what the wordlist will miss as
#   the LLM's surface forms drift.
# Sentence flagged if EITHER rule matches AND the exclusion (grounded-
# citation sentence) doesn't fire.

import re

_PIECE_NAMES = (
    "queen", "queens", "king", "kings", "rook", "rooks",
    "bishop", "bishops", "knight", "knights", "pawn", "pawns",
)
_PIECE_NAMES_RE = re.compile(
    r"\b(?:" + "|".join(_PIECE_NAMES) + r")\b", re.IGNORECASE,
)

# Rule A wordlist — common interpretive-claim phrases that the LLM
# reaches for when narrating PV mechanics or position assessment.
# Grows iteratively across versions (same dynamic as the forbidden-
# language list — see CLAUDE.md §8 Process pitfalls).
_INTERPRETIVE_PHRASES = (
    # state / safety
    "is exposed", "has no escape", "nowhere to hide", "is undefended",
    "is hanging", "in danger", "is overworked", "is overloaded",
    # claim verbs
    "threatens", "controls", "attacks", "defends", "dominates",
    "restricts", "holds", "falls", "drops", "escapes", "traps",
    # judgment
    "is decisive", "is winning", "is losing", "becomes weak",
    "is the key square", "is the key piece",
    # canonical 7b.1 leak
    "has nowhere to hide", "finds no",
)
_INTERPRETIVE_RE = re.compile(
    r"(?i)(?:" + "|".join(re.escape(p) for p in _INTERPRETIVE_PHRASES) + r")",
)

# Rule B structural patterns — relational claims about piece interactions
# that the wordlist will miss as the LLM's surface forms drift. The
# patterns target sentence-shapes, not specific words.
_RELATIONAL_PATTERNS = (
    # mobility / safety negations (catches "finds no safety", "lacks coordination")
    r"\bfinds?\s+no\s+\w+",
    r"\blacks?\s+(?:any\s+)?\w+",
    r"\bno\s+longer\s+\w+s",
    # piece-relations (coordination, attack/defend with object)
    r"\bcoordinat(?:e|es|ion)\b",
    r"\b(?:attacks?|defends?|pressures?|controls?|guards?)\s+(?:the\s+)?\w+",
    # mobility framing
    r"\bmobility\s+of\b",
    r"\bscope\s+of\b",
    # generic positional-claim shapes
    r"\b(?:weak|exposed|loose|hanging|stranded|trapped|cornered)\s+(?:king|queen|rook|bishop|knight|pawn|piece|pieces)\b",
)
_RELATIONAL_RE = re.compile(
    r"(?i)(?:" + "|".join(_RELATIONAL_PATTERNS) + r")",
)

# Sentence splitter — preserves punctuation so each yielded sentence
# can be re-rendered. Splits on `. `, `! `, `? ` AND at end of text.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(prose: str) -> list[str]:
    """Split prose into sentences, preserving terminal punctuation."""
    parts = _SENTENCE_SPLIT_RE.split(prose.strip())
    return [s.strip() for s in parts if s.strip()]


def _sentence_is_grounded_citation(sentence: str, findings_summary: dict) -> bool:
    """True if the sentence's substantive content is a direct findings citation.

    Grounded = contains a SAN/UCI move from findings_summary AND at least
    one substantive numeric citation (visit count / Q value / eval /
    rank / percentage). These are number-narrating sentences, NOT
    interpretive ones, and should not be flagged.

    "Substantive numeric" = 3+ consecutive digits (visit counts, depths,
    rank numbers in trajectories) OR a signed/unsigned decimal (Q values,
    eval values) OR a percentage. **Single-digit-after-letter square
    references like "d4" / "e5" are NOT substantive numerics** — they're
    coordinate labels, not findings-values. Without this distinction
    the heuristic would mark interpretive PV-narration sentences like
    "After Rxd4, the d4 knight falls" as grounded (Rxd4 trusted, "d4"
    contains a digit) and miss the flag.
    """
    if not findings_summary:
        return False
    # Gather all trusted SAN/UCI strings.
    trusted: set[str] = set()
    for key in ("top_moves_san", "top_moves_uci", "mcts_children_san",
                "mcts_children_uci", "stockfish_top_san", "stockfish_top_uci",
                "stockfish_pv_sans", "mcts_pv_sans"):
        for s in findings_summary.get(key, []) or []:
            if isinstance(s, str):
                trusted.add(s)
    for key in ("mcts_bestmove_san", "mcts_bestmove_uci",
                "focal_move_san", "focal_move_uci"):
        v = findings_summary.get(key)
        if isinstance(v, str):
            trusted.add(v)
    # Sentence must contain at least one trusted move AND a substantive numeric.
    has_trusted_move = any(m and m in sentence for m in trusted)
    has_substantive_numeric = bool(re.search(
        r"\d{3,}|[+-]?\d+\.\d+|\d+%", sentence,
    ))
    return has_trusted_move and has_substantive_numeric


def _extract_interpretive_claims(
    prose: str, findings_summary: dict,
) -> list[str]:
    """Surface candidate interpretive-claim sentences for 7o rater review.

    Returns a list of sentences from `prose` that look like chess-content
    interpretations not grounded in findings_summary. False positives are
    expected; the rater judges.

    The two-rule design (wordlist + structural) per the 7b.2 plan: Rule
    A catches common surface forms via a seed wordlist; Rule B catches
    relational/positional-claim shapes the wordlist will miss as the
    LLM's surface forms drift.
    """
    if not prose or not prose.strip():
        return []
    candidates: list[str] = []
    for sentence in _split_sentences(prose):
        if _sentence_is_grounded_citation(sentence, findings_summary):
            continue
        rule_a_fires = bool(
            _PIECE_NAMES_RE.search(sentence)
            and _INTERPRETIVE_RE.search(sentence)
        )
        rule_b_fires = bool(_RELATIONAL_RE.search(sentence))
        if rule_a_fires or rule_b_fires:
            candidates.append(sentence)
    return candidates


def _mcts_bestmove(ctx: Any, board: chess.Board) -> str | None:
    """Quick MCTS call to find Leela's current pick. ~1-2s per call.

    Used to record mcts_bestmove_uci per position so the summary can
    compute Branch-A-vs-Branch-B distribution against focal_uci.
    """
    from leela_chat.extractors.mcts import extract_mcts
    if ctx.lc0 is None:
        return None
    findings = extract_mcts(board, engine=ctx.lc0, nodes=1000)
    return findings.bestmove_uci


def _run_one(
    ctx: Any,
    position: dict,
    handler: str,
    mode: str,
    *,
    dry_run: bool = False,
) -> dict:
    """Run one (position, handler, mode) and return a JSONL-shaped dict.

    ``mode`` is one of ``"quick"`` or ``"deep"`` per the post-Chunk-7a
    naming (renamed from the old "default" / "detailed" strings).
    Threads through to ``_answer_one`` as the handler latency mode
    per ADR-0012.
    """
    from leela_chat.cli.__main__ import _answer_one

    fen = position["fen"]
    focal_uci = position["focal_uci"] if handler in ("why_this_move", "did_you_consider") else None

    out: dict[str, Any] = {
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "position_id": position["id"],
        "fen": fen,
        "category": position["category"],
        "stratum": position["stratum"],
        "themes": position["themes"],
        "rating": position["rating"],
        "handler": handler,
        "mode": mode,
        "focal_uci": focal_uci,
        "focal_source": position.get("focal_source"),
        "mcts_bestmove_uci": position.get("_mcts_bestmove_uci"),
        "focal_matches_mcts_pick": (
            focal_uci is not None
            and position.get("_mcts_bestmove_uci") is not None
            and focal_uci == position["_mcts_bestmove_uci"]
        ),
    }

    if dry_run:
        out.update({
            "source": "DRY_RUN",
            "validation_ok": None,
            "handler_s": 0.0,
            "translate_s": 0.0,
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "prose": "(dry-run; no LLM call)",
            "llm_attempt_texts": [],
            "findings_summary": {},
            "interpretive_claims": [],  # Chunk 7b.2.C — back-compat default.
        })
        return out

    try:
        question = (
            "What is Leela seeing here?" if handler == "what_is_leela_seeing"
            else f"Why this move ({focal_uci})?" if handler == "why_this_move"
            else f"Did you consider {focal_uci}?"
        )
        result, handler_s, translate_s, findings_dict = _answer_one(
            ctx, fen, question,
            handler_override=handler,
            move_uci=focal_uci,
            mode=mode,  # "quick" or "deep" per the matrix
        )
        total_in = sum(r.input_tokens for r in result.llm_responses)
        total_out = sum(r.output_tokens for r in result.llm_responses)
        out.update({
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
            "output_tokens": total_out,
            "prose": result.text,
            # Chunk 6.9 sub-item 1 — preserve LLM failed-attempt prose
            # so the aggregator's illegal_move classification can read
            # the actual hallucinated text (not just the templated
            # fallback's text, which doesn't contain the flagged spans).
            "llm_attempt_texts": [r.text for r in result.llm_responses],
            # Chunk 6.9 sub-item 1 — capture the trusted-moves arrays
            # the validator walks, so the diagnostic can distinguish
            # validator_gap (span IS in findings but walk missed it)
            # from real_hallucination (span NOT in findings). Without
            # this the validator-fix path in sub-item 2 is theatre.
            "findings_summary": _extract_findings_summary(findings_dict),
            # Chunk 7b.2.C — surface candidate interpretive-claim
            # sentences from the prose so the 2500+ rater at 7o can
            # scrutinize them explicitly (Option 1.5 from the 7b.1
            # retro). False positives are expected; the rater judges.
            "interpretive_claims": _extract_interpretive_claims(
                result.text, _extract_findings_summary(findings_dict),
            ),
        })
    except Exception as exc:
        logger.exception(f"smoke_battery: handler {handler}/{mode} on {position['id']} crashed")
        out.update({
            "source": "CRASH",
            "validation_ok": False,
            "validation_errors": [{"kind": "crash", "detail": str(exc), "span": ""}],
            "handler_s": 0.0,
            "translate_s": 0.0,
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "prose": f"(crash: {type(exc).__name__}: {exc})",
            "llm_attempt_texts": [],
            "findings_summary": {},
            "interpretive_claims": [],  # Chunk 7b.2.C — back-compat default.
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=None,
                        help="Cap matrix at first N positions (default: all 20)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Sampling seed for sample_stratified")
    parser.add_argument("--fixture-out", default="tests/eval/smoke_positions.yaml",
                        help="Where to write the sampled fixture YAML")
    parser.add_argument("--log-out", default=None,
                        help="JSONL output path (default: logs/smoke_battery_<ts>.jsonl)")
    parser.add_argument("--reuse-fixture", action="store_true",
                        help="Skip sampling; load fixture from --fixture-out if present")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip context build + LLM calls; just exercise the pipeline")
    args = parser.parse_args()

    fixture_path = Path(args.fixture_out)

    # === Step 1: build or load the fixture ===
    if args.reuse_fixture and fixture_path.exists():
        logger.info(f"loading existing fixture from {fixture_path}")
        fixture = _load_fixture(fixture_path)
    else:
        logger.info("sampling fixture from corpus")
        fixture = _build_fixture(seed=args.seed)
        _write_fixture(fixture, fixture_path)
        logger.info(f"  wrote {len(fixture)} positions to {fixture_path}")

    if args.positions:
        fixture = fixture[:args.positions]

    logger.info(f"running battery on {len(fixture)} positions ({len(fixture) * 5} LLM calls)")

    # === Step 2: build context (skipped in dry-run) ===
    ctx = None
    if not args.dry_run:
        from leela_chat.cli.__main__ import _build_context, _close_context
        from leela_chat.llm.client import LLMMode
        ctx = _build_context(mode=LLMMode.PROD, verbose=False)

    # === Step 3: per-position MCTS bestmove pre-pass ===
    if not args.dry_run:
        logger.info("computing MCTS bestmove per position (1000 nodes each)")
        for i, pos in enumerate(fixture, 1):
            try:
                board = chess.Board(pos["fen"])
                pos["_mcts_bestmove_uci"] = _mcts_bestmove(ctx, board)
            except Exception as exc:
                logger.warning(f"MCTS bestmove failed for {pos['id']}: {exc}")
                pos["_mcts_bestmove_uci"] = None
            if i % 5 == 0:
                logger.info(f"  bestmove {i}/{len(fixture)} done")

    # === Step 4: 5-call matrix per position ===
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = Path(args.log_out) if args.log_out else Path(f"logs/smoke_battery_{timestamp}.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"writing JSONL to {log_path}")

    # Post-Chunk-7a (ADR-0012) mode naming: "quick" / "deep" replace the
    # pre-7a "default" / "detailed" matrix labels. Aggregator just renders
    # whatever string is in the JSONL row, so this rename produces
    # different (clearer) labels in the markdown without breaking the
    # aggregator. did_you_consider stays "deep" (always-deep by construction
    # — a "quick" request is auto-upgraded inside _answer_one with INFO log).
    matrix = [
        ("what_is_leela_seeing", "quick"),
        ("what_is_leela_seeing", "deep"),
        ("why_this_move",        "quick"),
        ("why_this_move",        "deep"),
        ("did_you_consider",     "deep"),  # H3 is always deep
    ]

    # Chunk 6.8.1 Item 3B — consecutive-api-error circuit breaker.
    # NOTE: The 3-consecutive threshold catches sustained failure (credit
    # exhaustion, persistent outage). It does NOT catch an intermittent
    # ~50% failure pattern (e.g., 2 errors, 1 success, 2 errors, 1 success
    # repeating) because the counter resets on each success. That pattern
    # would waste ~50% of pod time + cost without triggering the breaker.
    # Acceptable for v1 since credit-exhaustion is binary; if a real
    # intermittent failure pattern appears, replace with a sliding-window
    # error-rate check.
    CIRCUIT_BREAKER_THRESHOLD = 3

    t_start = time.perf_counter()
    n_calls = 0
    consecutive_api_errors = 0
    last_error_detail = ""
    circuit_breaker_engaged = False
    with open(log_path, "w") as f:
        for i, pos in enumerate(fixture, 1):
            if circuit_breaker_engaged:
                break
            logger.info(f"position {i}/{len(fixture)}: {pos['id']} ({pos['category']}/{pos['stratum']})")
            for handler, mode in matrix:
                row = _run_one(ctx, pos, handler, mode, dry_run=args.dry_run)
                f.write(json.dumps(row) + "\n")
                f.flush()
                n_calls += 1

                # Update consecutive-api-error counter. Only `api_error` —
                # NOT validation failures (illegal_move, ungrounded_eval),
                # which are signal the battery exists to produce.
                row_errs = row.get("validation_errors") or []
                if (row.get("source") == "templated"
                        and any(e.get("kind") == "api_error" for e in row_errs)):
                    consecutive_api_errors += 1
                    last_error_detail = next(
                        (e.get("detail", "") for e in row_errs if e.get("kind") == "api_error"),
                        "",
                    )
                    if consecutive_api_errors >= CIRCUIT_BREAKER_THRESHOLD:
                        logger.error(
                            f"ABORTING: {CIRCUIT_BREAKER_THRESHOLD} consecutive Anthropic "
                            f"api_error responses. This is likely credit exhaustion or "
                            f"persistent API outage, not transient overload. Last error: "
                            f"{last_error_detail[:200]}. Resolve before re-running."
                        )
                        # Write a sentinel row so downstream consumers
                        # (the aggregator, future tooling) can detect the
                        # abort without parsing subprocess exit codes.
                        sentinel = {
                            "_circuit_breaker": True,
                            "consecutive_api_errors": consecutive_api_errors,
                            "last_error_kind": "api_error",
                            "last_error_detail": last_error_detail,
                            "ts": datetime.datetime.now(datetime.UTC).isoformat(),
                        }
                        f.write(json.dumps(sentinel) + "\n")
                        f.flush()
                        circuit_breaker_engaged = True
                        break
                else:
                    consecutive_api_errors = 0

    elapsed = time.perf_counter() - t_start
    if circuit_breaker_engaged:
        logger.error(
            f"battery aborted by circuit breaker after {n_calls} calls in {elapsed:.1f}s"
        )
        # Close context before exiting so resources release cleanly.
        if not args.dry_run and ctx is not None:
            from leela_chat.cli.__main__ import _close_context
            _close_context(ctx)
        sys.exit(3)
    logger.info(f"battery complete: {n_calls} calls in {elapsed:.1f}s")

    # === Step 5: close context ===
    if not args.dry_run and ctx is not None:
        from leela_chat.cli.__main__ import _close_context
        _close_context(ctx)

    logger.info(f"results: {log_path}")
    logger.info("next: write tests/eval/smoke_summaries/<YYYY-MM-DD>.md aggregate summary")


if __name__ == "__main__":
    main()
