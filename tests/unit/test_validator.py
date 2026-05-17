"""Unit tests for the validator (Phase 1 Chunk 3).

Synthetic data only — no Lc0, no LLM, no engine calls. Exercises:

- SAN/UCI move extraction and legality checking
- Number/percentage extraction and JSON-grounding (with unit conversions)
- Combined error reporting
- Result model shape (frozen Pydantic)
"""
from __future__ import annotations

from leela_chat.validator import ValidationResult, validate_response


# Starting position — most legality reasoning is easiest from here.
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# After 1.e4 e5 2.Nf3 Nc6 (i.e. Black has just played Nc6). Opens up f1 bishop
# diagonal so Bc4 becomes legal.
ITALIAN_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"

# After 1.e4 d5 — White can play exd5.
AFTER_E4_D5 = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"

# White pawn on a7, Black king tucked at d4 so promotion doesn't give check.
PROMOTION_FEN = "8/P7/8/3k4/8/8/8/7K w - - 0 1"


SAMPLE_FINDINGS = {
    "fen": ITALIAN_FEN,
    "side_to_move": "white",
    "eval_cp": 42,
    "wdl": {"white_wins": 0.45, "draw": 0.40, "black_wins": 0.15},
    "moves_left_estimate": 35.2,
    "top_moves": [
        {"uci": "f1c4", "san": "Bc4", "policy_pct": 28.5},
        {"uci": "d2d4", "san": "d4", "policy_pct": 22.0},
    ],
}


# === Legality ===============================================================


def test_no_moves_mentioned_passes():
    r = validate_response("Quiet position with no concrete play.", fen=START_FEN, findings={})
    assert r.ok and r.errors == []


def test_legal_san_passes():
    r = validate_response("White plays Nf3 here.", fen=START_FEN, findings={})
    assert r.ok, r.errors


def test_illegal_san_flagged():
    # Bc4 is blocked by the e2 pawn from the starting position.
    r = validate_response("White plays Bc4.", fen=START_FEN, findings={})
    assert not r.ok
    assert any(e.kind == "illegal_move" and "Bc4" in e.span for e in r.errors)


def test_legal_uci_passes():
    r = validate_response("Best move is e2e4.", fen=START_FEN, findings={})
    assert r.ok, r.errors


def test_illegal_uci_flagged():
    # No piece on a3 at start.
    r = validate_response("Best move is a3a4.", fen=START_FEN, findings={})
    assert not r.ok
    assert any(e.kind == "illegal_move" and "a3a4" in e.span for e in r.errors)


def test_bare_square_reference_not_flagged():
    # Bare squares like "e5" are NOT extracted as moves in v1.
    r = validate_response("Black controls the e5 square.", fen=START_FEN, findings={})
    assert r.ok, r.errors


def test_pawn_capture_legal():
    r = validate_response("White plays exd5.", fen=AFTER_E4_D5, findings={})
    assert r.ok, r.errors


def test_pawn_capture_illegal_flagged():
    # No pawn on f-file to capture g5 from starting position.
    r = validate_response("White plays fxg5.", fen=START_FEN, findings={})
    assert not r.ok
    assert any(e.kind == "illegal_move" and "fxg5" in e.span for e in r.errors)


def test_promotion_legal():
    r = validate_response("White promotes with a8=Q.", fen=PROMOTION_FEN, findings={})
    assert r.ok, r.errors


def test_castling_attempt_in_starting_position_flagged():
    # O-O isn't legal from starting position (knights and bishops in the way).
    r = validate_response("White castles with O-O.", fen=START_FEN, findings={})
    assert not r.ok
    assert any(e.kind == "illegal_move" and e.span.startswith("O-O") for e in r.errors)


def test_repeated_illegal_move_deduplicated():
    r = validate_response("Black plays Bc4, definitely Bc4.", fen=START_FEN, findings={})
    illegal = [e for e in r.errors if e.kind == "illegal_move"]
    assert len(illegal) == 1


def test_multiple_legal_moves_all_pass():
    r = validate_response("Considered Nf3 and Nc3.", fen=START_FEN, findings={})
    assert r.ok, r.errors


# === Grounding ==============================================================


def test_cp_eval_cited_as_cp_passes():
    r = validate_response("Eval is +42 centipawns.", fen=ITALIAN_FEN, findings=SAMPLE_FINDINGS)
    assert r.ok, r.errors


def test_cp_eval_cited_as_pawn_units_passes():
    r = validate_response("Eval is +0.42 for White.", fen=ITALIAN_FEN, findings=SAMPLE_FINDINGS)
    assert r.ok, r.errors


def test_ungrounded_signed_eval_flagged():
    r = validate_response("Eval is +5.5 for White.", fen=ITALIAN_FEN, findings=SAMPLE_FINDINGS)
    assert not r.ok
    assert any(e.kind == "ungrounded_eval" and "+5.5" in e.span for e in r.errors)


def test_percentage_matches_policy_pct():
    r = validate_response("Top move policy is 28%.", fen=ITALIAN_FEN, findings=SAMPLE_FINDINGS)
    assert r.ok, r.errors


def test_percentage_matches_wdl_probability():
    r = validate_response("White wins 45% of the time.", fen=ITALIAN_FEN, findings=SAMPLE_FINDINGS)
    assert r.ok, r.errors


def test_ungrounded_percentage_flagged():
    r = validate_response("White wins 80% of the time.", fen=ITALIAN_FEN, findings=SAMPLE_FINDINGS)
    assert not r.ok
    assert any(e.kind == "ungrounded_eval" and "80%" in e.span for e in r.errors)


def test_plain_integer_not_extracted():
    r = validate_response(
        "This is roughly 2200 elo strength.", fen=ITALIAN_FEN, findings=SAMPLE_FINDINGS,
    )
    assert r.ok, r.errors


def test_negative_eval_matches_negative_finding():
    findings = {"eval_cp": -150}
    r = validate_response("Black is better, eval -1.5.", fen=ITALIAN_FEN, findings=findings)
    assert r.ok, r.errors


def test_empty_findings_signed_eval_flagged():
    r = validate_response("Eval is +0.42.", fen=START_FEN, findings={})
    assert not r.ok
    assert any(e.kind == "ungrounded_eval" for e in r.errors)


def test_findings_with_nested_lists_walked():
    findings = {"top_moves": [{"policy_pct": 28.5}, {"policy_pct": 22.0}]}
    r = validate_response("Top move is 28% likely.", fen=START_FEN, findings=findings)
    assert r.ok, r.errors


def test_repeated_number_deduplicated():
    r = validate_response("Eval is +5.5, again +5.5.", fen=ITALIAN_FEN, findings={"eval_cp": 42})
    ungrounded = [e for e in r.errors if e.kind == "ungrounded_eval"]
    assert len(ungrounded) == 1


def test_booleans_in_findings_do_not_ground_numbers():
    # If we let bools pollute the numeric set, "+1" or "+0" would spuriously
    # ground. With the bool guard, they should be flagged.
    findings = {"flag": True, "other": False}
    r = validate_response("Eval is +1.", fen=START_FEN, findings=findings)
    assert not r.ok
    assert any(e.kind == "ungrounded_eval" for e in r.errors)


# === Combined / structural ==================================================


def test_clean_response_passes():
    r = validate_response(
        "White's eval is +0.42 with 45% chance of winning. Top move: Bc4.",
        fen=ITALIAN_FEN, findings=SAMPLE_FINDINGS,
    )
    assert r.ok, r.errors


def test_mixed_error_kinds_all_reported():
    # Bc4 illegal at START_FEN; +5.5 not in findings.
    r = validate_response(
        "White plays Bc4, eval is +5.5.",
        fen=START_FEN,
        findings={"eval_cp": 42},
    )
    kinds = {e.kind for e in r.errors}
    assert "illegal_move" in kinds
    assert "ungrounded_eval" in kinds


def test_result_is_frozen_pydantic():
    r = validate_response("Quiet text.", fen=START_FEN, findings={})
    assert isinstance(r, ValidationResult)
    dumped = r.model_dump()
    assert dumped == {"ok": True, "errors": []}
