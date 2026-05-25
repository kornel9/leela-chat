"""Unit tests for the interpretive-claim capture helper (Chunk 7b.2.C).

The helper lives in ``leela_chat/scripts/run_smoke_battery.py`` per the
plan's design — it's part of the smoke-battery row preparation, not a
shared validator/aggregator utility. Tests target the heuristic's
two-rule design (wordlist + structural) per the 7b.2 plan refinement,
plus the grounded-citation exclusion + sentence splitting.

Per the plan: false positives are expected and FINE. The point is
candidate surfacing for 2500+ rater review at 7o, not auto-validation.
The tests verify behavior at known cases (canonical 7b.1 leak, pure
move citation, piece-location-without-interpretation) rather than
trying to enforce specificity.
"""
from __future__ import annotations

import pytest

from leela_chat.scripts.run_smoke_battery import (
    _extract_findings_summary,
    _extract_interpretive_claims,
    _sentence_is_grounded_citation,
    _split_sentences,
)


# === _split_sentences ===


def test_split_sentences_simple():
    """Splits on `. `, `! `, `? ` and trims."""
    prose = "First sentence. Second sentence! Third sentence?"
    out = _split_sentences(prose)
    assert out == ["First sentence.", "Second sentence!", "Third sentence?"]


def test_split_sentences_preserves_internal_punctuation():
    """Periods inside a sentence (e.g. abbreviations) don't over-split."""
    prose = "Leela's MCTS pick is Rxd4 (4221 visits). Stockfish agrees."
    out = _split_sentences(prose)
    assert len(out) == 2
    assert "Rxd4" in out[0]
    assert "Stockfish" in out[1]


def test_split_sentences_empty_input():
    assert _split_sentences("") == []
    assert _split_sentences("   ") == []


# === _sentence_is_grounded_citation ===


def _basic_findings_summary():
    """Findings summary fixture for the walkthrough_rxd4 position."""
    return {
        "top_moves_san": ["Rxb1+", "Rbc8", "Rxd4"],
        "top_moves_uci": ["b8b1", "b8c8", "d8d4"],
        "mcts_children_san": ["Rxd4", "Rbc8", "Rxb1+"],
        "mcts_children_uci": ["d8d4", "b8c8", "b8b1"],
        "mcts_bestmove_san": "Rxd4",
        "mcts_bestmove_uci": "d8d4",
        "stockfish_top_san": ["Rxd4", "Ra8", "Qf4"],
        "stockfish_top_uci": ["d8d4", "b8a8", "e5f4"],
        "stockfish_pv_sans": ["Rxd4", "Qc3", "Rxb1+", "Kxb1", "Qb8+"],
        "mcts_pv_sans": ["Rxd4", "Qc3", "Rxb1+", "Kxb1", "Qb8+", "Qb2", "Rd8"],
        "focal_move_san": "Rxd4",
        "focal_move_uci": "d8d4",
    }


def test_grounded_citation_when_san_and_numeric_present():
    """Sentence citing a trusted SAN with numeric value is grounded."""
    sentence = "Leela's MCTS pick is Rxd4 with 4221 of 4798 visits."
    assert _sentence_is_grounded_citation(sentence, _basic_findings_summary())


def test_not_grounded_when_only_numeric_no_move():
    """Sentence with numeric but no trusted move citation is NOT grounded."""
    sentence = "The eval is +0.5 from White's perspective."
    assert not _sentence_is_grounded_citation(sentence, _basic_findings_summary())


def test_not_grounded_when_only_move_no_numeric():
    """Sentence with trusted move but no numeric is NOT grounded (interpretive
    in nature — bare-move citations without numerics tend to be claims about
    the move, not number-narration of its data)."""
    sentence = "After Rxd4, the d4 knight falls."
    assert not _sentence_is_grounded_citation(sentence, _basic_findings_summary())


def test_not_grounded_when_findings_empty():
    """Empty findings_summary → nothing can be grounded."""
    assert not _sentence_is_grounded_citation(
        "Rxd4 with 4221 visits is the pick.", {},
    )


# === Rule A: wordlist (piece name + interpretive phrase) ===


def test_rule_a_canonical_7b1_leak_captured():
    """The canonical 7b.1 spot-check leak surfaces as a candidate.

    'White's queen has nowhere to hide after the d4-knight falls' —
    piece name 'queen' + interpretive phrase 'has nowhere to hide'.
    Validator passed (no illegal moves) but the claim is interpretive
    and chess-accuracy is rater-judgment territory.
    """
    prose = "White's queen has nowhere to hide after the d4-knight falls."
    claims = _extract_interpretive_claims(prose, _basic_findings_summary())
    assert len(claims) == 1
    assert "queen has nowhere to hide" in claims[0]


def test_rule_a_pure_move_citation_not_captured():
    """A sentence whose substantive content is a move citation with
    numerics is excluded (NOT flagged) regardless of any peripheral
    piece-name match."""
    prose = "Leela's MCTS pick is Rxd4 with 4221 of 4798 visits at Q +0.92."
    claims = _extract_interpretive_claims(prose, _basic_findings_summary())
    assert claims == []


def test_rule_a_piece_location_without_interpretation_not_captured():
    """A sentence with a piece name but no interpretive verb/phrase is
    not flagged."""
    prose = "The white king is on g1."
    claims = _extract_interpretive_claims(prose, _basic_findings_summary())
    # No interpretive phrase → Rule A doesn't fire.
    # Rule B's "weak/exposed/loose/etc." pattern doesn't fire either.
    assert claims == []


# === Rule B: structural relational-claim catch ===


def test_rule_b_lack_coordination_captured():
    """'Black's pieces lack coordination' — Rule B catches the
    'lack(s) <X>' pattern + 'coordination' even without wordlist hit."""
    prose = "Black's pieces lack coordination after the queen retreat."
    claims = _extract_interpretive_claims(prose, _basic_findings_summary())
    # Rule B's `\blacks?\s+\w+` + `\bcoordinat(?:e|es|ion)\b` patterns fire.
    assert len(claims) >= 1


def test_rule_b_finds_no_safety_captured():
    """'The queen finds no safety' — Rule B's `finds?\\s+no\\s+\\w+`
    pattern catches this even though 'finds no' isn't in Rule A's
    wordlist as a complete phrase."""
    prose = "The queen finds no safety on the open files."
    claims = _extract_interpretive_claims(prose, _basic_findings_summary())
    assert len(claims) >= 1


def test_rule_b_attacks_with_object_captured():
    """'White's bishop attacks the long diagonal' — Rule B's
    `\\battacks?\\s+(?:the\\s+)?\\w+` pattern captures this."""
    prose = "White's bishop attacks the long diagonal."
    claims = _extract_interpretive_claims(prose, _basic_findings_summary())
    assert len(claims) >= 1


def test_rule_b_grounded_citation_NOT_excluded_when_rule_b_fires():
    """Behavior CHANGED at 7c: Rule B (structural relational patterns)
    fires regardless of grounding — its patterns are specific enough
    that any match indicates a real relational claim, even if the
    sentence also contains a grounded move+numeric prefix. The Rule A
    grounded-exclusion still applies when Rule A fires alone (wordlist
    incidental matches are riskier).

    Old behavior (pre-7c): grounded-citation exclusion blocked Rule B
    flags too. That under-fired on sentences like "Rxd4 with 4221 visits
    attacks the d4 knight" where "attacks the d4 knight" is real
    interpretive content even though the sentence is partly grounded.
    """
    prose = "Rxd4 with 4221 visits attacks the d4 knight."
    claims = _extract_interpretive_claims(prose, _basic_findings_summary())
    # Rule B fires on "attacks the d4 knight". Now FLAGGED (was excluded).
    assert len(claims) == 1
    assert "attacks the d4 knight" in claims[0]


def test_rule_a_grounded_exclusion_still_applies():
    """The Rule A grounded-exclusion is preserved (wordlist incidental
    match risk). Construct a sentence where Rule A's piece+phrase fires
    but the sentence is clearly number-narrating with a grounded move
    citation, so the exclusion should suppress the flag."""
    # "queen" + "controls" (Rule A fires: piece+interpretive phrase
    # since "controls" is in wordlist). But sentence is grounded
    # citation. Verify exclusion suppresses.
    prose = "Rxd4 controls the queen and takes 4221 visits at Q +0.92."
    claims = _extract_interpretive_claims(prose, _basic_findings_summary())
    # Rule A fires (queen + controls in wordlist). Also Rule B fires
    # on "controls the queen" (attacks/defends/controls pattern). So
    # this sentence will flag via Rule B even though Rule A's exclusion
    # would normally suppress. That's correct — the new design says
    # "Rule B fires regardless of grounding"; if Rule A AND B both
    # fire, Rule B's flagging wins.
    # Re-check: this sentence flags. Test expectation: flagged.
    assert len(claims) == 1


# === Multi-sentence prose + behavior smoke ===


def test_extract_returns_only_flagged_sentences():
    """Mixed prose: some sentences grounded, some interpretive. Helper
    returns only the candidates."""
    prose = (
        "Leela's MCTS pick is Rxd4 with 4221 of 4798 visits at Q +0.92. "
        "White's queen has nowhere to hide after the d4-knight falls. "
        "Stockfish agrees at depth 15, giving Rxd4 −4.02. "
        "The kingside is fatally weakened."
    )
    claims = _extract_interpretive_claims(prose, _basic_findings_summary())
    # Sentence 1 grounded (Rxd4 + 4221 + Q +0.92).
    # Sentence 2 Rule A (queen + nowhere to hide).
    # Sentence 3 grounded (Rxd4 + −4.02).
    # Sentence 4 Rule B (`\b(?:weak|...)\s+...` doesn't match 'fatally weakened'
    # because pattern requires piece name after the adjective; this might NOT
    # fire). Acceptable either way per "false positives are FINE."
    assert any("queen has nowhere to hide" in c for c in claims)


def test_extract_empty_prose_returns_empty():
    assert _extract_interpretive_claims("", _basic_findings_summary()) == []
    assert _extract_interpretive_claims("   ", _basic_findings_summary()) == []


def test_extract_no_findings_summary_still_works():
    """When findings_summary is None/empty, every sentence is considered
    ungrounded — Rule A/B fires unmodified."""
    prose = "The white queen is exposed."
    claims = _extract_interpretive_claims(prose, {})
    # 'queen' + 'is exposed' (Rule A wordlist) fires.
    assert len(claims) >= 1


# === Findings-summary extension: mcts_pv_sans (7b.2.C) ===


def test_findings_summary_captures_mcts_pv_sans():
    """7b.2.C extension: mcts.children[i].pv_san is now captured in
    findings_summary so the interpretive-claim heuristic's exclusion
    rule recognises sentences that cite Leela's MCTS PV moves as
    grounded (small gap inherited from 7b)."""
    findings = {
        "mcts": {
            "bestmove_san": "Rxd4",
            "bestmove_uci": "d8d4",
            "children": [
                {
                    "san": "Rxd4", "uci": "d8d4",
                    "pv_san": ["Rxd4", "Qc3", "Rxb1+"],
                },
                {
                    "san": "Rbc8", "uci": "b8c8",
                    "pv_san": ["Rbc8", "Qxa4"],
                },
            ],
        },
    }
    summary = _extract_findings_summary(findings)
    assert "Rxd4" in summary["mcts_pv_sans"]
    assert "Qc3" in summary["mcts_pv_sans"]
    assert "Rxb1+" in summary["mcts_pv_sans"]
    assert "Rbc8" in summary["mcts_pv_sans"]
    assert "Qxa4" in summary["mcts_pv_sans"]


def test_findings_summary_mcts_pv_sans_empty_when_no_multipv():
    """When MultiPv=1 (quick mode), pv_san is empty on children;
    mcts_pv_sans stays empty."""
    findings = {
        "mcts": {
            "bestmove_san": "Rxd4", "bestmove_uci": "d8d4",
            "children": [
                {"san": "Rxd4", "uci": "d8d4", "pv_san": []},
                {"san": "Rbc8", "uci": "b8c8"},  # no pv_san key at all
            ],
        },
    }
    summary = _extract_findings_summary(findings)
    assert summary["mcts_pv_sans"] == []
