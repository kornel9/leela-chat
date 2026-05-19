"""Unit tests for the trivial Phase-1 classifier.

The classifier is a placeholder — Phase 1 has only one handler, so any
question routes to it. These tests pin the contract so a future change
(say, real routing) breaks them and demands explicit re-thinking.
"""
from __future__ import annotations

import pytest

from leela_chat.classifier import classify


def test_classifies_to_what_is_leela_seeing():
    assert classify("What is Leela seeing here?") == "what_is_leela_seeing"
    assert classify("Why did White play Nf3?") == "what_is_leela_seeing"
    assert classify("Did you consider Bxh7+?") == "what_is_leela_seeing"
    assert classify("") == "what_is_leela_seeing"


def test_current_handlers_check_passes_when_handler_present():
    assert classify(
        "anything",
        current_handlers=["what_is_leela_seeing", "why_this_move"],
    ) == "what_is_leela_seeing"


def test_current_handlers_check_raises_when_target_absent():
    with pytest.raises(ValueError, match="not in current_handlers"):
        classify("anything", current_handlers=["only_other_handler"])
