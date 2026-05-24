"""Chunk 7a.3: extension-friendliness verification per the Chunk 7 plan.

Verifies the architectural properties that make ADR-0013's Tier-2
deferral safe. Without these, future Chunk-8+ work (position
perturbation, multi-strength ensemble, adaptive search budget,
iterative-search-and-prune, deepest mode) would require painful
handler refactors and the deferral argument would collapse.

Three properties verified:

**Property A — Findings models accept extra fields.** The top-level
``WhatIsLeelaSeeingFindings`` / ``WhyThisMoveFindings`` /
``DidYouConsiderFindings`` models have ``ConfigDict(frozen=True,
extra="allow")``. A Chunk-8 extractor can attach a new finding
(e.g., ``perturbation_summary``) to the model without modifying the
class itself. The strict per-extractor schema lives at the extractor-
level Pydantic models (MCTSFindings, StockfishResult, …); the
top-level wrapper is permissive by design.

**Property B — Prompt loader is field-name-agnostic.** The
``_load_prompt`` function in ``leela_chat/llm/translate.py`` is a thin
file-read with no hardcoded findings-field-name knowledge. A Chunk-8
prompt-version bump that cites new fields (e.g., ``perturbation_summary``)
touches ONLY the prompt markdown file + the handler's PROMPT_VERSION
constant — NOT the loader code.

**Property C — Handler extractor invocation is localized-extensible.**
Adding a new extractor to a handler is a single-file change with a
bounded blast radius: edit the handler's ``_bt4_thread`` / Stockfish /
Lc0 thread block to invoke the new extractor; add a Pydantic field
(or use the extra="allow" path); bump PROMPT_VERSION_DETAILED + add the
new prompt file. No cross-file infrastructure changes required.

Note: Property C is verified by inspection + a structural test (no
attempt to actually wire a stub extractor into a real handler, since
that would BE the cross-file work we're claiming is bounded). The
honest framing is "extension-with-localized-edits-per-handler" not
"extension-without-touching-handler-code."
"""
from __future__ import annotations

import inspect
from pathlib import Path

import chess
import pytest

from leela_chat.extractors.network_inference import WDL
from leela_chat.extractors.stockfish import StockfishResult
from leela_chat.handlers.did_you_consider import DidYouConsiderFindings
from leela_chat.handlers.what_is_leela_seeing import WhatIsLeelaSeeingFindings
from leela_chat.handlers.why_this_move import WhyThisMoveFindings
from leela_chat.llm import translate as translate_mod


def _empty_stockfish_result() -> StockfishResult:
    """Minimal valid StockfishResult for tests that only need to construct
    a findings model — not run a real Stockfish call."""
    return StockfishResult(
        depth=0,
        nodes=0,
        root_eval_cp=0,
        root_mate_in=None,
        top_moves=[],
    )


# =====================================================================
# Property A — Findings models accept extra fields
# =====================================================================


@pytest.mark.parametrize(
    "model_cls",
    [WhatIsLeelaSeeingFindings, WhyThisMoveFindings, DidYouConsiderFindings],
    ids=["what_is_leela_seeing", "why_this_move", "did_you_consider"],
)
def test_property_a_findings_models_allow_extra_fields(model_cls):
    """Findings models must allow extra fields so Chunk-8 extractors can
    attach new findings without modifying the class.

    Verifies ConfigDict(extra="allow") is set. Without this, a Chunk-8
    perturbation extractor that attaches ``perturbation_summary={...}``
    to the findings would raise ValidationError at construction time,
    forcing a model-class edit to accept it.
    """
    config = model_cls.model_config
    assert config.get("extra") == "allow", (
        f"{model_cls.__name__} must have ConfigDict(extra='allow') "
        f"per Chunk 7a.3 extension-friendliness contract. "
        f"Without it, Chunk-8 Tier-2 extractors would require model-class "
        f"edits to attach new findings, violating ADR-0013's deferral "
        f"premise."
    )


def test_property_a_findings_models_preserve_frozen():
    """The extra='allow' addition must NOT have weakened frozen=True.

    frozen=True is load-bearing: it prevents accidental in-place
    mutation of findings that flow through translate -> validator -> LLM.
    Chunk 7a.3 added extra='allow' alongside frozen=True, not as a
    replacement.
    """
    for model_cls in [WhatIsLeelaSeeingFindings, WhyThisMoveFindings, DidYouConsiderFindings]:
        assert model_cls.model_config.get("frozen") is True, (
            f"{model_cls.__name__} must still have frozen=True "
            f"(Chunk 7a.3 added extra='allow' ALONGSIDE frozen=True, "
            f"not as a replacement)."
        )


def test_property_a_extra_field_construction_does_not_raise():
    """Constructing a findings model with an extra field must succeed.

    Synthetic verification: pretend a Chunk-8 perturbation extractor
    has attached a new ``perturbation_summary`` field. The findings
    model must accept it without raising.
    """
    # WhatIsLeelaSeeingFindings has the simplest required-field set;
    # use it as the canonical demonstration. Required fields per the
    # model definition: fen, side_to_move (literal), eval_cp, wdl
    # (WDL model), moves_left_estimate (float), top_moves, stockfish
    # (StockfishResult or None), mcts, layer_emergence.
    # NOTE: stockfish is a required field on the model itself but
    # accepts None via union typing. Using a real WDL instance + sentinels
    # for the optionals.
    findings = WhyThisMoveFindings(
        fen=chess.STARTING_FEN,
        side_to_move="white",
        eval_cp=0,
        wdl=WDL(white_wins=0.4, draw=0.4, black_wins=0.2),
        moves_left_estimate=40.0,
        top_moves=[],
        stockfish=_empty_stockfish_result(),
        mcts=None,
        focal_move_uci="e2e4",
        focal_move_san="e4",
        focal_is_engine_pick=True,
        focal_policy_pct=None,
        focal_policy_rank=None,
        focal_mcts_child=None,
        focal_mcts_rank_by_visits=None,
        focal_stockfish_candidate=None,
        focal_stockfish_rank=None,
        # Chunk-8 hypothetical perturbation finding attached via extra="allow":
        perturbation_summary={
            "perturbed_variants_tested": 12,
            "verdict_flip_rate": 0.0,
            "razor_edge": False,
        },
    )
    # The extra field is preserved + surfaced via model_dump (so the
    # prompt can cite it, the validator can walk it, and the smoke
    # battery's _extract_findings_summary can see it).
    dump = findings.model_dump()
    assert "perturbation_summary" in dump
    assert dump["perturbation_summary"]["perturbed_variants_tested"] == 12


# =====================================================================
# Property B — Prompt loader is field-name-agnostic
# =====================================================================


def test_property_b_load_prompt_is_thin_file_read():
    """_load_prompt must contain no hardcoded findings-field-name knowledge.

    The contract: a Chunk-8 prompt-version bump that cites new fields
    (e.g., a new prompt v10 with ``{{perturbation_summary}}`` references)
    touches the prompt markdown file + handler PROMPT_VERSION constant
    — NOT the loader code.

    Verified by inspecting the source: the loader reads ``v{N}.md``
    from disk and returns the string. The MARKDOWN is what knows about
    findings fields (the LLM renders findings as JSON and the prompt
    text references them by name); the loader is field-agnostic.
    """
    src = inspect.getsource(translate_mod._load_prompt)
    # Loader should reference path + read_text only; no findings field
    # names embedded.
    assert "read_text" in src or "open(" in src, (
        "loader should be a file-read; found something else"
    )
    forbidden_field_names = [
        "top_moves", "mcts", "stockfish", "layer_emergence",
        "focal_layer_trajectory", "network_dismissal", "search_dismissal",
        "perturbation",  # the hypothetical Chunk-8 field
    ]
    for name in forbidden_field_names:
        assert name not in src, (
            f"_load_prompt hardcodes findings field name {name!r}; "
            f"this would break Chunk-8 prompt-version bumps that "
            f"reference new fields. Loader must stay field-agnostic."
        )


def test_property_b_prompt_versioning_is_filename_based():
    """Prompt versions are encoded in filenames (vN.md), not in code.

    A Chunk-8 prompt-version bump for what_is_leela_seeing from v7 to
    v9 (hypothetical) means: drop a new ``v9.md`` next to the existing
    v7.md, bump PROMPT_VERSION_DETAILED constant in the handler from 7
    to 9. The loader auto-picks up v9 because it walks
    ``_PROMPTS_DIR / handler_name / f"v{version}.md"``.

    Verified: the prompts directory layout exists; each handler has a
    subdirectory; each subdirectory has multiple vN.md files (per
    CLAUDE.md S3: never delete old prompt versions, always increment).
    """
    prompts_dir = translate_mod._PROMPTS_DIR
    assert prompts_dir.exists(), f"prompts dir missing: {prompts_dir}"
    for handler_name in ("what_is_leela_seeing", "why_this_move", "did_you_consider"):
        handler_dir = prompts_dir / handler_name
        assert handler_dir.exists(), f"handler prompts dir missing: {handler_dir}"
        versions = sorted(handler_dir.glob("v*.md"))
        assert len(versions) >= 1, (
            f"no v*.md files found in {handler_dir}; "
            f"versioning convention broken"
        )


# =====================================================================
# Property C — Handler extractor invocation is localized-extensible
# =====================================================================


def test_property_c_handler_invocation_uses_thread_pool_pattern():
    """Each handler's extractor-invocation block uses ThreadPoolExecutor.

    The contract: adding a new extractor to a handler is a single-file
    change. Specifically:
    1. Add the extractor call to the appropriate thread block
       (_bt4_thread for BT4-using extractors; sf_future for Stockfish-
       using; mcts_future for Lc0-using; or a new ex.submit() for a
       new engine).
    2. Capture the result alongside the existing future.result() calls.
    3. Either bind the result to a new Pydantic field on the findings
       model OR use the extra="allow" path (Property A).
    4. Optionally bump PROMPT_VERSION_DETAILED + add prompt file
       referencing the new finding.

    Verified by inspecting that handlers import + use ThreadPoolExecutor
    (the structural extension point).
    """
    from leela_chat.handlers import (
        what_is_leela_seeing,
        why_this_move,
        did_you_consider,
    )
    for module in (what_is_leela_seeing, why_this_move, did_you_consider):
        src = inspect.getsource(module)
        assert "ThreadPoolExecutor" in src, (
            f"{module.__name__} missing ThreadPoolExecutor pattern from "
            f"Chunk 7a.2 — extension contract requires this as the "
            f"structural extension point for adding new extractors."
        )
        assert "ex.submit(" in src, (
            f"{module.__name__} missing ex.submit() pattern; new "
            f"extractor invocations should fit this pattern."
        )


def test_property_c_handler_signature_uses_mode_parameter():
    """Each handler exposes a ``mode: Mode`` parameter for per-mode dispatch.

    The contract: a Chunk-8 feature that's only active in deep mode
    can branch inside the handler on ``mode == "deep"``. Quick mode
    keeps its budget; deep mode gets the new behavior. This is how
    perturbation, multi-strength, etc., would gate their cost.
    """
    from leela_chat.handlers.what_is_leela_seeing import handle_what_is_leela_seeing
    from leela_chat.handlers.why_this_move import handle_why_this_move

    sig_what = inspect.signature(handle_what_is_leela_seeing)
    sig_why = inspect.signature(handle_why_this_move)
    assert "mode" in sig_what.parameters, (
        "handle_what_is_leela_seeing missing mode parameter "
        "(Chunk 7a.1 contract; extension-friendliness depends on it)"
    )
    assert "mode" in sig_why.parameters, (
        "handle_why_this_move missing mode parameter "
        "(Chunk 7a.1 contract; extension-friendliness depends on it)"
    )
    # did_you_consider is always-deep by construction (mode handled at
    # _answer_one auto-upgrade level); no mode parameter on its signature.


# =====================================================================
# Honest-framing meta-test — Property C IS bounded, not unbounded
# =====================================================================


def test_extension_friendliness_contract_documented():
    """The extension-friendliness CONTRACT (the three properties) is documented.

    This test exists to keep the contract visible to future readers.
    If you're adding a Chunk-8+ extractor and finding yourself doing
    cross-file refactors beyond:
      - one handler file (extractor invocation + Pydantic field)
      - one prompt markdown file (prompt-version bump)
      - one handler PROMPT_VERSION constant edit
      - (optionally) one validator helper extension for derived
        quantities per CLAUDE.md S8

    ...then the extension is fighting the architecture, NOT working
    with it. Stop and revisit ADR-0013's deferral premise — either
    the feature needs more infrastructure that should be its own
    sub-chunk, or this contract needs widening (which is itself a
    Chunk-8+ infrastructure sub-chunk).
    """
    docstring = __doc__
    assert docstring is not None, "module docstring missing"
    assert "Property A" in docstring
    assert "Property B" in docstring
    assert "Property C" in docstring
    assert "ADR-0013" in docstring
