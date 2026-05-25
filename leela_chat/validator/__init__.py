"""Validator: gate-keeps every LLM response before it reaches the user.

Per CLAUDE.md §2 Rule 2:

> The output validator runs on every LLM response. It extracts every move-like
> string and verifies legality via python-chess. It extracts every eval number
> and checks it appears in the JSON within tolerance. It extracts every feature
> label and checks it's in the firing list. If validation fails:
> 1. Retry with stricter prompt
> 2. If retry fails, fall back to templated response built from the JSON directly
> 3. Log the failure
> 4. Never show an unvalidated response to the user

Per ADR-0005 G1.3 the validator is **strict mechanical only** — legality and
JSON-grounding. Semantic plausibility ("does this argument actually hold?")
lives in the eval grader's LLM-as-judge, not on the live path.

Phase 1 scope: legality + numeric grounding. Feature-firing checks
(``unknown_feature``) come online in Phase 4 when SAEs land — the validator
function will grow a ``firings`` kwarg then.

Known v1 limitations (deliberate — tightened as the system matures):

- Bare pawn-push SAN ("e4", "d5") is NOT extracted. The disambiguation cost
  versus square references in prose ("controls e5", "the d-file") is too high.
  The prompt template instructs the LLM to use full SAN (with disambig, capture,
  or promotion suffix) for pawn moves that could be confused with squares.
- Plain unsigned integers ("2200 elo") are NOT grounding-checked. Only signed
  decimals ("+0.42", "-1.5") and percentages ("28%") trigger grounding so we
  avoid flagging benign references to ratings, piece counts, move numbers, etc.
- Figurine ("♘f3"), lowercase ("nf3"), and non-English piece-letter
  notations are NOT recognised; they pass silently. The prompt enforces
  standard English SAN.
"""
from __future__ import annotations

import re
from typing import Any

import chess
from loguru import logger

from leela_chat.validator.types import ErrorKind, ValidationError, ValidationResult

__all__ = [
    "ErrorKind",
    "ValidationError",
    "ValidationResult",
    "validate_response",
]


_SAN_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:"
    r"O-O-O[+#]?"                                                # queenside castle
    r"|O-O[+#]?"                                                  # kingside castle
    r"|[KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?"          # piece move
    r"|[a-h]x[a-h][1-8](?:=[QRBN])?[+#]?"                         # pawn capture
    r"|[a-h][1-8]=[QRBN][+#]?"                                    # pawn promotion (no capture)
    r")"
    r"(?![A-Za-z0-9])"
)

_UCI_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"([a-h][1-8][a-h][1-8][qrbn]?)"
    r"(?![A-Za-z0-9])"
)

# Signed decimal: explicit + or - required so we don't sweep up every integer in
# prose like "2200 elo" or "5 pieces".
_SIGNED_NUMBER_RE = re.compile(r"(?<![\w.])([+\-]\d+(?:\.\d+)?)(?!\w)")
# Percentage: trailing % required.
_PERCENT_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*%")


def validate_response(
    response_text: str,
    *,
    fen: str,
    findings: dict[str, Any],
) -> ValidationResult:
    """Validate an LLM response for legality + JSON-grounding.

    Pure function — safe to call repeatedly on the same inputs.

    Args:
        response_text: The raw prose from the LLM.
        fen: The position the LLM was answering about. Used to legality-check
            any SAN/UCI moves mentioned.
        findings: The findings JSON the handler passed to the translation
            layer (i.e. the dict serialization of the handler's Pydantic
            findings model). Used to ground-check any numbers in the response.

    Returns:
        Frozen ``ValidationResult`` — ``ok=True`` with empty ``errors`` on pass,
        else ``ok=False`` with one or more ``ValidationError`` entries.
    """
    errors: list[ValidationError] = []
    errors.extend(_check_legality(response_text, fen, findings))
    errors.extend(_check_grounding(response_text, findings))
    errors.extend(_check_trough_completeness(response_text, findings))
    result = ValidationResult(ok=not errors, errors=errors)
    if not result.ok:
        logger.warning(
            f"validator: {len(errors)} error(s) "
            f"(first: {errors[0].kind} '{errors[0].span}')"
        )
    return result


# --- Legality ---------------------------------------------------------------


def _check_legality(
    response_text: str, fen: str, findings: dict[str, Any],
) -> list[ValidationError]:
    """Flag any SAN/UCI in ``response_text`` that is neither legal on ``fen``
    nor a trusted findings-provided move.

    The "trusted" carve-out matters because handlers can put PV (principal
    variation) move sequences in findings — for example
    ``stockfish.top_moves[*].pv_san`` lists moves like
    ``["Rxd4", "Rxd4", "Qxa3"]`` where only the first is legal on the root
    position; the others are legal only on subsequent boards along the PV.
    Without this carve-out, the validator would flag any LLM response that
    cites a PV beyond its first move. See ADR-0009 / the v3 prompt's
    "you may cite a PV" allowance.

    Behaviour: a SAN or UCI in the response passes if either (a) it appears
    in the flattened findings move set (top_moves.san, stockfish.top_moves.san,
    stockfish.top_moves.pv_san, top_moves.uci, stockfish.top_moves.uci), or
    (b) it's legal on the root board. Else it's flagged ``illegal_move``.
    """
    board = chess.Board(fen)
    trusted_sans, trusted_ucis = _collect_trusted_moves(findings)
    # Chunk 6.9 sub-item 2 — normalise check/mate marks when matching
    # against trusted_sans. The validator was flagging "Bxg7" as
    # illegal even though "Bxg7+" (with check) was in a Stockfish PV;
    # the LLM legitimately wrote the move without the check mark and
    # found itself rejected. Stripping `+`/`#` from both sides treats
    # them as semantically equivalent (same move; different annotation).
    # Do NOT strip `x` — `Bxg7` and `Bg7` are different moves (capture
    # vs quiet) and must not be treated as equivalent.
    _strip_san_marks = lambda s: s.rstrip("+#")  # noqa: E731
    normalised_trusted_sans = {_strip_san_marks(t) for t in trusted_sans}
    errors: list[ValidationError] = []
    seen: set[str] = set()

    for match in _SAN_RE.finditer(response_text):
        san = match.group(0)
        if san in seen:
            continue
        seen.add(san)
        if san in trusted_sans:
            continue
        # Try the normalised match before falling through to legality.
        if _strip_san_marks(san) in normalised_trusted_sans:
            continue
        if not _is_legal_san(san, board):
            errors.append(ValidationError(
                kind="illegal_move",
                detail=(
                    f"SAN move '{san}' is not legal on the root position "
                    f"and does not appear in findings."
                ),
                span=san,
            ))

    for match in _UCI_RE.finditer(response_text):
        uci = match.group(1)
        if uci in seen:
            continue
        seen.add(uci)
        if uci in trusted_ucis:
            continue
        if not _is_legal_uci(uci, board):
            errors.append(ValidationError(
                kind="illegal_move",
                detail=(
                    f"UCI move '{uci}' is not legal on the root position "
                    f"and does not appear in findings."
                ),
                span=uci,
            ))
    return errors


def _collect_trusted_moves(findings: Any) -> tuple[set[str], set[str]]:
    """Walk findings and collect every ``san`` / ``uci`` / ``pv_san`` string.

    Handler-agnostic: doesn't care about the schema shape. Recursively visits
    dicts and lists; whenever a key ``san`` or ``uci`` maps to a string, or
    a key ``pv_san`` maps to a list of strings, the values join the trusted
    set. Works for both the current findings shape (network + stockfish) and
    any future shape that follows the same key conventions.

    Returns a ``(sans, ucis)`` tuple of frozen-after-walk sets.
    """
    sans: set[str] = set()
    ucis: set[str] = set()

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "san" and isinstance(v, str):
                    sans.add(v)
                elif k == "uci" and isinstance(v, str):
                    ucis.add(v)
                elif k == "pv_san" and isinstance(v, list):
                    for s in v:
                        if isinstance(s, str):
                            sans.add(s)
                else:
                    walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(findings)
    return sans, ucis


def _is_legal_san(san: str, board: chess.Board) -> bool:
    try:
        board.parse_san(san)
    except ValueError:
        # python-chess raises InvalidMoveError / IllegalMoveError /
        # AmbiguousMoveError (all subclasses of ValueError). Any failure is a
        # legality failure for our purposes — the LLM shouldn't produce
        # ambiguous SAN either.
        return False
    return True


def _is_legal_uci(uci: str, board: chess.Board) -> bool:
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return False
    return move in board.legal_moves


# --- Grounding --------------------------------------------------------------


def _check_grounding(
    response_text: str, findings: dict[str, Any],
) -> list[ValidationError]:
    flat_values = _flatten_numerics(findings)
    # Visit-share percentages are structurally derivable from raw counts
    # (visits / total_nodes_searched). The prompts allow citing them as
    # either raw count or percentage; only raw counts live as literals in
    # the flattened set. Compute the derived percentages once and use a
    # SEPARATE candidate set for the percent loop so derived values don't
    # leak into signed-number grounding (a "+27" eval shouldn't match a
    # derived 27% visit share).
    pct_values = flat_values | set(_collect_derived_percentages(findings))
    errors: list[ValidationError] = []
    seen: set[str] = set()

    for match in _SIGNED_NUMBER_RE.finditer(response_text):
        span = match.group(0)
        if span in seen:
            continue
        seen.add(span)
        value = float(match.group(1))
        if not _is_grounded(value, flat_values, is_percent=False):
            errors.append(ValidationError(
                kind="ungrounded_eval",
                detail=f"Signed number '{span}' is not present in findings within tolerance.",
                span=span,
            ))

    for match in _PERCENT_RE.finditer(response_text):
        span = match.group(0)
        if span in seen:
            continue
        seen.add(span)
        value = float(match.group(1))
        if not _is_grounded(value, pct_values, is_percent=True):
            errors.append(ValidationError(
                kind="ungrounded_eval",
                detail=f"Percentage '{span}' is not present in findings within tolerance.",
                span=span,
            ))
    return errors


def _collect_derived_percentages(findings: dict[str, Any]) -> list[float]:
    """Visit-share style percentages derivable from the findings JSON.

    Phase 1+2 prompts allow citing MCTS visit shares as either a raw count
    (``842 visits``) OR a percentage of total search (``77% of search``).
    Only the raw counts live in the findings JSON as literals; the
    percentage form is structurally derivable from
    ``mcts.children[i].visits / mcts.total_nodes_searched`` (plus the
    ``visits_in_flight`` variant and the root_stats analogue). The
    flat-values grounding check sees only raw counts, so without this
    helper the validator flags every percent-form visit citation as
    ``ungrounded_eval``. Returns values in percentage form
    (e.g. 27.0, not 0.27) — the percent grounding path scales by 100x
    via ``_is_grounded``'s candidate expansion, so feeding it percent-form
    values keeps the symmetry with ``_PERCENT_RE``'s extraction.

    Scope is intentionally tight: only visit-share derivations. No
    general "scan every pair of scalars in findings" logic — that would
    risk accepting any coincidentally-near percentage. Chunk 7's new
    extractors (Stockfish multipv, threats, motifs, king-safety,
    material delta) may add their own derivable percentages; each
    addition gets its own explicit field-pair mapping plus one
    false-positive test, per the Chunk 6.6 plan's forward-reference.
    """
    out: list[float] = []
    mcts = findings.get("mcts")
    if not isinstance(mcts, dict):
        return out
    total = mcts.get("total_nodes_searched")
    if not isinstance(total, (int, float)) or total <= 0:
        return out
    for child in mcts.get("children", []) or []:
        if not isinstance(child, dict):
            continue
        v = child.get("visits")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(100.0 * v / total)
            vif = child.get("visits_in_flight")
            if isinstance(vif, (int, float)) and not isinstance(vif, bool):
                out.append(100.0 * (v + vif) / total)
    root_stats = mcts.get("root_stats")
    if isinstance(root_stats, dict):
        rv = root_stats.get("visits")
        if isinstance(rv, (int, float)) and not isinstance(rv, bool):
            out.append(100.0 * rv / total)
            rvif = root_stats.get("visits_in_flight")
            if isinstance(rvif, (int, float)) and not isinstance(rvif, bool):
                out.append(100.0 * (rv + rvif) / total)
    return out


def _flatten_numerics(obj: Any, out: set[float] | None = None) -> set[float]:
    """Walk ``obj`` recursively and collect every numeric leaf as a float.

    Bools are filtered: Python's ``bool`` subclasses ``int``, so without this
    guard ``True`` / ``False`` would pollute the set as 1.0 / 0.0 and cause
    spurious grounding matches.
    """
    if out is None:
        out = set()
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.add(float(obj))
        return out
    if isinstance(obj, dict):
        for v in obj.values():
            _flatten_numerics(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten_numerics(v, out)
    return out


def _is_grounded(value: float, flat_values: set[float], *, is_percent: bool) -> bool:
    """Check if ``value`` (with unit conversions) is in ``flat_values``.

    Evals are stored in findings as centipawns (``eval_cp: 42``) but the LLM
    typically cites them in pawn units ("+0.42"). Likewise WDL is stored as
    0-1 probabilities (``white_wins: 0.45``) but often cited as "45%". We
    enumerate the obvious equivalent representations before flagging.

    Tolerance: 10% relative or 0.02 absolute, whichever is larger — lenient
    enough to absorb LLM rounding ("+0.4" vs eval_cp=42) without making
    every number a free pass.
    """
    candidates: list[float] = [value]
    if is_percent:
        candidates.append(value / 100.0)
    else:
        if abs(value) < 25.0 and not value.is_integer():
            # Eval cited in pawn units → match a cp finding.
            candidates.append(value * 100.0)
        if abs(value) >= 10.0:
            # Eval cited in cp form → match a pawn-unit finding.
            candidates.append(value / 100.0)

    for cand in candidates:
        for fv in flat_values:
            tol = max(abs(fv) * 0.10, 0.02)
            if abs(cand - fv) <= tol:
                return True
    return False


# --- Trough-prose semantic-completeness check (Chunk 7c.5.A) ---------------


# Threshold for "the trajectory has a meaningful trough": q_trough deviates
# from q_first OR q_last by this much (in Q-units; range -1 to +1, so 0.10
# is a 10-percentage-point difference). Tuned 2026-05-25 against:
# - 7c walkthrough_rxd4 focused trajectory: max_delta = 0.276 → fires ✓
# - Canonical walkthrough_rxd4 normal-pass Rxd4: max_delta = 0.50 → fires ✓
# - Walkthrough Rxb1+ (refutation trajectory): max_delta = 0.38 → fires ✓
# - Italian-game Bc5 (concordant pick, no real trough): max_delta ≈ 0.03 → doesn't fire ✓
# T=0.10 is below all the "real trough" cases and above all the "no real
# trough" cases in the captured data. Re-tune if 7o or 7d surfaces an
# edge case where the threshold mis-fires.
#
# Chunk 7c.9 (sub-fix 4-pre.1): promoted from private `_TROUGH_DELTA_THRESHOLD`
# to public `TROUGH_DELTA_THRESHOLD` so the new `extract_trajectory_decomposition`
# extractor can import and gate on the same value. Future threshold tuning
# lands in ONE place. The private alias is preserved for back-compat (one
# external in-test reference exists; rather than touch tests we keep both
# names bound to the same float).
TROUGH_DELTA_THRESHOLD = 0.10
_TROUGH_DELTA_THRESHOLD = TROUGH_DELTA_THRESHOLD  # back-compat alias.

# Round-aware tolerance for "is this Q value cited in the prose?". 0.005
# means "+0.92" matches a target of 0.917 (delta 0.003) and "+0.917" also
# matches a target of 0.914 (delta 0.003) — both within rounding-to-2-or-3
# decimals tolerance. Tighter than _is_grounded's tolerance because we're
# checking semantic completeness, not numeric grounding.
_Q_CITATION_TOLERANCE = 0.005


def _collect_trajectory_summaries(findings: Any) -> list[dict]:
    """Walk findings recursively for any ``trajectory_summary`` substructure.

    A ``trajectory_summary`` is detected structurally: any dict containing
    both ``q_first`` AND ``q_trough`` keys with numeric values. This
    matches :class:`MCTSTrajectorySummary` (the Phase 2 MCTS trajectory
    summary used by the dual-timescale handlers) without coupling to the
    specific class name — future Pydantic models with the same shape
    (e.g., a Chunk-7d exploration-pass trajectory summary) will be
    detected automatically.

    Returns:
        List of dicts with ``q_first`` / ``q_last`` / ``q_trough`` /
        ``q_peak`` (when present) as floats, plus ``location`` (dotted
        path through the findings structure for error messages).
    """
    out: list[dict] = []

    def _walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            if (
                "q_first" in obj and "q_trough" in obj
                and isinstance(obj.get("q_first"), (int, float))
                and not isinstance(obj.get("q_first"), bool)
                and isinstance(obj.get("q_trough"), (int, float))
                and not isinstance(obj.get("q_trough"), bool)
                and isinstance(obj.get("q_last"), (int, float))
                and not isinstance(obj.get("q_last"), bool)
            ):
                out.append({
                    "q_first": float(obj["q_first"]),
                    "q_last": float(obj["q_last"]),
                    "q_trough": float(obj["q_trough"]),
                    "q_peak": float(obj["q_peak"]) if isinstance(obj.get("q_peak"), (int, float)) and not isinstance(obj.get("q_peak"), bool) else None,
                    "location": path or "<root>",
                })
            for k, v in obj.items():
                _walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _walk(v, f"{path}[{i}]")

    _walk(findings, "")
    return out


def _q_value_cited_in_prose(
    value: float, prose: str, tolerance: float = _Q_CITATION_TOLERANCE,
) -> bool:
    """Round-aware: does ANY signed-decimal in the prose match ``value``?

    Reuses ``_SIGNED_NUMBER_RE`` (the same extractor the grounding check
    uses) so the citation set is consistent across checks. Tolerance is
    deliberately tight: this is a semantic-completeness check, NOT a
    numeric-grounding check.
    """
    for match in _SIGNED_NUMBER_RE.finditer(prose):
        try:
            cited = float(match.group(1))
        except (TypeError, ValueError):  # pragma: no cover — regex guarantees parseability
            continue
        if abs(cited - value) <= tolerance:
            return True
    return False


def _check_trough_completeness(
    response_text: str, findings: dict[str, Any],
    *, threshold: float = TROUGH_DELTA_THRESHOLD,
) -> list[ValidationError]:
    """Flag prose that cites q_first/q_last from a trajectory_summary
    whose q_trough deviates by ≥ ``threshold`` without also citing q_trough.

    This is a CONDITIONAL-CITATION check (different shape from the
    ``_collect_derived_percentages`` check which is additive numeric
    grounding). Catches the 7c failure mode: prose said "Q stable in a
    tight +0.917 to +0.923 band" when the trajectory's q_trough was
    +0.647 — the warm-value-only citation framed a real trough as
    stability, misleading the reader.

    Per the trough-prose contract (v3+ prompts' rule 4): trough citation
    is required when the trough is meaningful. The validator enforces
    the contract; the prompt-side anti-pattern strengthening lands as a
    belt-and-suspenders iteration at 7m.
    """
    errors: list[ValidationError] = []
    for summary in _collect_trajectory_summaries(findings):
        q_first = summary["q_first"]
        q_last = summary["q_last"]
        q_trough = summary["q_trough"]
        location = summary["location"]

        # No real trough → nothing to enforce. Skip silently.
        max_delta = max(abs(q_trough - q_first), abs(q_trough - q_last))
        if max_delta <= threshold:
            continue

        # Real trough exists. Check what the prose cites.
        cites_first = _q_value_cited_in_prose(q_first, response_text)
        cites_last = _q_value_cited_in_prose(q_last, response_text)
        cites_trough = _q_value_cited_in_prose(q_trough, response_text)

        # Trough-prose contract: if the prose cites either warm value,
        # it MUST also cite q_trough. Citing q_trough without q_first /
        # q_last is fine (the trough story is complete on its own).
        if (cites_first or cites_last) and not cites_trough:
            # Pick whichever warm value was cited for the span; prefer
            # q_last (the more common rule-4 violation pattern).
            cited_value = q_last if cites_last else q_first
            errors.append(ValidationError(
                kind="incomplete_trough_citation",
                detail=(
                    f"Prose cites {cited_value:+.3f} (from {location}'s "
                    f"trajectory_summary q_first/q_last) but does not cite "
                    f"q_trough ({q_trough:+.3f}, delta "
                    f"{abs(q_trough - cited_value):.3f} > threshold "
                    f"{threshold}). Per the trough-prose contract, prose "
                    f"citing q_first or q_last from a trajectory with a "
                    f"meaningful trough MUST also cite q_trough — warm-value-"
                    f"only citations frame post-trough drift as 'stable' "
                    f"which is misleading. See v3+ prompts rule 4."
                ),
                span=f"{cited_value:+.3f}",
            ))
    return errors
