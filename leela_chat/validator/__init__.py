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
