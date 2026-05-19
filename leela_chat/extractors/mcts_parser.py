"""Parse Lc0 verbose-move-stats lines into structured MCTS findings.

Pure-Python, no I/O. Consumes the raw ``info string`` payloads captured
by :class:`leela_chat.engine.Lc0Engine` and emits the Pydantic
``MCTSFindings`` that handlers / the LLM prompt consume downstream.

Format observed on Lc0 v0.32.1 (per ADR-0004, pinned). Each per-move
candidate line:

::

    {uci} ({lc0_move_idx}) N: {visits} (+ {in_flight}) (P: {pct}%)
        (WL: {wl}) (D: {draw}) (M: {moves_left}) (Q: {q})
        (U: {u}) (S: {s}) (V: {v})

Plus a final root-aggregate line of the same shape but with the literal
token ``node`` in place of the UCI move and no ``U`` / ``S`` fields
(the root has no parent, so no exploration bonus or softmax term).

Field semantics — all values from the side-to-move's perspective:

- ``N`` visits (int, ≥0) — the engine's commitment metric, the load-
  bearing signal for "which move did MCTS actually pick"
- ``+ Y`` in-flight visits (int, ≥0) — visits dispatched but not yet
  backpropagated; nonzero at search-end is normal
- ``P`` policy prior, as a percent (float, 0-100)
- ``Q`` action value (float, -1 to +1; +1 = forced win, -1 = forced loss)
- ``V`` value-head output for this move's child (float, -1 to +1)
- ``WL`` win-minus-loss (float, -1 to +1; distinct from Q in that Q
  also blends in the draw-score)
- ``D`` draw probability (float, 0 to 1)
- ``M`` moves-left head estimate (float, ≥0; near 0 in mate situations)
- ``U`` PUCT exploration bonus (float, ≥0)
- ``S`` softmax/score-type marker (float, undocumented — stored opaque)

Design constraints from Phase 2 plan G2.3: preserve per-child raw counts
(visits, q, prior, etc.) rather than aggregating. Aggregates can be
derived; raws cannot be recovered.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import chess


# Matches the leading "move (idx)" portion of every stat line. The move is
# either a UCI move (4-5 chars) or the literal token "node" for the root
# summary line.
_HEADER_RE = re.compile(
    r"^\s*(?P<move>[a-h][1-8][a-h][1-8][qrbn]?|node)\s+"
    r"\(\s*(?P<idx>\d+)\s*\)"
)

# Matches the "N: X (+ Y)" sub-clause. N is unwrapped; the in-flight count
# is wrapped in parens.
_N_RE = re.compile(
    r"N:\s*(?P<visits>\d+)\s*\(\+\s*(?P<inflight>\d+)\s*\)"
)

# Matches every "(KEY: VALUE)" parenthesized field after the N clause.
# VALUE may be signed, decimal, with optional percent suffix.
_FIELD_RE = re.compile(
    r"\((?P<key>[A-Z]+):\s*(?P<value>-?\d+(?:\.\d+)?)\s*%?\s*\)"
)


class MCTSChild(BaseModel):
    """One root-level child move with its raw MCTS statistics.

    Per Phase 2 plan G2.3, raw per-child values are preserved (not
    aggregated into summary statistics) so that Phase 6 counterfactual
    experiments can replay PUCT selection with perturbed inputs.
    """

    uci: str = Field(..., description="Move in UCI form, e.g. ``e2e4``.")
    san: str = Field(..., description="Move in SAN, computed from the root board.")

    visits: int = Field(..., ge=0, description="Completed visits to this move's subtree.")
    visits_in_flight: int = Field(
        ..., ge=0,
        description="Visits dispatched but not yet backpropagated at capture time.",
    )

    prior_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Policy network prior probability, as a percent (0-100).",
    )
    q: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Mean MCTS action value across visits (side-to-move perspective).",
    )
    v: float | None = Field(
        ..., ge=-1.0, le=1.0,
        description=(
            "Value-head output for the position after this move, or None "
            "if Lc0 hasn't expanded this move yet (it emits the sentinel "
            "``(V: -.----)`` in that case — typically low-prior moves the "
            "search didn't even attempt)."
        ),
    )
    wl: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Win-minus-loss component across the subtree.",
    )
    draw: float = Field(
        ..., ge=0.0, le=1.0,
        description="Draw probability across the subtree visits.",
    )
    moves_left: float = Field(
        ..., ge=0.0,
        description="Moves-left head estimate (in plies / 2).",
    )
    u: float = Field(
        ..., ge=0.0,
        description="PUCT exploration bonus at capture time.",
    )
    s: float = Field(
        ...,
        description="Lc0 'S' field — undocumented; stored opaque.",
    )

    pv_san: list[str] = Field(
        default_factory=list,
        description=(
            "Principal variation in SAN, beginning with this move. Empty "
            "in Phase 2 Chunk 2 — verbose-move-stats does not surface per-"
            "child PV; Chunks 5-6 may revisit via deeper tree dumps or "
            "move-forcing if the handlers need it."
        ),
    )

    model_config = ConfigDict(frozen=True)


class MCTSRootStats(BaseModel):
    """Aggregate stats for the root position itself (Lc0's ``node`` summary line).

    Distinct from any individual child: this is the total visits and the
    blended Q/V/WL across the whole search.
    """

    visits: int = Field(..., ge=0, description="Total completed visits across the search.")
    visits_in_flight: int = Field(..., ge=0, description="In-flight visits at capture.")

    prior_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description=(
            "Sum of priors over reported children (near 100% when all moves "
            "are reported, less when low-prior moves were truncated)."
        ),
    )
    q: float = Field(..., ge=-1.0, le=1.0, description="Root Q (side-to-move perspective).")
    v: float = Field(..., ge=-1.0, le=1.0, description="Root value-head output.")
    wl: float = Field(..., ge=-1.0, le=1.0, description="Root W-L.")
    draw: float = Field(..., ge=0.0, le=1.0, description="Root draw probability.")
    moves_left: float = Field(..., ge=0.0, description="Root moves-left estimate (plies/2).")

    model_config = ConfigDict(frozen=True)


class MCTSFindings(BaseModel):
    """Structured MCTS analysis of a position, ready for handler / LLM consumption."""

    total_nodes_searched: int = Field(
        ..., ge=0,
        description=(
            "Engine's reported total search size (root visits + in-flight). "
            "May differ from caller-requested ``nodes`` because Lc0 reports "
            "completed visits, not requested budget."
        ),
    )
    bestmove_uci: str | None = Field(
        ...,
        description="Engine's selected best move in UCI form, or None on incomplete search.",
    )
    bestmove_san: str | None = Field(
        ...,
        description="Engine's selected best move in SAN, or None when bestmove_uci is None.",
    )
    children: list[MCTSChild] = Field(
        ...,
        description=(
            "All root-level candidates Lc0 reported, sorted by visits desc. "
            "Includes zero-visit candidates (low-prior moves Lc0 chose not "
            "to explore) — useful for ``did_you_consider``-style handlers."
        ),
    )
    root_stats: MCTSRootStats | None = Field(
        ...,
        description=(
            "Root-position aggregate from Lc0's final ``node`` summary line. "
            "None on incomplete search where Lc0 didn't emit the summary."
        ),
    )
    search_complete: bool = Field(
        ...,
        description="True iff Lc0 returned a bestmove (search wasn't interrupted).",
    )

    model_config = ConfigDict(frozen=True)


def _parse_one_line(line: str) -> dict | None:
    """Parse a single info-string line into a flat dict, or return None if
    the line isn't a stat-shaped line (candidate or node summary).

    Returns dict keys: ``move``, ``idx``, ``visits``, ``visits_in_flight``,
    plus any of ``P`` / ``WL`` / ``D`` / ``M`` / ``Q`` / ``U`` / ``S`` /
    ``V`` that appear in the line. Caller maps these to Pydantic.
    """
    header = _HEADER_RE.match(line)
    if not header:
        return None
    n_match = _N_RE.search(line, pos=header.end())
    if not n_match:
        # A "move (idx)" prefix without N: ... is malformed; skip.
        return None

    parsed: dict = {
        "move": header.group("move"),
        "idx": int(header.group("idx")),
        "visits": int(n_match.group("visits")),
        "visits_in_flight": int(n_match.group("inflight")),
    }
    for field in _FIELD_RE.finditer(line, pos=n_match.end()):
        parsed[field.group("key")] = float(field.group("value"))
    return parsed


def parse_verbose_move_stats(
    info_string_lines: list[str],
    *,
    board: "chess.Board",
    bestmove_uci: str | None,
    search_complete: bool,
) -> MCTSFindings:
    """Turn raw Lc0 info-string lines into a structured ``MCTSFindings``.

    Args:
        info_string_lines: The list of payloads from
            :class:`leela_chat.engine.Lc0SearchResult.info_string_lines`.
            Non-stat lines are filtered out automatically.
        board: The position Lc0 was analysing. Used for UCI → SAN
            translation; the caller is responsible for passing the board
            that matches the search.
        bestmove_uci: Lc0's chosen best move (from ``Lc0SearchResult``).
            May be None if the search was interrupted.
        search_complete: Whether Lc0 returned a bestmove. False propagates
            into the findings so downstream handlers can flag partial data.

    Returns:
        :class:`MCTSFindings` with children sorted by visits desc (highest-
        visit move first), preserving every candidate Lc0 reported. The
        ``node`` summary line, if present, is stored in ``root_stats``.
    """
    import chess  # local import to keep module-level imports cheap

    parsed_lines: list[dict] = []
    root_raw: dict | None = None

    for line in info_string_lines:
        parsed = _parse_one_line(line)
        if parsed is None:
            continue
        if parsed["move"] == "node":
            root_raw = parsed
        else:
            parsed_lines.append(parsed)

    # Sort children by visits descending. Stable tie-break by prior_pct
    # descending so equal-visit candidates have a deterministic order.
    parsed_lines.sort(
        key=lambda d: (d["visits"], d.get("P", 0.0)),
        reverse=True,
    )

    children: list[MCTSChild] = []
    for d in parsed_lines:
        uci = d["move"]
        # SAN translation is best-effort; fall back to UCI for any move
        # python-chess can't legalise (would only happen on parser bug
        # or malformed FEN).
        try:
            mv = chess.Move.from_uci(uci)
            san = board.san(mv)
        except (ValueError, AssertionError, chess.IllegalMoveError):
            san = uci

        children.append(MCTSChild(
            uci=uci,
            san=san,
            visits=d["visits"],
            visits_in_flight=d["visits_in_flight"],
            prior_pct=d.get("P", 0.0),
            q=d.get("Q", 0.0),
            v=d.get("V"),  # None when Lc0 emitted "(V: -.----)" sentinel
            wl=d.get("WL", 0.0),
            draw=d.get("D", 0.0),
            moves_left=d.get("M", 0.0),
            u=d.get("U", 0.0),
            s=d.get("S", 0.0),
        ))

    root_stats: MCTSRootStats | None = None
    if root_raw is not None:
        root_stats = MCTSRootStats(
            visits=root_raw["visits"],
            visits_in_flight=root_raw["visits_in_flight"],
            prior_pct=root_raw.get("P", 0.0),
            q=root_raw.get("Q", 0.0),
            v=root_raw.get("V", 0.0),
            wl=root_raw.get("WL", 0.0),
            draw=root_raw.get("D", 0.0),
            moves_left=root_raw.get("M", 0.0),
        )

    bestmove_san: str | None = None
    if bestmove_uci is not None:
        try:
            mv = chess.Move.from_uci(bestmove_uci)
            bestmove_san = board.san(mv)
        except (ValueError, AssertionError, chess.IllegalMoveError):
            bestmove_san = None

    if root_stats is not None:
        total_nodes = root_stats.visits + root_stats.visits_in_flight
    else:
        total_nodes = sum(c.visits for c in children)

    return MCTSFindings(
        total_nodes_searched=total_nodes,
        bestmove_uci=bestmove_uci,
        bestmove_san=bestmove_san,
        children=children,
        root_stats=root_stats,
        search_complete=search_complete,
    )
