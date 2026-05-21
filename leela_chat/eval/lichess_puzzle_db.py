"""Lichess Puzzle DB sampling API.

The Lichess Puzzles Open Database (~5M puzzles, free, legal, available
at ``https://database.lichess.org/lichess_db_puzzle.csv.zst``) is the
project's primary external corpus for:

- The Chunk 6.8 smoke battery (stratified ~20-position sample for
  validation-coverage / regression / operational checks).
- Phase 3 probe-training data (themes are pre-labeled concept tags;
  ``fork`` / ``pin`` / ``discoveredAttack`` / ``endgame`` etc. are
  concept labels delivered for free at scale).
- Phase 4 SAE training data (position-activation pairs at scale).
- Phase 6+ counterfactual experiments (theme-stratified sampling).

This module is the canonical sampling layer. The corpus itself lives on
the network volume at ``/workspace/lichess_puzzle_db/`` (see
``leela_chat/scripts/download_lichess_puzzles.py`` for the
download + parse + integrity-check pipeline).

Schema of the parsed Parquet (matches the CSV columns):
- ``PuzzleId`` (str)
- ``FEN`` (str) — position from which the solver moves first
- ``Moves`` (str, space-separated UCI) — the FULL solution including
  the opponent's setup move at index 0. **The solver's first move is
  at index 1** (use ``solver_first_uci``).
- ``Rating`` (int)
- ``RatingDeviation`` (int)
- ``Popularity`` (int)
- ``NbPlays`` (int)
- ``Themes`` (list[str] OR space-separated str depending on parse)
- ``GameUrl`` (str)
- ``OpeningTags`` (str | null)

The Lichess docs note that "all player moves of the solution are 'only
moves'" — so at lower ratings (1800-2200) MCTS at 1000 nodes finds the
solution ~95%+ of the time (concordant case). The harder bands
(2600-3000, 3000-3400, 3400+) produce more divergent cases where
``focal_uci != mcts_bestmove_uci`` — those exercise the dual-timescale
framework's most-distinctive prose patterns (search-refutes,
MCTS-rescue).
"""
from __future__ import annotations

import functools
import hashlib
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import chess
from loguru import logger

# Lazy imports — pyarrow + zstandard only imported when actually used so
# unit tests that build synthetic in-memory tables don't pay the import
# cost (and don't fail if these deps aren't installed locally).


DEFAULT_PARQUET_PATH = Path(
    "/workspace/lichess_puzzle_db/lichess_db_puzzle.parquet"
)
DEFAULT_CSV_PATH = Path(
    "/workspace/lichess_puzzle_db/lichess_db_puzzle.csv"
)
LICHESS_PUZZLE_DB_URL = (
    "https://database.lichess.org/lichess_db_puzzle.csv.zst"
)


# === material-signature filter ============================================


def material_signature(fen: str) -> str:
    """Return a stable string describing the position's material.

    Format: ``<white_pieces>_vs_<black_pieces>``, with pieces sorted
    alphabetically and kings included. Example: ``KNN_vs_K`` is the
    canonical "two-knight endgame" theoretical-draw position that BT4
    doesn't interpret correctly (project-memory endgame-theory gap).

    Args:
        fen: Position in FEN.

    Returns:
        Material signature string. Pieces are upper-case symbols (P, N,
        B, R, Q, K). Both sides always include their king.
    """
    board = chess.Board(fen)
    white_pieces: list[str] = []
    black_pieces: list[str] = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        sym = piece.symbol().upper()
        if piece.color == chess.WHITE:
            white_pieces.append(sym)
        else:
            black_pieces.append(sym)
    return "".join(sorted(white_pieces)) + "_vs_" + "".join(sorted(black_pieces))


# === Moves-column parsing =================================================


def solver_first_uci(moves: str) -> str:
    """Extract the solver's first move from the Moves column.

    Per the current Lichess Puzzle DB convention (database.lichess.org,
    2024+): ``FEN`` is the position from which the solver plays first,
    and ``Moves`` is the space-separated UCI sequence of the resulting
    line — ``Moves[0]`` is the solver's first move, ``Moves[1]`` is the
    opponent's reply, ``Moves[2]`` is the solver's second move, etc.

    The side-to-move in ``FEN`` is the solver's color; ``Moves[0]`` is
    legal on ``FEN``.

    (Earlier versions of the Lichess puzzle data used a different
    convention where Moves[0] was the opponent's setup; that convention
    is NOT current. Sanity check: the focal move must be legal on the
    FEN — verified by python-chess in the handler call sites.)

    Args:
        moves: The Moves column value (space-separated UCI).

    Returns:
        UCI of the solver's first move (``Moves[0]``).

    Raises:
        ValueError: if the Moves string is empty.
    """
    parts = moves.split()
    if not parts:
        raise ValueError(f"Moves column is empty: {moves!r}")
    return parts[0]


def themes_to_list(themes_value: Any) -> list[str]:
    """Normalise the Themes column to a list of strings.

    The Themes column may parse as a single space-separated string
    (CSV path) or as a list (Parquet path). This helper accepts either
    and returns a clean list.

    Args:
        themes_value: The Themes column value from the corpus.

    Returns:
        List of theme tags (e.g. ``["fork", "middlegame", "short"]``).
    """
    if themes_value is None:
        return []
    if isinstance(themes_value, str):
        return themes_value.split()
    if isinstance(themes_value, (list, tuple)):
        return [str(t) for t in themes_value]
    return [str(themes_value)]


# === corpus loading =======================================================


@functools.lru_cache(maxsize=2)
def load_corpus(path: Path | str = DEFAULT_PARQUET_PATH):
    """Lazy-load the parsed corpus (Parquet) as a PyArrow Table.

    Cached at module level — subsequent calls return the same table.
    Use ``load_corpus.cache_clear()`` in tests if you need a fresh load.

    Args:
        path: Path to the Parquet file. Defaults to the network-volume
            location set up by ``download_lichess_puzzles.py``.

    Returns:
        ``pyarrow.Table`` with the columns documented at module level.

    Raises:
        FileNotFoundError: if the Parquet doesn't exist.
    """
    import pyarrow.parquet as pq  # lazy import
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Lichess puzzle Parquet not found at {p}. "
            f"Run `python -m leela_chat.scripts.download_lichess_puzzles` "
            f"to fetch + parse the corpus first."
        )
    return pq.read_table(str(p))


def _is_solver_move_legal(fen: str, solver_uci: str) -> bool:
    """Return True iff ``solver_uci`` parses cleanly AND is legal on ``fen``.

    Permanent immunity to the Lichess `Moves[0]`/`Moves[1]` convention bug
    class (Chunk 6.8): if `solver_first_uci` ever returns a move that's the
    opponent's reply (opposite color from FEN's side-to-move), that move
    will be illegal on the FEN and we catch it at sampling time rather
    than at smoke-battery-runtime.

    Wraps ``chess.Move.from_uci()`` in try/except because the corpus has
    5.94M rows and any data-drift edge case (truncated UCI, wrong format,
    missing promotion piece) must be handled gracefully rather than crash
    the whole sampling pass.

    Args:
        fen: Position in FEN.
        solver_uci: The candidate solver move in UCI.

    Returns:
        ``True`` iff the move parses AND is legal on the FEN-derived board.
        ``False`` for either parse failure OR illegality.
    """
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(solver_uci)
    except (ValueError, IndexError):
        return False
    return move in board.legal_moves


# === sampling API =========================================================


def sample_by_theme(
    theme: str | Sequence[str],
    n: int,
    rating_range: tuple[int, int] = (1500, 2400),
    exclude_themes: Sequence[str] = (),
    exclude_material_signatures: Sequence[str] = (),
    seed: int = 0,
    table=None,
) -> list[dict]:
    """Stratified random sample of puzzles matching theme + rating + exclusions.

    Theme matching is **all-of** when ``theme`` is a sequence (every theme
    must appear in the puzzle's Themes column) and **single-tag** when
    ``theme`` is a string. To match "any of [a, b, c]," call this function
    separately per theme and merge results.

    Args:
        theme: A single theme tag or a sequence of tags (all-of match).
        n: Maximum number of rows to return. If the filter matches fewer
            than ``n``, returns all matching rows (does not pad).
        rating_range: ``(min_rating, max_rating)`` inclusive.
        exclude_themes: Puzzles containing ANY of these themes are
            dropped (e.g. ``["mateIn1", "mateIn2"]`` to skip too-easy).
        exclude_material_signatures: Material signatures (from
            ``material_signature``) to drop, e.g. ``["KNN_vs_K"]`` for
            the known endgame-theory gap.
        seed: PRNG seed for deterministic sampling. Same seed +
            same corpus → same sample.
        table: A PyArrow Table to sample from. If None, calls
            ``load_corpus()``. Provided for tests + parameterised use.

    Returns:
        List of dicts with keys: ``PuzzleId``, ``FEN``,
        ``solver_first_uci``, ``Rating``, ``Themes``, ``OpeningTags``,
        ``material_signature``. Length ≤ ``n``.
    """
    if table is None:
        table = load_corpus()

    themes_required: list[str] = (
        [theme] if isinstance(theme, str) else list(theme)
    )
    exclude_set = set(exclude_themes)
    exclude_mat_set = set(exclude_material_signatures)

    # Convert to a row-iterator: Parquet → Python dicts. This is slow on
    # a 5M-row corpus if we naively iterate; we filter pre-iteration via
    # PyArrow's compute kernels.
    import pyarrow.compute as pc  # lazy

    # Rating filter is cheap — use PyArrow.
    rating = table.column("Rating")
    rating_mask = pc.and_(
        pc.greater_equal(rating, rating_range[0]),
        pc.less_equal(rating, rating_range[1]),
    )
    filtered = table.filter(rating_mask)

    # Theme filter requires per-row inspection because themes are
    # space-separated strings (or lists). Materialize as Python.
    rows: list[dict] = []
    examined_count = 0
    dropped_count = 0  # legality + parse failures (Chunk 6.8.1 guard)
    for batch in filtered.to_batches(max_chunksize=10000):
        records = batch.to_pylist()
        for rec in records:
            examined_count += 1
            tags = themes_to_list(rec.get("Themes"))
            tag_set = set(tags)
            # All required themes present?
            if not all(t in tag_set for t in themes_required):
                continue
            # Any excluded theme present?
            if exclude_set & tag_set:
                continue
            # Material signature exclusion?
            if exclude_mat_set:
                msig = material_signature(rec["FEN"])
                if msig in exclude_mat_set:
                    continue
            # Build the output row.
            try:
                focal = solver_first_uci(rec["Moves"])
            except ValueError:
                continue  # malformed Moves field; skip silently
            # Permanent guard against the Lichess Moves[0]/Moves[1]
            # convention-bug class: the solver's first move MUST be
            # legal on the puzzle's FEN. If not, drop the row with
            # a WARNING log (data-drift visibility); if `solver_first_uci`
            # ever regresses to returning the opponent's reply, we catch
            # it at sample time, not at smoke-battery runtime.
            if not _is_solver_move_legal(rec["FEN"], focal):
                logger.warning(
                    f"sample_by_theme: dropping puzzle {rec.get('PuzzleId')}: "
                    f"solver move {focal!r} is not legal on FEN "
                    f"{rec['FEN'][:40]}..."
                )
                dropped_count += 1
                continue
            rows.append({
                "PuzzleId": rec["PuzzleId"],
                "FEN": rec["FEN"],
                "solver_first_uci": focal,
                "Rating": rec["Rating"],
                "Themes": tags,
                "OpeningTags": rec.get("OpeningTags"),
                "material_signature": material_signature(rec["FEN"]),
            })

    # If any rows were dropped due to legality / parse failures,
    # escalate log level by drop ratio: a small absolute count over a
    # large examined_count is expected data-integrity noise (e.g. 5.94M
    # rows have edge cases); a high ratio is a real signal (either
    # `solver_first_uci` has a bug OR the corpus drifted).
    if dropped_count > 0 and examined_count > 0:
        ratio = dropped_count / examined_count
        if ratio <= 0.001:  # ≤ 0.1%: expected noise
            logger.info(
                f"sample_by_theme: dropped {dropped_count} of "
                f"{examined_count} rows ({ratio:.3%}) due to illegal/"
                f"malformed solver moves"
            )
        else:  # > 0.1%: real signal worth investigating
            logger.warning(
                f"sample_by_theme: dropped {dropped_count} of "
                f"{examined_count} rows ({ratio:.2%}) due to illegal/"
                f"malformed solver moves — drop rate is unusually high; "
                f"investigate solver_first_uci OR corpus drift"
            )

    # Deterministic sample.
    if len(rows) <= n:
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, n)


def sample_stratified(
    strata: dict[str, dict],
    seed: int = 0,
    table=None,
) -> dict[str, list[dict]]:
    """Sample multiple categories per a strata configuration.

    Calls ``sample_by_theme`` once per stratum and deduplicates by
    ``PuzzleId`` across all strata (a puzzle that matches two strata's
    filters appears in the first stratum only).

    Each stratum dict accepts:

    - ``themes`` (str | list[str], required) — passed as ``theme`` to
      ``sample_by_theme``.
    - ``n`` (int, required) — number of samples to draw.
    - ``rating_range`` (tuple[int, int], required) — ``(min, max)``.
    - ``exclude_themes`` (list[str], optional)
    - ``exclude_material_signatures`` (list[str], optional)
    - ``fallback_to_lower`` (bool, optional, default False) — if the
      band has fewer matching puzzles than ``n``, the missing slots
      do NOT crash; the next-lower stratum (by rating_range[1]) fills
      them. Use for "extreme" bands (e.g. 3400+) where the DB may have
      very few matching puzzles.

    Returns:
        Dict from stratum name to list of sample rows. Each list has
        length ≤ that stratum's ``n``. With ``fallback_to_lower=True``,
        the effective ``n`` is honored by borrowing from lower bands.
    """
    if table is None:
        table = load_corpus()

    # Deduplication set across all strata.
    seen_puzzle_ids: set[str] = set()
    out: dict[str, list[dict]] = {}

    # Order strata by rating_range[1] descending so high bands fill first
    # and fallback_to_lower borrows from already-sampled lower bands.
    # But the strata config is a dict (preserves insertion order in
    # Python 3.7+); we honor that order for primary sampling and only
    # fall back when needed.
    for name, cfg in strata.items():
        themes = cfg.get("themes")
        n = cfg["n"]
        rating_range = cfg["rating_range"]
        exclude_themes = tuple(cfg.get("exclude_themes", ()))
        exclude_mats = tuple(cfg.get("exclude_material_signatures", ()))
        # Salt the seed per stratum so different strata use different
        # PRNG sequences even with the same global seed.
        stratum_seed = (seed * 1000 + hash(name)) & 0x7FFFFFFF

        rows = sample_by_theme(
            themes,
            n + 50,  # over-sample so dedup doesn't starve us
            rating_range=rating_range,
            exclude_themes=exclude_themes,
            exclude_material_signatures=exclude_mats,
            seed=stratum_seed,
            table=table,
        )

        # Dedup across strata.
        unique = [r for r in rows if r["PuzzleId"] not in seen_puzzle_ids]
        chosen = unique[:n]
        for r in chosen:
            seen_puzzle_ids.add(r["PuzzleId"])
        out[name] = chosen

    # Fallback pass: any stratum below its target n that opted in?
    for name, cfg in strata.items():
        if not cfg.get("fallback_to_lower", False):
            continue
        n_target = cfg["n"]
        if len(out[name]) >= n_target:
            continue
        deficit = n_target - len(out[name])
        # Walk strata in reverse (lower-rated first) looking for surplus.
        # Lower-rated strata are MORE likely to have surplus.
        for other_name, other_cfg in strata.items():
            if other_name == name:
                continue
            # Try a lower-rated re-sample with the borrower's other constraints.
            lower_range = other_cfg["rating_range"]
            if lower_range[1] >= cfg["rating_range"][1]:
                continue  # not strictly lower
            extra_rows = sample_by_theme(
                cfg.get("themes"),
                deficit + 20,
                rating_range=lower_range,
                exclude_themes=tuple(cfg.get("exclude_themes", ())),
                exclude_material_signatures=tuple(
                    cfg.get("exclude_material_signatures", ())
                ),
                seed=(seed * 1000 + hash(name) + 7) & 0x7FFFFFFF,
                table=table,
            )
            extra_unique = [
                r for r in extra_rows if r["PuzzleId"] not in seen_puzzle_ids
            ]
            take = extra_unique[:deficit]
            for r in take:
                seen_puzzle_ids.add(r["PuzzleId"])
            out[name].extend(take)
            deficit -= len(take)
            if deficit <= 0:
                break

    return out
