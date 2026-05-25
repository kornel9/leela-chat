"""Threshold-calibration sweep for TROUGH_DELTA_THRESHOLD.

7d entry-sequence follow-up (2026-05-26): characterize the empirical
distribution of `max_delta` (= max(|q_trough - q_first|, |q_trough -
q_last|)) on cold-engine MCTS trajectories across N hard puzzles. The
~0.10 threshold inherited from 7c.5.A was hand-tuned against 3 fixtures.
This sweep gives us the population distribution behind it so it can be
empirically anchored before Phase 3 probes train against
`trough_explanation` findings.

Per ADR-0016 (hybrid warm/cold engine architecture): each measurement
runs on ONE long-lived Lc0 subprocess with a fresh `game=object()`
sentinel per query, which triggers `ucinewgame` between queries (NN
cache + tree cleared) — the cold-regime that matches what production
trough_explanation will see at handler trajectory call sites once Step 5
wires `game=object()` through.

Stratification:
- 5 themes × N_per_theme positions each = N_total total.
- Rating range biased AGGRESSIVELY hard (default: 2200-3500).
- Themes default: sacrifice / discoveredAttack / attraction /
  endgame / middlegame. (First three are tactical; expected to surface
  troughs reliably. Endgame is different MCTS dynamics; middlegame
  is broad baseline.)

Output (JSONL, one line per measurement; written incrementally so
partial results are usable if the run is interrupted):

  {
    "puzzle_id": "...", "fen": "...", "focal_uci": "...",
    "rating": 2342, "themes": ["sacrifice", "middlegame"],
    "material_signature": "...",
    "stratum": "sacrifice",  # which theme bucket this position came from
    "max_delta": 0.421, "q_first": 0.123, "q_trough": -0.298,
    "q_last": 0.105, "q_trough_at_visits": 1234,
    "focal_visits": 23456, "total_nodes_searched": 50001,
    "wall_s": 5.4, "error": null
  }

Cost: NO LLM calls. Pure MCTS measurement.
Estimated pod wall-clock: ~5s per measurement × N_total positions.
  5000 positions ≈ ~7 hours; 3000 ≈ ~4 hours.

Usage (on the pod):

    uv run python -m leela_chat.scripts.threshold_calibration_sweep

    # Custom sample size + rating band:
    uv run python -m leela_chat.scripts.threshold_calibration_sweep \\
        --n-per-theme 1000 --min-rating 2200 --max-rating 3500

    # Custom output path:
    uv run python -m leela_chat.scripts.threshold_calibration_sweep \\
        --out /workspace/threshold_calibration_5k.jsonl
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import chess
import chess.engine
from loguru import logger


DEFAULT_THEMES = [
    "sacrifice",
    "discoveredAttack",
    "attraction",
    "endgame",
    "middlegame",
]
DEFAULT_N_PER_THEME = 1000  # 5 themes × 1000 = 5000 total
DEFAULT_MIN_RATING = 2200  # FIDE Master+ equivalent puzzle difficulty
DEFAULT_MAX_RATING = 3500  # anything goes
NODES = 50_000
SAMPLE_SEED = 42  # deterministic stratified sample


class _EngineProxy:
    """Pass game= through to engine.analysis(); matches
    Lc0Engine.analyse_verbose's signature so extract_mcts_trajectory
    can be reused. Same shape as Step B's verify_game_sentinel.py
    proxy (commit d960d8f).
    """
    def __init__(self, raw_engine, game_param):
        self._eng = raw_engine
        self._game = game_param

    def analyse_verbose(
        self, board, *, nodes, log_live_stats=False, multipv=1, root_moves=None,
    ):
        from leela_chat.engine.lc0_engine import Lc0SearchResult
        del log_live_stats  # we always stream; matches Step B proxy
        info_string_lines: list[str] = []
        principal_variations: dict[int, list[str]] = {}
        analysis_kwargs: dict = {"multipv": multipv, "game": self._game}
        if root_moves is not None:
            analysis_kwargs["root_moves"] = root_moves
        t0 = time.perf_counter()
        with self._eng.analysis(
            board, chess.engine.Limit(nodes=nodes), **analysis_kwargs,
        ) as analysis:
            for info in analysis:
                s = info.get("string")
                if isinstance(s, str) and s:
                    info_string_lines.append(s)
                pv_moves = info.get("pv")
                if pv_moves:
                    pv_idx = info.get("multipv")
                    idx = int(pv_idx) if pv_idx is not None else 1
                    principal_variations[idx] = [m.uci() for m in pv_moves]
            best = analysis.wait()
        return Lc0SearchResult(
            info_string_lines=info_string_lines,
            bestmove_uci=best.move.uci() if best.move else None,
            nodes_requested=nodes,
            wall_seconds=time.perf_counter() - t0,
            search_complete=best.move is not None,
            multipv_requested=multipv,
            principal_variations=principal_variations,
        )


def _measure_one(eng, position: dict) -> dict:
    """Cold-engine measurement on one puzzle. Returns one JSONL row.

    On any error (illegal move, parse failure, MCTS error), returns a
    row with `error` set and other fields None so the JSONL is
    schema-consistent across rows.
    """
    from leela_chat.extractors.mcts_trajectory import extract_mcts_trajectory

    base_row = {
        "puzzle_id": position["PuzzleId"],
        "fen": position["FEN"],
        "focal_uci": position["solver_first_uci"],
        "rating": int(position["Rating"]),
        "themes": list(position["Themes"]),
        "material_signature": position["material_signature"],
        "stratum": position["_stratum"],
    }

    try:
        board = chess.Board(position["FEN"])
    except Exception as exc:
        return {**base_row, "error": f"fen_parse: {exc}",
                "max_delta": None, "q_first": None, "q_trough": None,
                "q_last": None, "q_trough_at_visits": None,
                "focal_visits": None, "total_nodes_searched": None,
                "wall_s": None}

    focal_uci = position["solver_first_uci"]
    t0 = time.perf_counter()
    try:
        mcts = extract_mcts_trajectory(
            board, engine=_EngineProxy(eng, object()),
            max_nodes=NODES, multipv=1,
        )
    except Exception as exc:
        return {**base_row, "error": f"mcts_error: {exc}",
                "max_delta": None, "q_first": None, "q_trough": None,
                "q_last": None, "q_trough_at_visits": None,
                "focal_visits": None, "total_nodes_searched": None,
                "wall_s": time.perf_counter() - t0}
    wall_s = time.perf_counter() - t0

    focal_child = next((c for c in mcts.children if c.uci == focal_uci), None)
    if focal_child is None:
        return {**base_row, "error": "focal_not_in_children",
                "max_delta": None, "q_first": None, "q_trough": None,
                "q_last": None, "q_trough_at_visits": None,
                "focal_visits": None,
                "total_nodes_searched": mcts.total_nodes_searched,
                "wall_s": wall_s}
    if focal_child.trajectory_summary is None:
        return {**base_row, "error": "no_trajectory_summary",
                "max_delta": None, "q_first": None, "q_trough": None,
                "q_last": None, "q_trough_at_visits": None,
                "focal_visits": focal_child.visits,
                "total_nodes_searched": mcts.total_nodes_searched,
                "wall_s": wall_s}

    ts = focal_child.trajectory_summary
    max_delta = max(abs(ts.q_trough - ts.q_first), abs(ts.q_trough - ts.q_last))
    return {**base_row, "error": None,
            "max_delta": max_delta,
            "q_first": ts.q_first, "q_trough": ts.q_trough, "q_last": ts.q_last,
            "q_trough_at_visits": ts.q_trough_at_visits,
            "focal_visits": focal_child.visits,
            "total_nodes_searched": mcts.total_nodes_searched,
            "wall_s": wall_s}


def _sample_positions(themes: list[str], n_per_theme: int,
                      rating_range: tuple[int, int]) -> list[dict]:
    """Stratified sample across themes; each position stamped with `_stratum`
    (which theme bucket it came from) for downstream attribution."""
    from leela_chat.eval.lichess_puzzle_db import sample_by_theme, load_corpus

    logger.info(f"loading Lichess puzzle corpus")
    table = load_corpus()
    logger.info(f"corpus loaded: {len(table):_} puzzles total")

    all_positions: list[dict] = []
    for theme in themes:
        logger.info(
            f"sampling {n_per_theme} puzzles for theme={theme!r} "
            f"rating=[{rating_range[0]}, {rating_range[1]}]"
        )
        rows = sample_by_theme(
            theme=theme, n=n_per_theme,
            rating_range=rating_range,
            seed=SAMPLE_SEED,
            table=table,
        )
        for r in rows:
            r["_stratum"] = theme
        logger.info(f"  → {len(rows)} puzzles for {theme}")
        all_positions.extend(rows)
    logger.info(
        f"total sampled: {len(all_positions)} positions across "
        f"{len(themes)} themes"
    )
    return all_positions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--themes", nargs="+", default=DEFAULT_THEMES,
        help=f"Theme buckets (one stratum per theme; default: {DEFAULT_THEMES})",
    )
    parser.add_argument(
        "--n-per-theme", type=int, default=DEFAULT_N_PER_THEME,
        help=f"Positions per theme (default: {DEFAULT_N_PER_THEME})",
    )
    parser.add_argument(
        "--min-rating", type=int, default=DEFAULT_MIN_RATING,
        help=f"Minimum puzzle rating (default: {DEFAULT_MIN_RATING})",
    )
    parser.add_argument(
        "--max-rating", type=int, default=DEFAULT_MAX_RATING,
        help=f"Maximum puzzle rating (default: {DEFAULT_MAX_RATING})",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="JSONL output path (default: /workspace/threshold_calibration_<ts>.jsonl)",
    )
    parser.add_argument(
        "--progress-every", type=int, default=100,
        help="Log progress every N measurements (default: 100)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap total measurements (testing aid; default: all sampled)",
    )
    args = parser.parse_args()

    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.out is None:
        args.out = Path(f"/workspace/threshold_calibration_{timestamp}.jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"output: {args.out}")

    # Sample positions FIRST (cheap; ~1-2 min) so we know the work
    # budget before spawning the engine.
    positions = _sample_positions(
        themes=args.themes,
        n_per_theme=args.n_per_theme,
        rating_range=(args.min_rating, args.max_rating),
    )
    if args.limit is not None:
        positions = positions[:args.limit]
        logger.info(f"limited to first {len(positions)} positions per --limit")
    total = len(positions)
    if total == 0:
        logger.error("no positions sampled; check theme + rating-range filters")
        return 2

    # Spawn ONE long-lived Lc0 (cold-per-call via fresh game=object()
    # sentinel — matches Step B verification's production-shape test).
    lc0_path = os.environ.get("LC0_BIN", "/workspace/bin/lc0")
    lc0_weights = os.environ.get(
        "LC0_WEIGHTS",
        "/workspace/weights/BT4-1024x15x32h-swa-6147500/model.pb.gz",
    )
    command = [
        lc0_path,
        "--verbose-move-stats",
        "--log-live-stats",
        f"--weights={lc0_weights}",
    ]
    logger.info("spawning Lc0 subprocess (long-lived; one per sweep)")
    eng = chess.engine.SimpleEngine.popen_uci(command)

    overall_t0 = time.perf_counter()
    n_done = 0
    n_errors = 0
    sum_max_delta = 0.0
    n_with_max_delta = 0
    try:
        with args.out.open("w") as fh:
            for i, pos in enumerate(positions, 1):
                row = _measure_one(eng, pos)
                fh.write(json.dumps(row) + "\n")
                fh.flush()  # incremental write so partial results survive
                n_done += 1
                if row.get("error"):
                    n_errors += 1
                if row.get("max_delta") is not None:
                    sum_max_delta += row["max_delta"]
                    n_with_max_delta += 1
                if i % args.progress_every == 0 or i == total:
                    elapsed = time.perf_counter() - overall_t0
                    rate = i / elapsed if elapsed > 0 else 0
                    eta_s = (total - i) / rate if rate > 0 else 0
                    mean_md = (sum_max_delta / n_with_max_delta
                               if n_with_max_delta else 0.0)
                    logger.info(
                        f"[{i:>5}/{total}] elapsed={elapsed/60:5.1f}min "
                        f"rate={rate:.2f}/s eta={eta_s/60:5.1f}min "
                        f"errors={n_errors} mean_max_delta={mean_md:.3f}"
                    )
    finally:
        logger.info("closing Lc0 subprocess")
        try:
            eng.close()
        except Exception as exc:
            logger.warning(f"engine close failed: {exc}")

    elapsed_total = time.perf_counter() - overall_t0
    logger.info(
        f"sweep done: {n_done}/{total} measurements in "
        f"{elapsed_total/3600:.2f} hours; {n_errors} errors; "
        f"mean max_delta = {sum_max_delta / max(1, n_with_max_delta):.4f}"
    )
    logger.info(f"output: {args.out}")
    print()
    print("=== Threshold calibration sweep done ===")
    print(f"  total measurements: {n_done}")
    print(f"  errors: {n_errors}")
    print(f"  mean max_delta (non-error): "
          f"{sum_max_delta / max(1, n_with_max_delta):.4f}")
    print(f"  wall: {elapsed_total/3600:.2f} hours")
    print(f"  output: {args.out}")
    print(f"  run analysis with: leela_chat/scripts/analyze_threshold_calibration.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
