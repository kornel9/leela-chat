"""Step A — MCTS trajectory-variance experiment.

Per 7d Step 4 R2 retro discussion: Pre-flight C and R2 both ran the same
positions at 50k nodes but produced wildly different max_delta values
(user_06: 0.551 vs 0.100; user_08: 0.868 vs 0.002). Hypothesis: Lc0's
default multi-threaded MCTS produces non-deterministic intra-search
Q-trajectories even at the same FEN + budget. This script measures the
distribution empirically.

Design:
- For each of 2 positions (user_06_kh8_idea + user_08_short_queen_sac),
  run 3 fresh-engine trials of extract_mcts_trajectory at 50k nodes.
- Spawn fresh Lc0 engine per trial (no NN cache contamination across
  trials — per CLAUDE.md §8 NN cache pitfall to be codified).
- Capture per-trial: max_delta, q_first/q_trough/q_last,
  q_trough_at_visits, total_nodes_searched, wall-clock.
- Combined with Pre-flight C (1 measurement) + R2 (1 measurement) =
  5 data points per position.

NO LLM calls; pure Lc0 MCTS measurement. Cost: $0; pod wall-clock
~3-4 min (6 trials × ~30s each = ~3 min + engine spawn overhead).

Usage (on the pod):

    uv run python -m leela_chat.scripts.variance_experiment
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import chess
from loguru import logger


POSITIONS = [
    {
        "id": "user_06_kh8_idea",
        "fen": "2r2rk1/p5bp/2p5/4p1B1/N3Pn2/P2P4/2P2PpK/RR6 b - - 1 28",
        "focal_uci": "g8h8",
        "prior_50k_runs": {
            "preflight_c_max_delta": 0.551,
            "r2_max_delta": 0.100,
        },
    },
    {
        "id": "user_08_short_queen_sac",
        "fen": "r1q2rk1/pb1n1pnp/1p1N2pB/2ppP1b1/8/6PQ/PP3PBP/R2R2K1 b - - 1 21",
        "focal_uci": "d7e5",
        "prior_50k_runs": {
            "preflight_c_max_delta": 0.868,
            "r2_max_delta": 0.002,
        },
    },
]
NODES = 50_000
TRIALS_PER_POSITION = 3


def _spawn_engine(lc0_path: str, lc0_weights: str):
    from leela_chat.engine.lc0_engine import Lc0Engine
    return Lc0Engine(binary=lc0_path, weights=lc0_weights)


def _measure_one(eng, board, focal_uci):
    """Run extract_mcts_trajectory + capture focal trajectory dynamics."""
    from leela_chat.extractors.mcts_trajectory import extract_mcts_trajectory

    t0 = time.perf_counter()
    mcts = extract_mcts_trajectory(board, engine=eng, max_nodes=NODES, multipv=1)
    wall_s = time.perf_counter() - t0

    focal_child = next((c for c in mcts.children if c.uci == focal_uci), None)
    if focal_child is None or focal_child.trajectory_summary is None:
        return {
            "wall_s": wall_s,
            "total_nodes_searched": mcts.total_nodes_searched,
            "bestmove": mcts.bestmove_san,
            "focal_visits": None,
            "q_first": None,
            "q_trough": None,
            "q_last": None,
            "q_trough_at_visits": None,
            "max_delta": None,
        }
    ts = focal_child.trajectory_summary
    max_delta = max(abs(ts.q_trough - ts.q_first), abs(ts.q_trough - ts.q_last))
    return {
        "wall_s": wall_s,
        "total_nodes_searched": mcts.total_nodes_searched,
        "bestmove": mcts.bestmove_san,
        "focal_visits": focal_child.visits,
        "q_first": ts.q_first,
        "q_trough": ts.q_trough,
        "q_last": ts.q_last,
        "q_trough_at_visits": ts.q_trough_at_visits,
        "max_delta": max_delta,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials", type=int, default=TRIALS_PER_POSITION,
        help=f"Trials per position (default: {TRIALS_PER_POSITION})",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("tests/eval/variance_experiment.md"),
        help="Markdown report output path",
    )
    parser.add_argument(
        "--no-warmup", action="store_true",
        help="Skip the 1k-node warmup before the 50k measurement. Tests "
             "hypothesis that the warmup is what flattens max_delta — see "
             "the Pre-flight-C-clean vs Step-A-warmup contradiction.",
    )
    args = parser.parse_args()

    lc0_path = os.environ.get("LC0_BIN", "/workspace/bin/lc0")
    lc0_weights = os.environ.get(
        "LC0_WEIGHTS",
        "/workspace/weights/BT4-1024x15x32h-swa-6147500/model.pb.gz",
    )

    logger.info(
        f"variance experiment: {len(POSITIONS)} positions × {args.trials} trials × {NODES:_} nodes; "
        f"fresh engine per trial (no cache contamination)"
    )

    results = {}
    for pos in POSITIONS:
        results[pos["id"]] = []
        board = chess.Board(pos["fen"])
        for trial in range(args.trials):
            logger.info(f"  {pos['id']} trial {trial+1}/{args.trials}: spawning fresh engine")
            eng = _spawn_engine(lc0_path, lc0_weights)
            try:
                # Tiny warmup (1k nodes) so backend + weights load isn't
                # in the measured time. NOTE: skipping warmup (--no-warmup)
                # tests the hypothesis that warmup itself shifts the
                # measured trajectory (warmup primes Lc0's NN cache → first
                # log-live-stats emit happens after more visits → q_first
                # is closer to q_last → smaller max_delta).
                from leela_chat.extractors.mcts_trajectory import extract_mcts_trajectory
                if not args.no_warmup:
                    _ = extract_mcts_trajectory(board, engine=eng, max_nodes=1_000, multipv=1)
                r = _measure_one(eng, board, pos["focal_uci"])
                results[pos["id"]].append(r)
                logger.info(
                    f"    -> max_delta={r['max_delta']:.3f}  "
                    f"q_first={r['q_first']:+.3f}  q_trough={r['q_trough']:+.3f}  q_last={r['q_last']:+.3f}  "
                    f"visits={r['focal_visits']}  nodes={r['total_nodes_searched']:_}  "
                    f"wall={r['wall_s']:.1f}s"
                )
            finally:
                try:
                    eng.close()
                except Exception as exc:
                    logger.warning(f"engine close failed: {exc}")

    # Build markdown report.
    lines = []
    lines.append("# Step A — MCTS trajectory-variance experiment")
    lines.append("")
    lines.append(f"Captured: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    lines.append(f"Trials per position: **{args.trials}** (fresh engine per trial; no cache contamination)")
    lines.append(f"Budget per trial: **{NODES:_}** nodes")
    lines.append("")
    for pos in POSITIONS:
        lines.append(f"## {pos['id']} (focal: {pos['focal_uci']})")
        lines.append("")
        lines.append("| Trial | max_delta | q_first | q_trough | q_last | focal visits | nodes_searched | wall (s) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(results[pos["id"]], 1):
            if r["max_delta"] is None:
                lines.append(f"| {i} | — | — | — | — | — | {r['total_nodes_searched']:_} | {r['wall_s']:.1f} |")
            else:
                lines.append(
                    f"| {i} | {r['max_delta']:.3f} | {r['q_first']:+.3f} | "
                    f"{r['q_trough']:+.3f} | {r['q_last']:+.3f} | "
                    f"{r['focal_visits']:_} | {r['total_nodes_searched']:_} | {r['wall_s']:.1f} |"
                )
        # Prior measurements for context.
        prior = pos["prior_50k_runs"]
        lines.append(f"| (prior: Pre-flight C) | **{prior['preflight_c_max_delta']:.3f}** | | | | | | |")
        lines.append(f"| (prior: R2 capture)   | **{prior['r2_max_delta']:.3f}** | | | | | | |")
        # Variance stats across all 5 measurements.
        all_deltas = [r["max_delta"] for r in results[pos["id"]] if r["max_delta"] is not None]
        all_deltas += [prior["preflight_c_max_delta"], prior["r2_max_delta"]]
        if all_deltas:
            mn = min(all_deltas)
            mx = max(all_deltas)
            lines.append("")
            lines.append(
                f"**Range across all {len(all_deltas)} measurements at 50k nodes:** "
                f"min={mn:.3f}, max={mx:.3f}, **spread={mx - mn:.3f}**"
            )
            lines.append(
                f"**Threshold-firing rate:** {sum(1 for d in all_deltas if d > 0.10)} of {len(all_deltas)} runs would fire trough_explanation (max_delta > 0.10)"
            )
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("If spread is large (>0.3) across nominally-identical runs → MCTS variance is the dominant factor; trough_explanation reliability is structurally bounded by MCTS stochasticity. Investigate Lc0 determinism flags (Step B).")
    lines.append("")
    lines.append("If spread is small (<0.1) → variance isn't the explanation for the Pre-flight C vs R2 mismatch; the gap was likely caused by cache contamination or environmental factors (e.g., GPU thermal state). Investigate Pre-flight C scan methodology (Step D).")
    lines.append("")
    lines.append("If spread is moderate (0.1-0.3) → variance is real but not crushing; threshold tuning (0.10 → 0.05 or similar) + multi-trial averaging could rescue the feature reliability.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    logger.info(f"wrote report to {args.out}")

    # Quick stdout summary.
    print()
    print("=== Variance experiment summary ===")
    for pos in POSITIONS:
        deltas = [r["max_delta"] for r in results[pos["id"]] if r["max_delta"] is not None]
        all_deltas = deltas + [pos["prior_50k_runs"]["preflight_c_max_delta"], pos["prior_50k_runs"]["r2_max_delta"]]
        if all_deltas:
            print(
                f"  {pos['id']:>28}: this run deltas {[f'{d:.3f}' for d in deltas]} | "
                f"all 5 deltas: min={min(all_deltas):.3f} max={max(all_deltas):.3f} spread={max(all_deltas)-min(all_deltas):.3f}"
            )
    print(f"  report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
