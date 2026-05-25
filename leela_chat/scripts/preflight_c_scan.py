"""Pre-flight C: offline MCTS scan against the spot-check bank.

Per Chunk 7d entry sequence (plan-file ``### 7d`` section). Reads
``tests/eval/spot_check_bank.yaml``, runs ``extract_mcts_trajectory``
on each position, records observed dynamics (engine pick, focal
trajectory_summary, whether trough_explanation populates), and writes
a markdown report.

**No LLM calls.** Pod-side MCTS + lightweight BT4 + Stockfish at modest
budget (50k nodes per position). Per-position cost: ~10-30s wall clock.

For each position the scan records:
- engine_pick (bestmove_uci + bestmove_san) — used as focal for the
  decomposition run.
- focal trajectory_summary: q_first / q_last / q_trough /
  q_trough_at_visits / max_delta.
- trough_explanation: populated or None, and if populated, the
  culprit's san + q_delta_pre_to_trough.
- top-3 children by visits + their final Q (for "PUCT distribution"
  context).
- the assigned ACTUAL category based on observed dynamics:
  - refutation: focal is sharp (long pv, real q_first ≥ 0.5, no real trough)
  - redirection: trough_explanation populated
  - recovery: i_at = 0 (low q_first, rising) — currently None-returning
    but observable via q_first vs q_last delta
  - hydra: top-2 children have similar visits (within 1.5x)
  - quiet: low max_delta + no clear top candidate dominance
  - unclear: doesn't fit any of the above cleanly

Output:
- ``tests/eval/spot_check_bank.preflight_c.md`` — full markdown report
  with per-position table + bucket assignment + shortlist of positions
  for 7d's exit-gate populated-path verification.

Usage (on the pod):

    uv run python -m leela_chat.scripts.preflight_c_scan
"""
from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path

import chess
import yaml
from loguru import logger


# Default MCTS budget for the scan. 50k nodes = enough to see real
# trough dynamics + far below 7d's verdict-pass budget. Adaptive MultiPv
# (Chunk 7c.8 / ADR-0015) is applied per-position.
DEFAULT_SCAN_NODES = 50_000


def _classify_dynamics(
    focal_summary: dict | None,
    decomposition_present: bool,
    children_visits: list[int],
) -> tuple[str, str]:
    """Return (category, rationale) based on observed dynamics."""
    if focal_summary is None:
        return "unclear", "No trajectory_summary on focal child"

    q_first = focal_summary.get("q_first", 0.0)
    q_last = focal_summary.get("q_last", 0.0)
    q_trough = focal_summary.get("q_trough", 0.0)
    max_delta = max(abs(q_trough - q_first), abs(q_trough - q_last))

    # Hydra check: top-2 children visit-share ratio < 1.5x → contested.
    hydra = False
    if len(children_visits) >= 2 and children_visits[0] > 0 and children_visits[1] > 0:
        ratio = children_visits[0] / max(1, children_visits[1])
        hydra = ratio < 1.5

    # Recovery check: q_last - q_first > 0.10 AND q_first is the trough
    # (or very close to it) → focal recovered from low first impression.
    recovery = (
        (q_last - q_first > 0.10)
        and (abs(q_first - q_trough) < 0.05)
    )

    if decomposition_present:
        return "redirection", f"trough_explanation populated; max_delta={max_delta:.3f}"
    if recovery:
        return "recovery", (
            f"q_first={q_first:+.3f} → q_last={q_last:+.3f}; "
            f"focal recovered from low first impression"
        )
    if hydra:
        return "hydra", (
            f"top-2 visits ratio {children_visits[0]}:{children_visits[1]} "
            f"(< 1.5x); contested"
        )
    if max_delta > 0.10:
        # Real trough on focal but no clear culprit → refutation with
        # internal-noise trough (the q dipped but no other child took
        # the blame; PUCT was still committing to focal).
        return "refutation", (
            f"max_delta={max_delta:.3f} (real trough) but no culprit; "
            f"focal q stayed in winning bucket"
        )
    if abs(max_delta) < 0.05:
        return "quiet", f"low max_delta={max_delta:.3f}; calm position"

    return "unclear", (
        f"max_delta={max_delta:.3f}; no clear bucket fit "
        f"(not redirection / recovery / hydra / quiet)"
    )


def _scan_one(ctx, position: dict, nodes: int) -> dict:
    """Run MCTS + decomposition on one position; return observed-dynamics dict."""
    from leela_chat.extractors.mcts_trajectory import extract_mcts_trajectory
    from leela_chat.extractors.adaptive_multipv import (
        compute_adaptive_multipv, policy_entropy_normalized,
    )
    from leela_chat.extractors.network_inference import extract_network_inference
    from leela_chat.extractors.trajectory_decomposition import (
        extract_trajectory_decomposition,
    )

    fen = position["fen"]
    pid = position["id"]
    board = chess.Board(fen)

    # Network forward pass for adaptive multipv input.
    leela = extract_network_inference(board, model=ctx.model, top_n=5)
    pcts = [m.policy_pct for m in leela.top_moves]
    adaptive_mpv = compute_adaptive_multipv(
        top_prob=(leela.top_moves[0].policy_pct / 100.0) if leela.top_moves else 0.0,
        policy_entropy=policy_entropy_normalized(pcts),
        mode="deep",
    )

    # Main MCTS scan.
    t0 = time.perf_counter()
    mcts = extract_mcts_trajectory(
        board, engine=ctx.lc0, max_nodes=nodes, multipv=adaptive_mpv,
    )
    scan_s = time.perf_counter() - t0

    # Engine pick = focal for the decomposition.
    focal_uci = mcts.bestmove_uci
    if focal_uci is None:
        return {
            "id": pid,
            "scan_s": scan_s,
            "error": "MCTS returned no bestmove",
            "category": "unclear",
            "rationale": "no engine pick",
        }
    focal_san = mcts.bestmove_san

    # Find focal child + extract trajectory_summary.
    focal_child = next((c for c in mcts.children if c.uci == focal_uci), None)
    focal_summary_dict = None
    if focal_child is not None and focal_child.trajectory_summary is not None:
        ts = focal_child.trajectory_summary
        focal_summary_dict = {
            "q_first": ts.q_first,
            "q_last": ts.q_last,
            "q_peak": ts.q_peak,
            "q_peak_at_visits": ts.q_peak_at_visits,
            "q_trough": ts.q_trough,
            "q_trough_at_visits": ts.q_trough_at_visits,
            "block_count": ts.block_count,
        }

    # Run decomposition (returns TrajectoryDecomposition or None).
    decomp = extract_trajectory_decomposition(mcts, focal_uci=focal_uci)
    decomp_present = decomp is not None
    decomp_summary = None
    if decomp is not None:
        decomp_summary = {
            "candidate_san": decomp.san,
            "candidate_uci": decomp.uci,
            "q_delta_pre_to_trough": decomp.q_delta_pre_to_trough,
            "snapshots": [
                {"phase": s.phase, "visits": s.visits, "q": s.q,
                 "nodes_searched": s.nodes_searched}
                for s in decomp.snapshots
            ],
            "pv_san_first_5": decomp.pv_san[:5],
        }

    # Top-3 children by visits.
    top3 = [
        {"san": c.san, "uci": c.uci, "visits": c.visits, "q": c.q}
        for c in sorted(mcts.children, key=lambda c: c.visits, reverse=True)[:3]
    ]

    # Classify.
    children_visits = [c["visits"] for c in top3]
    category, rationale = _classify_dynamics(
        focal_summary_dict, decomp_present, children_visits,
    )

    return {
        "id": pid,
        "scan_s": scan_s,
        "adaptive_multipv": adaptive_mpv,
        "engine_pick_uci": focal_uci,
        "engine_pick_san": focal_san,
        "focal_summary": focal_summary_dict,
        "trough_explanation_present": decomp_present,
        "trough_explanation": decomp_summary,
        "top3_children": top3,
        "observed_category": category,
        "observed_rationale": rationale,
        "total_nodes_searched": mcts.total_nodes_searched,
    }


def _render_markdown(results: list[dict], bank_meta: dict) -> str:
    """Format the scan results as a markdown report."""
    lines = []
    lines.append("# Pre-flight C scan results")
    lines.append("")
    lines.append(f"Captured: {datetime.datetime.now(datetime.UTC).isoformat()}")
    lines.append(f"Scan budget: {bank_meta['nodes']} nodes per position.")
    lines.append(f"Positions scanned: {len(results)}.")
    lines.append("")

    # === Per-position table ===
    lines.append("## Per-position observed dynamics")
    lines.append("")
    lines.append("| id | engine pick | q_first | q_trough | q_last | max_delta | trough_expl? | observed category | scan_s |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if "error" in r:
            lines.append(
                f"| {r['id']} | (error) | — | — | — | — | — | {r['observed_category']} | {r.get('scan_s', 0):.1f}s |"
            )
            continue
        fs = r.get("focal_summary") or {}
        q_first = fs.get("q_first")
        q_trough = fs.get("q_trough")
        q_last = fs.get("q_last")
        if q_first is not None and q_last is not None and q_trough is not None:
            max_delta = max(abs(q_trough - q_first), abs(q_trough - q_last))
        else:
            max_delta = None
        te = "YES" if r["trough_explanation_present"] else "no"
        q_first_s = f"{q_first:+.3f}" if q_first is not None else "—"
        q_trough_s = f"{q_trough:+.3f}" if q_trough is not None else "—"
        q_last_s = f"{q_last:+.3f}" if q_last is not None else "—"
        mds = f"{max_delta:.3f}" if max_delta is not None else "—"
        lines.append(
            f"| {r['id']} | {r['engine_pick_san']} ({r['engine_pick_uci']}) | "
            f"{q_first_s} | {q_trough_s} | {q_last_s} | {mds} | {te} | "
            f"{r['observed_category']} | {r['scan_s']:.1f}s |"
        )
    lines.append("")

    # === Hypothesis vs observation ===
    lines.append("## Hypothesis vs observation")
    lines.append("")
    lines.append("| id | expected | observed | match? | rationale |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        if "error" in r:
            continue
        # Find expected_category from the bank meta.
        expected = next(
            (p["expected_category"] for p in bank_meta["positions"] if p["id"] == r["id"]),
            "?",
        )
        match = "✓" if expected == r["observed_category"] else "✗"
        lines.append(
            f"| {r['id']} | {expected} | {r['observed_category']} | "
            f"{match} | {r['observed_rationale']} |"
        )
    lines.append("")

    # === Bucket counts ===
    counts = {}
    for r in results:
        if "error" in r:
            continue
        c = r["observed_category"]
        counts[c] = counts.get(c, 0) + 1
    lines.append("## Observed bucket counts")
    lines.append("")
    for c in ("refutation", "redirection", "recovery", "hydra", "quiet", "unclear"):
        lines.append(f"- **{c}**: {counts.get(c, 0)}")
    lines.append("")

    # === 7d exit-gate shortlist ===
    shortlist = [r for r in results if r.get("trough_explanation_present")]
    lines.append("## 7d exit-gate shortlist (trough_explanation populated)")
    lines.append("")
    if shortlist:
        lines.append(f"**{len(shortlist)} position(s) produce populated trough_explanation** — ready for v6 prompt verification at 7d exit gate.")
        lines.append("")
        for r in shortlist:
            decomp = r["trough_explanation"]
            lines.append(
                f"- **{r['id']}** — engine pick: {r['engine_pick_san']} ({r['engine_pick_uci']}); "
                f"culprit: {decomp['candidate_san']} ({decomp['candidate_uci']}); "
                f"q_delta_pre_to_trough: {decomp['q_delta_pre_to_trough']:+.3f}; "
                f"culprit pv (first 5): {decomp['pv_san_first_5']}"
            )
    else:
        lines.append("**No positions produced populated trough_explanation at 50k nodes.** Options for the 7d exit gate:")
        lines.append("")
        lines.append("- Re-run at 100k nodes (7d verdict-pass budget) — deeper search may surface real troughs.")
        lines.append("- Lower TROUGH_DELTA_THRESHOLD temporarily for one capture (manufacture-a-trough path).")
        lines.append("- Add more positions specifically targeting the redirection dynamic.")
    lines.append("")

    # === Recovery candidates (algorithm currently None-returning) ===
    recovery = [r for r in results if r["observed_category"] == "recovery"]
    lines.append("## Recovery candidates (algorithm currently None-returning per i_at=0 gate)")
    lines.append("")
    if recovery:
        lines.append(f"**{len(recovery)} position(s) exhibit recovery dynamics** — focal q rose from low first impression. These would benefit from a future ``recovery_explanation`` companion field (docstring noted in trajectory_decomposition.py per 7d step 2b).")
        lines.append("")
        for r in recovery:
            fs = r["focal_summary"]
            lines.append(
                f"- **{r['id']}** — engine pick: {r['engine_pick_san']}; "
                f"q_first={fs['q_first']:+.3f} → q_last={fs['q_last']:+.3f} "
                f"(Δ {fs['q_last'] - fs['q_first']:+.3f})"
            )
    else:
        lines.append("No clear recovery cases at 50k nodes.")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bank", type=Path,
        default=Path("tests/eval/spot_check_bank.yaml"),
        help="Position bank YAML (default: tests/eval/spot_check_bank.yaml)",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("tests/eval/spot_check_bank.preflight_c.md"),
        help="Markdown report output path (default: alongside the bank)",
    )
    parser.add_argument(
        "--nodes", type=int, default=DEFAULT_SCAN_NODES,
        help=f"MCTS budget per position (default: {DEFAULT_SCAN_NODES:_})",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap scan to first N positions (for quick dry-run; default: all)",
    )
    args = parser.parse_args()

    bank = yaml.safe_load(args.bank.read_text())
    positions = bank["positions"]
    if args.limit:
        positions = positions[: args.limit]

    logger.info(f"scanning {len(positions)} positions at {args.nodes:_} nodes each")
    logger.info(f"output: {args.out}")

    # Build LLM context (no LLMClient needed since no LLM calls; only BT4
    # + Lc0 + Stockfish), but the existing _build_context is the simplest
    # path that gets all three engines ready.
    from leela_chat.cli.__main__ import _build_context, _close_context
    from leela_chat.llm.client import LLMMode
    ctx = _build_context(mode=LLMMode.PROD, verbose=False)

    results = []
    try:
        for i, pos in enumerate(positions, 1):
            logger.info(f"  [{i}/{len(positions)}] {pos['id']}")
            try:
                r = _scan_one(ctx, pos, args.nodes)
                results.append(r)
                logger.info(
                    f"    engine: {r.get('engine_pick_san')} | "
                    f"category: {r['observed_category']} | "
                    f"trough_expl: {r.get('trough_explanation_present')} | "
                    f"{r.get('scan_s', 0):.1f}s"
                )
            except Exception as exc:
                logger.exception(f"    scan failed: {exc}")
                results.append({
                    "id": pos["id"],
                    "error": str(exc),
                    "observed_category": "unclear",
                    "observed_rationale": f"scan failed: {type(exc).__name__}",
                })
    finally:
        logger.info("closing engines")
        _close_context(ctx)

    bank_meta = {"nodes": args.nodes, "positions": bank["positions"]}
    md = _render_markdown(results, bank_meta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    logger.info(f"wrote report to {args.out}")

    # Quick stdout summary so the user sees the headline counts.
    print()
    print("=== Pre-flight C scan summary ===")
    print(f"positions scanned: {len(results)}")
    counts = {}
    for r in results:
        c = r["observed_category"]
        counts[c] = counts.get(c, 0) + 1
    for c in ("refutation", "redirection", "recovery", "hydra", "quiet", "unclear"):
        print(f"  {c:>12}: {counts.get(c, 0)}")
    trough_count = sum(1 for r in results if r.get("trough_explanation_present"))
    print(f"  populated trough_explanation: {trough_count}")
    print(f"  report: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
