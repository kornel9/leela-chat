"""Analyze the threshold-calibration sweep JSONL.

Companion to `threshold_calibration_sweep.py`. Reads the sweep's JSONL
(one row per puzzle, with cold-engine max_delta + per-stratum metadata)
and produces a markdown report that surfaces the population distribution
of `max_delta` so `TROUGH_DELTA_THRESHOLD` can be empirically anchored.

Outputs:
- Per-theme + overall percentile breakdown of max_delta.
- Threshold-fire-rate at T = {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50}.
- Bimodality check (mode counting via histogram).
- Rating-band breakdown (does max_delta vary with puzzle rating within
  the hard band?).
- Error rate + error-kind breakdown.

Stdlib-only (no numpy / scipy / matplotlib). All math done by hand
because the dataset is bounded (~5000 rows) and the report stays
plain-markdown so it renders anywhere.

Usage:

    uv run python -m leela_chat.scripts.analyze_threshold_calibration \\
        --jsonl /workspace/threshold_calibration_5k.jsonl

    # Or with explicit output path:
    uv run python -m leela_chat.scripts.analyze_threshold_calibration \\
        --jsonl <path> --out tests/eval/threshold_calibration_5k.md
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]
RATING_BANDS = [
    (2200, 2399, "2200-2399"),
    (2400, 2599, "2400-2599"),
    (2600, 2799, "2600-2799"),
    (2800, 2999, "2800-2999"),
    (3000, 3500, "3000+"),
]
# Bimodality detection: simple 20-bin histogram of max_delta in [0, 1],
# count "modes" as local maxima separated by ≥1 bin with a strictly-
# lower count. Reported as a sanity check; the report shows the
# histogram so the user can eyeball regardless.
HIST_BINS = 20
HIST_RANGE = (0.0, 1.0)


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolation percentile (matches numpy.percentile default).

    sorted_vals must be pre-sorted ascending.
    """
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def _summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": float("nan"), "stdev": float("nan"),
                "min": float("nan"), "max": float("nan"),
                "percentiles": {p: float("nan") for p in PERCENTILES}}
    sorted_vals = sorted(values)
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "percentiles": {p: _percentile(sorted_vals, p) for p in PERCENTILES},
    }


def _fire_rates(values: list[float], thresholds: list[float]) -> dict:
    """For each threshold T, return fraction of values strictly > T."""
    if not values:
        return {T: float("nan") for T in thresholds}
    n = len(values)
    return {T: sum(1 for v in values if v > T) / n for T in thresholds}


def _histogram(values: list[float], bins: int = HIST_BINS,
               lo: float = HIST_RANGE[0], hi: float = HIST_RANGE[1]) -> list[int]:
    """Bucket counts; out-of-range values clipped into the nearest bin."""
    counts = [0] * bins
    if not values:
        return counts
    width = (hi - lo) / bins
    for v in values:
        idx = int((v - lo) / width)
        idx = max(0, min(bins - 1, idx))
        counts[idx] += 1
    return counts


def _count_modes(hist: list[int]) -> int:
    """Strict local-maxima count; smoothing-free for clarity."""
    if not hist:
        return 0
    modes = 0
    for i, c in enumerate(hist):
        left = hist[i - 1] if i > 0 else -1
        right = hist[i + 1] if i < len(hist) - 1 else -1
        if c > left and c > right and c > 0:
            modes += 1
    return modes


def _render_hist_ascii(hist: list[int], lo: float = 0.0, hi: float = 1.0,
                       width: int = 50) -> list[str]:
    """Render histogram as a small ASCII block for the markdown report."""
    if not hist or max(hist) == 0:
        return ["(empty histogram)"]
    width_per_bin = (hi - lo) / len(hist)
    peak = max(hist)
    out = []
    for i, c in enumerate(hist):
        bin_lo = lo + i * width_per_bin
        bar_len = int((c / peak) * width)
        bar = "█" * bar_len
        out.append(f"  [{bin_lo:.2f}-{bin_lo + width_per_bin:.2f})  {bar} {c}")
    return out


def _render_summary_table(name: str, summary: dict) -> list[str]:
    out = []
    out.append(f"### {name}")
    out.append("")
    out.append(f"- **n**: {summary['n']}")
    out.append(f"- **mean**: {summary['mean']:.4f}")
    out.append(f"- **stdev**: {summary['stdev']:.4f}")
    out.append(f"- **min/max**: {summary['min']:.4f} / {summary['max']:.4f}")
    out.append("")
    out.append("| Percentile | max_delta |")
    out.append("|---|---|")
    for p in PERCENTILES:
        out.append(f"| p{p:>2} | {summary['percentiles'][p]:.4f} |")
    return out


def _render_fire_rate_table(by_label: dict[str, dict[float, float]]) -> list[str]:
    """Render a fire-rate matrix: row per threshold, column per label."""
    out = []
    labels = list(by_label.keys())
    header = "| Threshold T | " + " | ".join(labels) + " |"
    sep = "|---" * (len(labels) + 1) + "|"
    out.append(header)
    out.append(sep)
    for T in THRESHOLDS:
        cells = [f"{by_label[L][T]*100:5.1f}%" if not math.isnan(by_label[L][T])
                 else "  n/a" for L in labels]
        out.append(f"| > {T:.2f} | " + " | ".join(cells) + " |")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl", type=Path, required=True,
        help="Path to the calibration sweep JSONL",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("tests/eval/threshold_calibration_5k.md"),
        help="Markdown report output (default: tests/eval/threshold_calibration_5k.md)",
    )
    args = parser.parse_args()

    if not args.jsonl.exists():
        print(f"ERROR: {args.jsonl} does not exist")
        return 2

    # Load.
    rows: list[dict] = []
    with args.jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    total = len(rows)
    if total == 0:
        print(f"ERROR: {args.jsonl} is empty")
        return 2

    # Split: successful measurements vs errors.
    ok_rows = [r for r in rows if r.get("error") is None
               and r.get("max_delta") is not None]
    err_rows = [r for r in rows if r.get("error") is not None
                or r.get("max_delta") is None]
    n_ok = len(ok_rows)
    n_err = len(err_rows)

    # Per-stratum (theme).
    by_stratum: dict[str, list[float]] = defaultdict(list)
    for r in ok_rows:
        by_stratum[r["stratum"]].append(r["max_delta"])

    # Per rating band.
    by_band: dict[str, list[float]] = defaultdict(list)
    for r in ok_rows:
        rating = r.get("rating", 0)
        band_label = None
        for lo, hi, label in RATING_BANDS:
            if lo <= rating <= hi:
                band_label = label
                break
        if band_label:
            by_band[band_label].append(r["max_delta"])

    # All max_deltas (overall distribution).
    all_md = [r["max_delta"] for r in ok_rows]

    # Compute summaries.
    overall_summary = _summarize(all_md)
    per_stratum_summary = {k: _summarize(v) for k, v in by_stratum.items()}
    per_band_summary = {k: _summarize(v) for k, v in by_band.items()}

    # Fire rates.
    overall_fire = _fire_rates(all_md, THRESHOLDS)
    per_stratum_fire = {k: _fire_rates(v, THRESHOLDS) for k, v in by_stratum.items()}
    per_band_fire = {k: _fire_rates(v, THRESHOLDS) for k, v in by_band.items()}

    # Histogram + bimodality check (overall + per stratum).
    overall_hist = _histogram(all_md)
    overall_modes = _count_modes(overall_hist)
    per_stratum_hist = {k: _histogram(v) for k, v in by_stratum.items()}
    per_stratum_modes = {k: _count_modes(h) for k, h in per_stratum_hist.items()}

    # Error breakdown.
    error_counter = Counter(r.get("error", "(missing)") for r in err_rows)

    # === Build markdown report ===
    lines: list[str] = []
    lines.append("# Threshold calibration sweep — analysis report")
    lines.append("")
    lines.append(f"Source: `{args.jsonl}` ({total} JSONL rows)")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total measurements: **{total}**")
    lines.append(f"- Successful: **{n_ok}** ({n_ok/total*100:.1f}%)")
    lines.append(f"- Errors: **{n_err}** ({n_err/total*100:.1f}%)")
    lines.append(f"- Strata observed: {sorted(by_stratum.keys())}")
    lines.append(f"- Rating range observed: "
                 f"{min(r['rating'] for r in ok_rows)} to "
                 f"{max(r['rating'] for r in ok_rows)}")
    lines.append("")
    if err_rows:
        lines.append("### Error breakdown")
        lines.append("")
        for kind, count in error_counter.most_common():
            lines.append(f"- `{kind}`: {count} ({count/total*100:.2f}%)")
        lines.append("")

    lines.append("## Overall max_delta distribution")
    lines.append("")
    lines.extend(_render_summary_table("Overall", overall_summary))
    lines.append("")
    lines.append(f"**Histogram** (20 bins over [0.0, 1.0]; modes detected: "
                 f"{overall_modes}):")
    lines.append("```")
    lines.extend(_render_hist_ascii(overall_hist))
    lines.append("```")
    lines.append("")
    if overall_modes == 1:
        lines.append("→ **Unimodal** distribution: threshold choice is a "
                     "smooth gradient; pick the percentile that matches the "
                     "desired fire-rate per the table below.")
    elif overall_modes >= 2:
        lines.append(f"→ **Bimodal/multimodal** distribution ({overall_modes} "
                     "modes detected): consider placing the threshold in the "
                     "valley between modes for cleanest classifier behavior. "
                     "Inspect histogram visually before trusting the mode count.")
    else:
        lines.append("→ No clear mode structure detected.")
    lines.append("")

    lines.append("## Threshold fire rates")
    lines.append("")
    lines.append("Fraction of positions where `max_delta > T` — this is the rate "
                 "at which `extract_trajectory_decomposition` AND the validator's "
                 "`incomplete_trough_citation` check would engage at each "
                 "candidate threshold.")
    lines.append("")
    lines.append("### Overall")
    lines.append("")
    lines.append("| Threshold T | Fire rate |")
    lines.append("|---|---|")
    for T in THRESHOLDS:
        lines.append(f"| > {T:.2f} | {overall_fire[T]*100:.1f}% |")
    lines.append("")
    lines.append(f"_Reference: current `TROUGH_DELTA_THRESHOLD = 0.10`._")
    lines.append("")

    lines.append("### Per-theme")
    lines.append("")
    lines.extend(_render_fire_rate_table(per_stratum_fire))
    lines.append("")

    lines.append("### Per rating-band")
    lines.append("")
    if per_band_fire:
        lines.extend(_render_fire_rate_table(per_band_fire))
    else:
        lines.append("(no rating-band data)")
    lines.append("")

    lines.append("## Per-theme distribution")
    lines.append("")
    for theme in sorted(per_stratum_summary.keys()):
        lines.extend(_render_summary_table(theme, per_stratum_summary[theme]))
        lines.append("")
        lines.append(f"**Histogram** (modes: {per_stratum_modes[theme]}):")
        lines.append("```")
        lines.extend(_render_hist_ascii(per_stratum_hist[theme]))
        lines.append("```")
        lines.append("")

    lines.append("## Per rating-band distribution")
    lines.append("")
    for band_label in [b[2] for b in RATING_BANDS]:
        if band_label not in per_band_summary:
            continue
        lines.extend(_render_summary_table(band_label, per_band_summary[band_label]))
        lines.append("")

    lines.append("## Threshold-tuning guidance")
    lines.append("")
    lines.append("Suggested threshold choices, ranked by ground each one stands on:")
    lines.append("")
    lines.append(f"- **T = p50 ({overall_summary['percentiles'][50]:.3f})** — fires on the upper-half of positions. Useful if production wants \"trough_explanation on positions where the trough is at-least-typical.\"")
    lines.append(f"- **T = p75 ({overall_summary['percentiles'][75]:.3f})** — fires on the top quartile. Picks out positions with meaningfully-larger-than-typical troughs.")
    lines.append(f"- **T = p90 ({overall_summary['percentiles'][90]:.3f})** — fires on the top decile. Reserve trough_explanation for clearly-exceptional troughs.")
    lines.append(f"- **T = current 0.10** — fires on **{overall_fire[0.10]*100:.1f}%** of positions. Compare to the percentile-based options above to decide whether 0.10 is sensibly placed for the hard-puzzle population, or whether it should move.")
    lines.append("")
    lines.append(f"**Note on stratification.** This sweep is biased AGGRESSIVELY toward hard puzzles (rating 2200-3500), so the population is NOT a uniform cross-section of \"all chess positions.\" The threshold inferred here is calibrated to the question that motivates trough_explanation: \"when production analyzes a hard / interesting position, what fraction warrant WHY-narration?\" Easy positions (rating < 2000) will have shallower troughs and the same threshold would fire less often there. For Phase 3 probe training, the hard-puzzle calibration is the right reference; for general chat-style position analysis, expect production fire-rates to be lower than the rates in this report.")
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"wrote analysis to {args.out}")
    print()
    print("=== Quick summary ===")
    print(f"  n: {n_ok} ok / {n_err} err / {total} total")
    print(f"  overall mean max_delta: {overall_summary['mean']:.4f}")
    print(f"  overall percentiles: "
          f"p25={overall_summary['percentiles'][25]:.3f}, "
          f"p50={overall_summary['percentiles'][50]:.3f}, "
          f"p75={overall_summary['percentiles'][75]:.3f}, "
          f"p90={overall_summary['percentiles'][90]:.3f}")
    print(f"  fire rate at T=0.10: {overall_fire[0.10]*100:.1f}%")
    print(f"  modes (overall histogram): {overall_modes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
