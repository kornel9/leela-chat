"""Analyze the controlled-sample smoke battery JSONL for 7c.5.E.

Decomposes per-call input_tokens into approximate components:
- system_prompt_tokens (approx via char-count proxy ~4 chars/token)
- findings_json_tokens (input_tokens - system_prompt_tokens - small_overhead)

Computes per-handler-per-mode median + p25/p75 for input_tokens,
output_tokens, translate_s, handler_s. Compares against the 7b.2.B
baseline (Handler 3 v3: in=21573, translate_s=10.23s).

Output is a markdown block suitable for direct copy-paste into
PHASE_NOTES under the 7c.5.E retrospective entry.

Usage:

    uv run python -m leela_chat.scripts.analyze_token_baseline \\
        --jsonl logs/smoke_battery_<YYYYMMDDTHHMMSSZ>.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

PROMPT_FILES = {
    ("what_is_leela_seeing", "quick"): "leela_chat/llm/prompts/what_is_leela_seeing/v6.md",
    ("what_is_leela_seeing", "deep"): "leela_chat/llm/prompts/what_is_leela_seeing/v8.md",
    ("why_this_move", "quick"): "leela_chat/llm/prompts/why_this_move/v3.md",
    ("why_this_move", "deep"): "leela_chat/llm/prompts/why_this_move/v5.md",
    ("did_you_consider", "deep"): "leela_chat/llm/prompts/did_you_consider/v4.md",
}

# Approximate Claude/cl100k token cost: 1 token ~ 4 chars for English/JSON-y
# input. Real Anthropic counter would be more accurate; this is a proxy good
# enough for the decomposition signal.
CHARS_PER_TOKEN = 4.0


def _approx_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if p <= 0:
        return s[0]
    if p >= 100:
        return s[-1]
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    if f >= len(s) - 1:
        return s[-1]
    return s[f] + (s[f + 1] - s[f]) * (k - f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True, type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("source") not in ("DRY_RUN", "CRASH", "circuit_breaker_sentinel")]

    # Compute prompt-file token approximations.
    prompt_token_approx: dict[tuple, int] = {}
    for (handler, mode), path in PROMPT_FILES.items():
        text = Path(path).read_text()
        prompt_token_approx[(handler, mode)] = _approx_tokens(text)

    # Group rows by (handler, mode).
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        handler = r.get("handler")
        mode = r.get("mode")
        if handler and mode:
            cells[(handler, mode)].append(r)

    print("# 7c.5.E controlled-sample measurement")
    print()
    print(f"JSONL: `{args.jsonl.name}`")
    print(f"Total rows: {len(rows)} (after filtering DRY_RUN/CRASH/sentinel)")
    print()
    print("## Per-handler-per-mode statistics (current state: v6/v8/v3/v5/v4 prompts)")
    print()
    print("| Handler | Mode | n | in_tokens median | in_tokens p25-p75 | out_tokens median | translate_s median | handler_s median | prompt_token_approx |")
    print("|---|---|---|---|---|---|---|---|---|")
    for key in sorted(cells.keys()):
        handler, mode = key
        cell_rows = cells[key]
        n = len(cell_rows)
        if n == 0:
            continue
        in_toks = [r.get("input_tokens", 0) for r in cell_rows]
        out_toks = [r.get("output_tokens", 0) for r in cell_rows]
        translate_s = [r.get("translate_s", 0.0) for r in cell_rows]
        handler_s = [r.get("handler_s", 0.0) for r in cell_rows]
        prompt_t = prompt_token_approx.get(key, 0)
        print(
            f"| {handler} | {mode} | {n} | "
            f"{int(statistics.median(in_toks))} | "
            f"{int(_percentile(in_toks, 25))}-{int(_percentile(in_toks, 75))} | "
            f"{int(statistics.median(out_toks))} | "
            f"{statistics.median(translate_s):.2f} | "
            f"{statistics.median(handler_s):.2f} | "
            f"{prompt_t} |"
        )
    print()

    print("## Decomposition: findings_json_tokens ≈ in_tokens − prompt_approx − ~200 framing")
    print()
    print("| Handler | Mode | in_tokens median | prompt_approx | findings_json_approx |")
    print("|---|---|---|---|---|")
    for key in sorted(cells.keys()):
        handler, mode = key
        cell_rows = cells[key]
        if not cell_rows:
            continue
        in_toks_med = int(statistics.median(r.get("input_tokens", 0) for r in cell_rows))
        prompt_t = prompt_token_approx.get(key, 0)
        findings_approx = max(0, in_toks_med - prompt_t - 200)
        print(f"| {handler} | {mode} | {in_toks_med} | {prompt_t} | {findings_approx} |")
    print()

    print("## Baseline comparison (vs 7b.2.B Handler 3 v3: in=21573, translate_s=10.23)")
    print()
    dyc_deep = cells.get(("did_you_consider", "deep"), [])
    if dyc_deep:
        in_med = int(statistics.median(r.get("input_tokens", 0) for r in dyc_deep))
        ts_med = statistics.median(r.get("translate_s", 0.0) for r in dyc_deep)
        delta_in = in_med - 21573
        delta_ts = ts_med - 10.23
        print(f"- **Handler 3 (did_you_consider) deep**: in_tokens median = {in_med} (Δ {delta_in:+d} vs 7b.2.B baseline 21573)")
        print(f"- **Handler 3 (did_you_consider) deep**: translate_s median = {ts_med:.2f}s (Δ {delta_ts:+.2f}s vs 7b.2.B baseline 10.23s)")
        print(f"- Per-row sample variance (translate_s std dev): "
              f"{statistics.stdev(r.get('translate_s', 0.0) for r in dyc_deep) if len(dyc_deep) > 1 else 0:.2f}s")
    print()
    print("## Trough-prose check fires (Chunk 7c.5.A integration signal)")
    print()
    trough_rows = [
        r for r in rows
        if any(e.get("kind") == "incomplete_trough_citation" for e in (r.get("validation_errors") or []))
    ]
    print(f"- `incomplete_trough_citation` fires: **{len(trough_rows)}** of {len(rows)} cells ({100.0 * len(trough_rows) / max(1, len(rows)):.1f}%)")
    if trough_rows:
        print()
        print("  Cells with the violation:")
        for r in trough_rows[:10]:
            errs = [e for e in (r.get("validation_errors") or []) if e.get("kind") == "incomplete_trough_citation"]
            print(f"  - {r.get('handler')}/{r.get('mode')} on {r.get('position_id')}: {len(errs)} violation(s)")
        if len(trough_rows) > 10:
            print(f"  ... ({len(trough_rows) - 10} more cells)")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
