"""Analyze a smoke-battery JSONL — per-cell decomposition + trough-scan.

Originally written for 7c.5.E (controlled-sample measurement baseline).
Extended at 7c.7 to (a) replace the 4-chars-per-token heuristic with
``anthropic.messages.count_tokens`` for exact counts, (b) surface a
trough-scan section that disambiguates "0 fires across N cells"
(validator works) from "0 trough cases in sample" (validator untested),
and (c) extract the walkthrough_rxd4 row from the JSONL and compare
its per-call telemetry directly to the 7b.2.B single-call baseline
(in=21573, translate_s=10.23s) — the apples-to-apples comparison.

Decomposes per-call input_tokens into approximate components:
- system_prompt_tokens (exact via anthropic.messages.count_tokens)
- findings_json_tokens (best-effort: persisted only as findings_summary
  in the JSONL, which is lossy — so this is a lower-bound proxy)
- the residual "framing" overhead (small)

Computes per-handler-per-mode median + p25/p75 for input_tokens,
output_tokens, translate_s, handler_s. Compares against the 7b.2.B
baseline (Handler 3 v3: in=21573, translate_s=10.23s).

Trough-scan (added 7c.7 sub-fix 1a):
- Reads each row's ``trajectory_summaries`` field (persisted at row
  write time per 7c.7 sub-fix 1a-struct in run_smoke_battery.py).
- For each summary, compute max_delta = max(|q_trough - q_first|,
  |q_trough - q_last|). Bucket: how many cells had max_delta > T=0.10
  (the trigger condition)? How many of those also had the validator
  fire (``incomplete_trough_citation`` in validation_errors)?
- Older JSONL rows (pre-7c.7) lack the ``trajectory_summaries`` field;
  the analyzer reports them as "N undetermined" and computes F from
  validation_errors alone.

Output is a markdown block suitable for direct copy-paste into
PHASE_NOTES under the relevant retrospective entry.

Usage:

    uv run python -m leela_chat.scripts.analyze_token_baseline \\
        --jsonl logs/smoke_battery_<YYYYMMDDTHHMMSSZ>.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
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

# Trough-scan threshold — matches the validator's _TROUGH_DELTA_THRESHOLD
# (7c.5.A). Kept in sync manually for now; 7c.9 sub-fix 4-pre.1 may
# promote to a public constant in the validator that this script imports.
TROUGH_DELTA_THRESHOLD = 0.10

# 7b.2.B baseline (Handler 3 v3 single call on walkthrough_rxd4).
WALKTHROUGH_BASELINE = {
    "input_tokens": 21573,
    "translate_s": 10.23,
    "handler": "did_you_consider",
    "mode": "deep",
    "prompt_version": 3,
    "fixture": "walkthrough_rxd4",
}


def _build_token_counter():
    """Return an exact-count callable backed by anthropic.messages.count_tokens.

    The endpoint is free or essentially so (no generation; just
    tokenization); per-call latency is sub-second; result is the EXACT
    count Anthropic's API will use at translate time. Per ADR-0005,
    prod uses claude-opus-4-7 — the count_tokens result is model-specific
    so we pin the model.

    Returns:
        A callable ``count(text: str) -> int``. Raises if the API key
        isn't available; the script then must be run with ANTHROPIC_API_KEY
        in the environment.
    """
    from anthropic import Anthropic
    client = Anthropic()
    model = "claude-opus-4-7"

    def count(text: str) -> int:
        # The count_tokens API expects a messages-shaped payload.
        resp = client.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": text}],
        )
        return int(resp.input_tokens)

    return count


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


def _row_trough_scan(row: dict) -> dict:
    """Compute trough-scan facts for one JSONL row.

    Returns dict with:
    - ``has_trajectory_data``: True if row persisted ``trajectory_summaries``
      (pre-7c.7 rows have this field absent → False).
    - ``trough_count_total``: number of trajectory_summary entries in the row.
    - ``trough_count_triggering``: number of trajectory_summary entries
      where max_delta > THRESHOLD (i.e., the trigger condition met).
    - ``validator_fired``: True if any validation_errors entry has
      kind == "incomplete_trough_citation".
    - ``max_delta_per_traj``: list of computed max_delta values (for
      threshold-tuning visibility).
    """
    trajs = row.get("trajectory_summaries")
    has_data = trajs is not None  # field present (may be [] for no-trajectory cells)
    trajs = trajs or []

    deltas: list[float] = []
    triggering = 0
    for summary in trajs:
        q_first = summary.get("q_first")
        q_last = summary.get("q_last")
        q_trough = summary.get("q_trough")
        if not all(isinstance(v, (int, float)) for v in (q_first, q_last, q_trough)):
            continue
        max_delta = max(abs(q_trough - q_first), abs(q_trough - q_last))
        deltas.append(max_delta)
        if max_delta > TROUGH_DELTA_THRESHOLD:
            triggering += 1

    fired = any(
        e.get("kind") == "incomplete_trough_citation"
        for e in (row.get("validation_errors") or [])
    )

    return {
        "has_trajectory_data": has_data,
        "trough_count_total": len(trajs),
        "trough_count_triggering": triggering,
        "validator_fired": fired,
        "max_delta_per_traj": deltas,
    }


def _render_trough_scan(rows: list[dict]) -> None:
    """Emit the trough-scan markdown section to stdout.

    Disambiguates the "0 fires" headline:
    - F = cells where the validator fired.
    - N = cells where the trigger condition met (max_delta > T).
    - M = cells with trajectory data persisted.
    - F / N: real validator compliance rate (after 7c.7-persisted data flows).
    - M < total: indicates older rows that pre-date 7c.7's persistence.
    """
    facts = [_row_trough_scan(r) for r in rows]
    total = len(rows)
    m_persisted = sum(1 for f in facts if f["has_trajectory_data"])
    n_triggering = sum(f["trough_count_triggering"] for f in facts if f["has_trajectory_data"])
    n_total_trajs = sum(f["trough_count_total"] for f in facts if f["has_trajectory_data"])
    f_fired = sum(1 for f in facts if f["validator_fired"])

    print("## Trough-prose check: trigger-condition vs fire-rate")
    print()
    print(f"- Total rows scanned: **{total}**")
    print(f"- Rows with `trajectory_summaries` field persisted (7c.7+): **{m_persisted}** of {total}")
    if m_persisted < total:
        print(f"  - **{total - m_persisted} rows pre-date 7c.7's persistence** — N undetermined for these; F still computable from validation_errors.")
    print(f"- Total `trajectory_summary` entries across persisted rows: **{n_total_trajs}**")
    print(f"- Entries where max_delta > {TROUGH_DELTA_THRESHOLD} (trigger condition met): **{n_triggering}**")
    print(f"- Cells where validator fired (`incomplete_trough_citation`): **{f_fired}** of {total}")

    # Headline disambiguation:
    if m_persisted == 0:
        print()
        print("**Headline (this JSONL):** validator fire rate is the ONLY signal here — `trajectory_summaries` was not persisted at row write time. Re-run smoke battery against post-7c.7 code to get the full trigger-vs-fire breakdown.")
    elif n_triggering == 0:
        print()
        print("**Headline:** F=0 reflects an empty trigger pool — no trajectory in this JSONL had max_delta > threshold. The validator is **untested in this sample**; need a sample with real trough cases to validate compliance.")
    else:
        compliance = (n_triggering - f_fired) / n_triggering * 100.0 if n_triggering else 100.0
        print()
        print(f"**Headline:** {n_triggering} trigger conditions across persisted rows; {f_fired} validator fires; **compliance rate {compliance:.1f}%**.")

    # If any trough fires, list the cells for diagnosis.
    fires = [r for r, f in zip(rows, facts) if f["validator_fired"]]
    if fires:
        print()
        print("Cells where the validator fired:")
        for r in fires[:10]:
            errs = [e for e in (r.get("validation_errors") or [])
                    if e.get("kind") == "incomplete_trough_citation"]
            print(f"  - {r.get('handler')}/{r.get('mode')} on {r.get('position_id')}: {len(errs)} violation(s)")
        if len(fires) > 10:
            print(f"  ... ({len(fires) - 10} more cells)")
    print()


def _render_walkthrough_baseline_delta(rows: list[dict]) -> None:
    """Apples-to-apples: extract walkthrough_rxd4 row and compare to 7b.2.B.

    The smoke battery's `--reuse-fixture` mode runs against the 20-position
    fixture which includes walkthrough_rxd4. This section pulls that
    single row and shows the delta vs the 7b.2.B single-call telemetry.
    """
    fixture_rows = [
        r for r in rows
        if r.get("position_id") == WALKTHROUGH_BASELINE["fixture"]
        and r.get("handler") == WALKTHROUGH_BASELINE["handler"]
        and r.get("mode") == WALKTHROUGH_BASELINE["mode"]
    ]

    print("## Walkthrough-only baseline-delta (vs 7b.2.B)")
    print()
    print(
        f"7b.2.B baseline (single call on `{WALKTHROUGH_BASELINE['fixture']}`, "
        f"{WALKTHROUGH_BASELINE['handler']}/{WALKTHROUGH_BASELINE['mode']}, "
        f"v{WALKTHROUGH_BASELINE['prompt_version']} prompt): "
        f"in_tokens={WALKTHROUGH_BASELINE['input_tokens']}, "
        f"translate_s={WALKTHROUGH_BASELINE['translate_s']}s."
    )
    print()

    if not fixture_rows:
        print(
            "_No `walkthrough_rxd4` row found in this JSONL for the "
            "did_you_consider/deep cell. (The smoke battery's "
            "`--reuse-fixture` mode normally includes this position; "
            "absence here means the JSONL was sampled stratified "
            "without the walkthrough fixture pinned.)_"
        )
        print()
        return

    if len(fixture_rows) > 1:
        print(f"_(Multiple {WALKTHROUGH_BASELINE['fixture']} rows found; using first.)_")
        print()
    row = fixture_rows[0]
    in_tok = row.get("input_tokens", 0)
    out_tok = row.get("output_tokens", 0)
    handler_s = row.get("handler_s", 0.0)
    translate_s = row.get("translate_s", 0.0)
    prompt_v = row.get("prompt_version", "?")

    d_in = in_tok - WALKTHROUGH_BASELINE["input_tokens"]
    d_ts = translate_s - WALKTHROUGH_BASELINE["translate_s"]

    print(f"Current row: in_tokens={in_tok}, out_tokens={out_tok}, handler_s={handler_s:.2f}s, translate_s={translate_s:.2f}s (prompt v{prompt_v}).")
    print()
    print(f"- **Δ input_tokens vs 7b.2.B:** {d_in:+d} ({(d_in / WALKTHROUGH_BASELINE['input_tokens'] * 100):+.1f}%)")
    print(f"- **Δ translate_s vs 7b.2.B:** {d_ts:+.2f}s ({(d_ts / WALKTHROUGH_BASELINE['translate_s'] * 100):+.1f}%)")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument(
        "--no-count-tokens", action="store_true",
        help="Skip the anthropic.messages.count_tokens calls. Use when "
             "the API key isn't available; output shows N/A for the "
             "prompt-token columns.",
    )
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("source") not in ("DRY_RUN", "CRASH", "circuit_breaker_sentinel")]

    # Compute prompt-file token counts via anthropic.messages.count_tokens.
    prompt_token_counts: dict[tuple, int | None] = {}
    if args.no_count_tokens or not os.environ.get("ANTHROPIC_API_KEY"):
        if not args.no_count_tokens:
            print("# WARNING: ANTHROPIC_API_KEY not set; falling back to --no-count-tokens behavior.")
        for key in PROMPT_FILES.keys():
            prompt_token_counts[key] = None
    else:
        count_tokens = _build_token_counter()
        for (handler, mode), path in PROMPT_FILES.items():
            text = Path(path).read_text()
            prompt_token_counts[(handler, mode)] = count_tokens(text)

    # Group rows by (handler, mode).
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        handler = r.get("handler")
        mode = r.get("mode")
        if handler and mode:
            cells[(handler, mode)].append(r)

    print(f"# Smoke battery analysis — `{args.jsonl.name}`")
    print()
    print(f"Total rows: {len(rows)} (after filtering DRY_RUN/CRASH/sentinel)")
    print()
    print("## Per-handler-per-mode statistics")
    print()
    print("| Handler | Mode | n | in_tokens median | in_tokens p25-p75 | out_tokens median | translate_s median | handler_s median | system_prompt_tokens (exact) |")
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
        prompt_t = prompt_token_counts.get(key)
        prompt_t_str = str(prompt_t) if prompt_t is not None else "N/A (no API key)"
        print(
            f"| {handler} | {mode} | {n} | "
            f"{int(statistics.median(in_toks))} | "
            f"{int(_percentile(in_toks, 25))}-{int(_percentile(in_toks, 75))} | "
            f"{int(statistics.median(out_toks))} | "
            f"{statistics.median(translate_s):.2f} | "
            f"{statistics.median(handler_s):.2f} | "
            f"{prompt_t_str} |"
        )
    print()

    print("## Token decomposition (with prompt-caching split)")
    print()
    print(
        "_Anthropic's `usage.input_tokens` field with prompt caching "
        "enabled reports the UNCACHED portion only (the user message: "
        "question + findings_json). The cached system prompt's cost "
        "lives in `cache_creation_input_tokens` (first call) or "
        "`cache_read_input_tokens` (subsequent calls). Pre-7c.7 smoke "
        "battery rows persist only `input_tokens`, missing both cache "
        "fields — those rows show `cache_write` / `cache_read` as `N/A`. "
        "Post-7c.7 rows persist all three (per 7c.7 sub-fix 1c-struct)._"
    )
    print()
    print(
        "_`findings_json` token count is a lower-bound proxy: "
        "`findings_summary` is intentionally lossy (per its 6.9 design) "
        "so the JSONL doesn't capture the exact JSON the LLM saw. The "
        "lower-bound = max(0, in_tokens − ~50 framing) since "
        "`input_tokens` is dominated by the findings JSON for any "
        "non-trivial findings._"
    )
    print()
    print("| Handler | Mode | in_tokens (user-msg) median | system_prompt_exact (cached) | cache_write median | cache_read median | findings_json_lower_bound |")
    print("|---|---|---|---|---|---|---|")
    for key in sorted(cells.keys()):
        handler, mode = key
        cell_rows = cells[key]
        if not cell_rows:
            continue
        in_toks_med = int(statistics.median(r.get("input_tokens", 0) for r in cell_rows))
        prompt_t = prompt_token_counts.get(key)
        # Pre-7c.7 rows omit cache_creation / cache_read; .get returns None or 0.
        # Distinguish via membership check on the first row.
        first = cell_rows[0]
        has_cache_fields = (
            "cache_creation_input_tokens" in first
            or "cache_read_input_tokens" in first
        )
        if has_cache_fields:
            cache_write_med = int(statistics.median(
                r.get("cache_creation_input_tokens", 0) for r in cell_rows
            ))
            cache_read_med = int(statistics.median(
                r.get("cache_read_input_tokens", 0) for r in cell_rows
            ))
            cache_write_str = str(cache_write_med)
            cache_read_str = str(cache_read_med)
        else:
            cache_write_str = "N/A (pre-7c.7)"
            cache_read_str = "N/A (pre-7c.7)"

        prompt_t_str = str(prompt_t) if prompt_t is not None else "N/A"
        findings_lb = max(0, in_toks_med - 50)
        print(
            f"| {handler} | {mode} | {in_toks_med} | {prompt_t_str} | "
            f"{cache_write_str} | {cache_read_str} | {findings_lb} |"
        )
    print()

    print("## Baseline comparison (vs 7b.2.B Handler 3 v3: in=21573, translate_s=10.23)")
    print()
    dyc_deep = cells.get(("did_you_consider", "deep"), [])
    if dyc_deep:
        in_med = int(statistics.median(r.get("input_tokens", 0) for r in dyc_deep))
        ts_med = statistics.median(r.get("translate_s", 0.0) for r in dyc_deep)
        delta_in = in_med - 21573
        delta_ts = ts_med - 10.23
        print(f"- **Handler 3 (did_you_consider) deep, all cells**: in_tokens median = {in_med} (Δ {delta_in:+d} vs 7b.2.B baseline 21573)")
        print(f"- **Handler 3 (did_you_consider) deep, all cells**: translate_s median = {ts_med:.2f}s (Δ {delta_ts:+.2f}s vs 7b.2.B baseline 10.23s)")
        print(
            f"- Per-row sample variance (translate_s std dev): "
            f"{statistics.stdev(r.get('translate_s', 0.0) for r in dyc_deep) if len(dyc_deep) > 1 else 0:.2f}s"
        )
        print()
        print(
            "_Note: this aggregate compares mixed positions in the JSONL "
            "to the 7b.2.B single-call walkthrough baseline — apples-to-"
            "oranges. The walkthrough-only baseline-delta below is the "
            "apples-to-apples comparison._"
        )
    print()

    _render_walkthrough_baseline_delta(rows)

    _render_trough_scan(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
