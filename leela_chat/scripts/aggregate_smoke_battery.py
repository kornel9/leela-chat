"""Aggregate Chunk 6.8 smoke battery JSONL into a dated markdown summary.

Reads `logs/smoke_battery_<ts>.jsonl` and writes
`tests/eval/smoke_summaries/<YYYY-MM-DD>.md` with: aggregate stats per
the hard gates, per-category breakdown, per-theme + per-difficulty-band
slicing (for tactical), Branch A vs B distribution, eyeball rubric
grid (deterministic first-output-per-cell selection), outlier list,
backlog items.

Run on the pod after the battery completes:

    uv run python -m leela_chat.scripts.aggregate_smoke_battery \
        --jsonl logs/smoke_battery_<ts>.jsonl \
        --out tests/eval/smoke_summaries/2026-05-21.md
"""
from __future__ import annotations

import argparse
import datetime
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from loguru import logger


FORBIDDEN_PHRASES = [
    "monotonic", "recovered to", "noticed", "realized",
    "discovered", "saw", "found",
]


def _load_jsonl(path: Path) -> list[dict]:
    """Load JSONL into a list of dicts."""
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _band(rating: int) -> str:
    """Map rating to a difficulty band label."""
    if rating < 1800:
        return "<1800"
    if rating < 2200:
        return "1800-2200"
    if rating < 2600:
        return "2200-2600"
    if rating < 3000:
        return "2600-3000"
    if rating < 3400:
        return "3000-3400"
    return "3400+"


def _classify_illegal_move_error(err: dict, fen: str, prose: str) -> str:
    """Classify an illegal_move error per the Chunk 6.7 CLAUDE.md §8 rule.

    Returns 'hallucination' (genuine Rule 1 violation) or 'parser_fp'
    (validator regex false-positive on prose like 'after Nf3 lines').
    Heuristic: if the span appears in the prose with surrounding context
    that's clearly not a move citation (e.g. 'after X lines' or 'X push'),
    classify as parser_fp; else hallucination.
    """
    span = err.get("span", "")
    if not span:
        return "hallucination"
    # Look for the span in prose with at least 20 chars context on either side.
    idx = prose.find(span)
    if idx < 0:
        return "hallucination"
    left = prose[max(0, idx - 30):idx].lower()
    right = prose[idx + len(span):idx + len(span) + 30].lower()
    hint_phrases = [
        "after", "lines", "push", "the ", " square", "controls", "develop",
    ]
    if any(h in left or h in right for h in hint_phrases):
        # Looks like prose-context not move-citation.
        return "parser_fp"
    return "hallucination"


def _classify_api_error_subkind(detail: str) -> str:
    """Sub-classify api_error by detail-string heuristics.

    Anthropic returns the same `api_error` kind for credit exhaustion,
    rate limits, overloads, and timeouts; we sub-classify by the
    detail's substring patterns so the banner can surface what actually
    went wrong instead of saying "check the field." See Chunk 6.8.1
    plan, Item 3A.
    """
    d = (detail or "").lower()
    if "credit" in d or "balance is too low" in d or "billing" in d:
        return "credit_exhausted"
    if "rate" in d or "429" in d:
        return "rate_limit"
    if "overload" in d or "529" in d:
        return "overloaded"
    if "timeout" in d or "timed out" in d:
        return "timeout"
    return "other"


def _strip_san_marks(s: str) -> str:
    """Normalise a SAN for comparison: strip check/mate marks.

    Chunk 6.9 helper. `Kxf3+` and `Kxf3` are the same move semantically
    but differ in string membership; for diagnostic purposes we treat
    them as equivalent when checking findings inclusion.
    """
    return s.rstrip("+#")


def _span_in_findings_summary(span: str, summary: dict) -> bool:
    """True if the SAN span appears in any trusted-moves array of findings_summary.

    Chunk 6.9 sub-item 1. Distinguishes validator_gap (Y here) from
    real_hallucination (N here) in the illegal_move classification.
    Normalises both sides by stripping check/mate marks; matches against
    every trusted SAN list in the summary.

    Args:
        span: The validator-flagged SAN string (may include + / # suffixes).
        summary: The row's `findings_summary` dict (see
            `run_smoke_battery._extract_findings_summary` for shape).

    Returns:
        True iff a normalised match is found in any trusted-moves array.
    """
    if not span or not summary:
        return False
    normalised = _strip_san_marks(span)
    san_lists = (
        (summary.get("top_moves_san") or [])
        + (summary.get("mcts_children_san") or [])
        + (summary.get("stockfish_top_san") or [])
        + (summary.get("stockfish_pv_sans") or [])
    )
    for s in san_lists:
        if isinstance(s, str) and _strip_san_marks(s) == normalised:
            return True
    # Also check the singleton fields.
    for k in ("mcts_bestmove_san", "focal_move_san"):
        v = summary.get(k)
        if isinstance(v, str) and _strip_san_marks(v) == normalised:
            return True
    return False


def _excerpt_around_span(prose: str, span: str, radius: int = 80) -> str:
    """Excerpt ~2*radius chars of prose around the span for human review.

    Chunk 6.9 sub-item 1. If span not found in prose, returns a marker
    string instead of empty (signals to the reader that the prose may
    be templated fallback, not the original LLM-failed text).
    """
    if not span:
        return "(no span)"
    idx = prose.find(span)
    if idx < 0:
        return f"(span '{span}' not in prose; row may be templated fallback)"
    start = max(0, idx - radius)
    end = idx + len(span) + radius
    excerpt = prose[start:end].replace("\n", " ").replace("|", "\\|")
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(prose) else ""
    return f"{prefix}{excerpt}{suffix}"


def _detect_circuit_breaker_sentinel(rows: list[dict]) -> dict | None:
    """Find the {_circuit_breaker: True, ...} sentinel row if present.

    The runner writes this as the last JSONL entry when the
    consecutive-api-error breaker engages (Chunk 6.8.1 Item 3B). Return
    the sentinel dict (without `_circuit_breaker` mixed into stats) or
    None if no breaker engaged.
    """
    for r in rows:
        if r.get("_circuit_breaker") is True:
            return r
    return None


def _render(rows: list[dict], jsonl_path: Path) -> str:
    """Build the markdown summary."""
    # Filter out any circuit-breaker sentinel before computing stats; it's
    # a meta-row, not a real (position, handler, mode) data point.
    sentinel = _detect_circuit_breaker_sentinel(rows)
    rows = [r for r in rows if not r.get("_circuit_breaker")]

    n_total = len(rows)
    if n_total == 0:
        return "# Smoke battery — empty JSONL\n"

    # === Three-rate classification (Chunk 6.8.1) ===
    # LLM-success: the LLM actually returned content (source=llm or llm_retry).
    n_llm_success = sum(
        1 for r in rows
        if r.get("source") in ("llm", "llm_retry") and (r.get("llm_calls") or 0) > 0
    )
    llm_success_rate = n_llm_success / n_total if n_total else 0.0
    # Validation-pass AMONG llm-success runs only — the real prose-quality signal.
    n_val_pass_among_llm_success = sum(
        1 for r in rows
        if r.get("source") in ("llm", "llm_retry")
        and (r.get("llm_calls") or 0) > 0
        and r.get("validation_ok") is True
    )
    val_pass_rate_among_llm_success = (
        n_val_pass_among_llm_success / n_llm_success if n_llm_success else 0.0
    )
    # Top api_error sub-kinds (when LLM-success is below the gate).
    api_error_subkinds: Counter = Counter()
    for r in rows:
        for e in (r.get("validation_errors") or []):
            if e.get("kind") == "api_error":
                api_error_subkinds[_classify_api_error_subkind(e.get("detail", ""))] += 1

    # === Legacy single-rate (demoted but preserved for backward compat) ===
    n_validation_ok = sum(1 for r in rows if r.get("validation_ok") is True)
    val_rate = n_validation_ok / n_total

    by_source: Counter = Counter(r.get("source") for r in rows)
    n_templated = by_source.get("templated", 0)
    templated_rate = n_templated / n_total

    # Latency
    handler_s = [r.get("handler_s", 0.0) for r in rows if r.get("handler_s") is not None]
    translate_s = [r.get("translate_s", 0.0) for r in rows if r.get("translate_s") is not None]
    total_s = [h + t for h, t in zip(handler_s, translate_s)]

    # Tokens
    in_toks = sum(r.get("input_tokens", 0) or 0 for r in rows)
    out_toks = sum(r.get("output_tokens", 0) or 0 for r in rows)

    # === Per-category breakdown ===
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r.get("category", "?")].append(r)
    cat_stats: list[dict] = []
    for cat, cat_rows in by_cat.items():
        n_cat = len(cat_rows)
        ok = sum(1 for r in cat_rows if r.get("validation_ok") is True)
        templated = sum(1 for r in cat_rows if r.get("source") == "templated")
        med_lat = statistics.median([r.get("handler_s", 0.0) + r.get("translate_s", 0.0) for r in cat_rows])
        cat_stats.append({
            "category": cat,
            "n": n_cat,
            "validation_ok_rate": ok / n_cat if n_cat else 0.0,
            "templated_rate": templated / n_cat if n_cat else 0.0,
            "median_latency_s": med_lat,
        })

    # === Per-theme breakdown (tactical only) ===
    tactical_rows = [r for r in rows if r.get("category") == "tactical_puzzle"]
    by_theme: dict[str, list[dict]] = defaultdict(list)
    for r in tactical_rows:
        themes = r.get("themes") or []
        for t in themes:
            if t in ("fork", "pin", "sacrifice", "discoveredAttack", "skewer", "mateIn3", "mateIn4", "mateIn5"):
                by_theme[t].append(r)
    theme_stats: list[dict] = []
    for theme, theme_rows in sorted(by_theme.items()):
        n_t = len(theme_rows)
        ok = sum(1 for r in theme_rows if r.get("validation_ok") is True)
        mentioned = sum(1 for r in theme_rows if theme.lower() in (r.get("prose") or "").lower())
        theme_stats.append({
            "theme": theme,
            "n": n_t,
            "validation_ok_rate": ok / n_t if n_t else 0.0,
            "mentioned_rate": mentioned / n_t if n_t else 0.0,
        })

    # === Per-difficulty-band breakdown ===
    by_band: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("category") == "tactical_puzzle" and r.get("rating") is not None:
            by_band[_band(r["rating"])].append(r)
    band_stats: list[dict] = []
    for band, band_rows in sorted(by_band.items()):
        n_b = len(band_rows)
        ok = sum(1 for r in band_rows if r.get("validation_ok") is True)
        branch_b = sum(
            1 for r in band_rows
            if r.get("focal_uci") is not None
            and r.get("mcts_bestmove_uci") is not None
            and r.get("focal_uci") != r.get("mcts_bestmove_uci")
        )
        band_stats.append({
            "band": band,
            "n": n_b,
            "validation_ok_rate": ok / n_b if n_b else 0.0,
            "branch_b_rate": branch_b / n_b if n_b else 0.0,
        })

    # === Branch A vs B distribution (handlers 2 + 3 only) ===
    h2h3 = [r for r in rows if r.get("handler") in ("why_this_move", "did_you_consider")]
    branch_a = sum(
        1 for r in h2h3
        if r.get("focal_matches_mcts_pick") is True
    )
    branch_b = sum(
        1 for r in h2h3
        if r.get("focal_uci") is not None
        and r.get("mcts_bestmove_uci") is not None
        and not r.get("focal_matches_mcts_pick", False)
    )

    # === illegal_move classification (Chunk 6.9 enhanced) ===
    illegal_records: list[dict] = []
    for r in rows:
        for err in r.get("validation_errors", []) or []:
            if err.get("kind") == "illegal_move":
                span = err.get("span", "")
                # Use the LAST llm_attempt_texts entry (the one the validator
                # most-recently rejected) if available; fall back to the
                # templated `prose` for old JSONLs.
                attempts = r.get("llm_attempt_texts") or []
                source_prose = attempts[-1] if attempts else (r.get("prose") or "")
                klass = _classify_illegal_move_error(
                    err, r.get("fen", ""), source_prose,
                )
                # Chunk 6.9: check if the span IS in findings_summary
                # (validator_gap detection — was theatre before this field
                # existed; see chunk plan Risks section).
                in_findings = _span_in_findings_summary(
                    span, r.get("findings_summary") or {},
                )
                # Failed-prose excerpt around the span.
                excerpt = _excerpt_around_span(source_prose, span, radius=80)
                # Refined classification: if span IS in findings but
                # validator flagged it, that's a validator_gap (overrides
                # the heuristic classifier).
                if in_findings:
                    klass = "validator_gap"
                illegal_records.append({
                    "position": r.get("position_id"),
                    "handler": r.get("handler"),
                    "mode": r.get("mode"),
                    "span": span,
                    "klass": klass,
                    "detail": err.get("detail"),
                    "in_findings": in_findings,
                    "failed_prose_excerpt": excerpt,
                })
    n_halluc = sum(1 for x in illegal_records if x["klass"] == "hallucination")
    n_parser_fp = sum(1 for x in illegal_records if x["klass"] == "parser_fp")
    n_validator_gap = sum(1 for x in illegal_records if x["klass"] == "validator_gap")

    # === Eyeball rubric — first output per (category × handler × mode) cell ===
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r.get("category"), r.get("handler"), r.get("mode"))].append(r)
    eyeball_rows: list[dict] = []
    for (cat, handler, mode), cell_rows in sorted(cells.items()):
        # Order by position_id for determinism.
        cell_rows.sort(key=lambda r: r.get("position_id", ""))
        first = cell_rows[0]
        prose = first.get("prose", "") or ""
        forbidden_hits = [p for p in FORBIDDEN_PHRASES if p in prose.lower()]
        # Chunk 7b.2.C: per-cell interpretive-claim candidates surfaced
        # for 2500+ rater review at 7o. Populated from the JSONL row's
        # interpretive_claims field (added in 7b.2.C). Back-compat:
        # older JSONL rows without the field treated as empty list.
        interp_claims = first.get("interpretive_claims", []) or []
        eyeball_rows.append({
            "category": cat,
            "handler": handler,
            "mode": mode,
            "position_id": first.get("position_id"),
            "validation_ok": first.get("validation_ok"),
            "forbidden_hits": forbidden_hits,
            "prose": prose,
            "interpretive_claims": interp_claims,
        })

    # === Outliers — non-templated runs that failed validation (Rule 1 / Rule 2 catches) ===
    outliers: list[dict] = [
        {
            "position": r.get("position_id"),
            "handler": r.get("handler"),
            "mode": r.get("mode"),
            "source": r.get("source"),
            "errors": [e.get("kind") for e in (r.get("validation_errors") or [])],
        }
        for r in rows
        if r.get("validation_ok") is False and r.get("source") not in ("templated", "DRY_RUN", "CRASH")
    ]

    # === Render ===
    lines: list[str] = []
    today = datetime.date.today().isoformat()
    lines.append(f"# Smoke battery summary — {today}")
    lines.append("")
    lines.append(f"Raw data: `{jsonl_path}`")
    lines.append("")

    # === Circuit-breaker sentinel banner (if engaged) ===
    if sentinel is not None:
        lines.append("## ⚠️ CIRCUIT BREAKER ENGAGED")
        lines.append("")
        lines.append(
            f"The smoke battery runner detected **{sentinel.get('consecutive_api_errors', '?')} "
            f"consecutive `{sentinel.get('last_error_kind', 'api_error')}` responses** and aborted "
            f"the run early. Last error detail: "
            f"`{(sentinel.get('last_error_detail') or '')[:140]}...`"
        )
        lines.append("")
        lines.append(
            "This is almost always Anthropic credit exhaustion or a persistent "
            "API outage — fix the underlying issue and re-run via `--reuse-fixture`."
        )
        lines.append("")

    # === Three-rate classification (Chunk 6.8.1) ===
    lines.append("## Headline (three-rate classification)")
    lines.append("")
    # LLM-success leads — if it's below the gate, validation numbers are unreliable.
    if llm_success_rate < 0.95:
        top_kinds_str = ", ".join(
            f"`api_error:{k}` ({c})" for k, c in api_error_subkinds.most_common(3)
        )
        lines.append(
            f"**⚠️ LLM-success rate is {llm_success_rate:.1%}** "
            f"({n_llm_success}/{n_total} calls returned LLM content). "
        )
        if top_kinds_str:
            lines.append(f"Top api_error kinds: {top_kinds_str}.")
        lines.append(
            "The validation-pass numbers below reflect only the LLM-success "
            "calls and may be unreliable until LLM health is restored."
        )
        lines.append("")
    lines.append(f"- **LLM-success rate:** {llm_success_rate:.1%} ({n_llm_success}/{n_total})")
    lines.append(
        f"- **Validation-pass among LLM-success:** "
        f"{val_pass_rate_among_llm_success:.1%} "
        f"({n_val_pass_among_llm_success}/{n_llm_success}) — "
        f"{'real prose-quality signal' if llm_success_rate >= 0.95 else 'unreliable; see warning above'}"
    )
    lines.append(f"- **Templated-fallback rate:** {templated_rate:.1%} (target ≤ 10%)")
    lines.append("")

    # === Legacy single-rate + latency + tokens (demoted) ===
    lines.append("## Aggregate (legacy single-rate + latency)")
    lines.append("")
    lines.append(f"- **Runs:** {n_total} (expected 100; missing = crashes / fixture issues)")
    lines.append(f"- **Validation_ok rate (legacy single-rate):** {val_rate:.1%} (target ≥ 90%)")
    lines.append(f"- **Sources:** {dict(by_source)}")
    lines.append(f"- **Latency** (handler_s + translate_s):")
    if total_s:
        lines.append(f"  - mean {statistics.mean(total_s):.2f}s")
        lines.append(f"  - median {statistics.median(total_s):.2f}s")
        lines.append(f"  - p95 {_percentile(total_s, 95):.2f}s")
        lines.append(f"  - p99 {_percentile(total_s, 99):.2f}s")
    lines.append(f"- **Token consumption:** {in_toks:,} in / {out_toks:,} out")
    lines.append("")

    lines.append("## illegal_move classification")
    lines.append("")
    lines.append(f"- **Genuine hallucinations** (Rule 1 violations the validator caught): {n_halluc}")
    lines.append(f"- **Validator parser false-positives** (prose-context regex misreads): {n_parser_fp}")
    lines.append(f"- **Validator gaps** (span IS in findings, validator missed it): {n_validator_gap}")
    if illegal_records:
        lines.append("")
        lines.append("| Position | Handler/mode | Span | In findings? | Classification | Failed LLM prose excerpt |")
        lines.append("|---|---|---|---|---|---|")
        for x in illegal_records:
            in_f = "✓" if x.get("in_findings") else "✗"
            excerpt = x.get("failed_prose_excerpt", "")
            lines.append(
                f"| {x['position']} | {x['handler']}/{x['mode']} | `{x['span']}` | "
                f"{in_f} | {x['klass']} | {excerpt} |"
            )
    lines.append("")

    lines.append("## Per-category breakdown")
    lines.append("")
    lines.append("| Category | N | validation_ok | templated | median latency (s) |")
    lines.append("|---|---|---|---|---|")
    for s in cat_stats:
        lines.append(
            f"| {s['category']} | {s['n']} | "
            f"{s['validation_ok_rate']:.0%} | {s['templated_rate']:.0%} | "
            f"{s['median_latency_s']:.2f} |"
        )
    lines.append("")

    lines.append("## Per-theme breakdown (tactical only)")
    lines.append("")
    lines.append("| Theme | N runs | validation_ok | prose-mentions-theme |")
    lines.append("|---|---|---|---|")
    for s in theme_stats:
        lines.append(
            f"| {s['theme']} | {s['n']} | "
            f"{s['validation_ok_rate']:.0%} | {s['mentioned_rate']:.0%} |"
        )
    lines.append("")

    lines.append("## Per-difficulty-band breakdown (tactical only)")
    lines.append("")
    lines.append("| Band | N runs | validation_ok | Branch B rate |")
    lines.append("|---|---|---|---|")
    for s in band_stats:
        lines.append(
            f"| {s['band']} | {s['n']} | "
            f"{s['validation_ok_rate']:.0%} | {s['branch_b_rate']:.0%} |"
        )
    lines.append("")
    lines.append(f"Branch A (focal == MCTS pick): {branch_a} of {len(h2h3)} H2+H3 runs")
    lines.append(f"Branch B (focal ≠ MCTS pick):  {branch_b} of {len(h2h3)} H2+H3 runs")
    lines.append("")

    lines.append("## Eyeball rubric (first output per category × handler × mode)")
    lines.append("")
    lines.append("Per-cell rubric (Y/N):")
    lines.append("1. **Position-character match** — judge manually per Chunk 6.8 plan.")
    lines.append("2. **Grounding** — auto-Y for validation_ok=True; manually re-check the others.")
    lines.append("3. **Forbidden-language check** — auto-fail if the auto-scan column shows hits.")
    lines.append("4. **Over-claim check** — judge manually per the plan.")
    lines.append("")
    lines.append("| Cell | Position | val_ok | Forbidden hits | Prose (first 400 chars) |")
    lines.append("|---|---|---|---|---|")
    for e in eyeball_rows:
        cell = f"{e['category']} / {e['handler']} / {e['mode']}"
        hits = ", ".join(f"`{h}`" for h in e["forbidden_hits"]) if e["forbidden_hits"] else "(none)"
        prose_trunc = (e["prose"] or "")[:400].replace("\n", " ").replace("|", "\\|")
        lines.append(f"| {cell} | {e['position_id']} | {e['validation_ok']} | {hits} | {prose_trunc}… |")
    lines.append("")

    # Chunk 7b.2.C: per-cell interpretive-claim candidates for 2500+ rater
    # review at 7o (Option 1.5 from the 7b.1 retro). The validator catches
    # hallucinated MOVES but not hallucinated INTERPRETATIONS; this section
    # surfaces candidates for explicit rater scrutiny rather than relying
    # on the rater to spot them in unstructured prose. False positives are
    # expected and FINE; the rater judges.
    interp_counts = [len(e["interpretive_claims"]) for e in eyeball_rows]
    n_cells = len(eyeball_rows)
    n_cells_with_claims = sum(1 for c in interp_counts if c > 0)
    total_claims = sum(interp_counts)
    median_claims = (
        sorted(interp_counts)[n_cells // 2] if n_cells > 0 else 0
    )
    max_claims = max(interp_counts) if interp_counts else 0
    lines.append("## Interpretive claims to scrutinize (per-cell — for 2500+ rater)")
    lines.append("")
    lines.append(
        "Per Chunk 7b.2.C / Option 1.5 from the 7b.1 retro. The validator "
        "catches hallucinated MOVES but not hallucinated INTERPRETATIONS "
        "(positional/relational claims like 'queen has nowhere to hide'). "
        "The smoke-battery runner flags candidate interpretive-claim "
        "sentences via a two-rule heuristic (wordlist + structural). "
        "False positives are expected; the rater judges chess-accuracy."
    )
    lines.append("")
    lines.append(
        f"**Heuristic-output stats:** {n_cells_with_claims} of {n_cells} "
        f"cells have ≥1 candidate; median per cell = {median_claims}; "
        f"max per cell = {max_claims}; total = {total_claims}."
    )
    lines.append("")
    lines.append(
        "**Rater instruction:** for each cell, review the candidates "
        "FIRST. Flag any claim that's chess-inaccurate OR unsupported by "
        "the prose's other citations OR makes a positional claim the "
        "findings JSON doesn't ground. ≥3 flagged claims in a cell fails "
        "the prose-quality gate for that cell regardless of other criteria."
    )
    lines.append("")
    if total_claims == 0:
        lines.append("(no candidates surfaced — either heuristic is too narrow OR prose is clean)")
    else:
        for e in eyeball_rows:
            if not e["interpretive_claims"]:
                continue
            cell = f"{e['category']} / {e['handler']} / {e['mode']}"
            lines.append(f"### {cell} ({e['position_id']}) — {len(e['interpretive_claims'])} candidate(s)")
            for claim in e["interpretive_claims"]:
                # Escape pipe chars in case rater renders in a table-context view.
                claim_md = claim.replace("|", "\\|")
                lines.append(f"- {claim_md}")
            lines.append("")

    lines.append("## Outliers (validation_ok=False, source != templated)")
    lines.append("")
    if not outliers:
        lines.append("None — every non-templated run validated cleanly.")
    else:
        lines.append("| Position | Handler/mode | Source | Errors |")
        lines.append("|---|---|---|---|")
        for o in outliers:
            lines.append(f"| {o['position']} | {o['handler']}/{o['mode']} | {o['source']} | {o['errors']} |")
    lines.append("")

    lines.append("## Backlog candidates (to be filed in PHASE_NOTES)")
    lines.append("")
    lines.append("Pre-filled from automatic analysis; human review may add more:")
    if val_rate < 0.90:
        lines.append(f"- Validation rate {val_rate:.1%} is below 90% gate — per-category decision needed (Chunk 7 entry deferral OR ship anyway?).")
    if templated_rate > 0.10:
        lines.append(f"- Templated-fallback rate {templated_rate:.1%} is above 10% gate — per-category decision needed.")
    if n_halluc > 0:
        lines.append(f"- {n_halluc} genuine hallucination(s) — Rule 1 violation; halt-and-investigate per CLAUDE.md §8.")
    if n_parser_fp > 0:
        lines.append(f"- {n_parser_fp} validator parser false-positive(s) — file as validator backlog item.")
    forbidden_any = sum(1 for e in eyeball_rows if e["forbidden_hits"])
    if forbidden_any > 0:
        lines.append(f"- {forbidden_any} eyeball cells have forbidden-language hits — file as prompt v6/v3 iteration target.")
    if branch_b == 0:
        lines.append("- 0 Branch B cases across H2+H3 runs — difficulty stratification didn't produce divergence; consider bumping tactical ratings higher OR pulling from leela-interp figshare distribution.")
    # Chunk 7b.2.C rater-overwhelm mitigation: tune heuristic before 7o
    # if per-cell counts trend too high (target at 7o: median 1-2, max 4).
    # Thresholds TIGHTENED 2026-05-24 from 7b.2 retro: during 7c-7f the
    # early-warning threshold is median > 1 OR max > 3 (vs the
    # post-real-data threshold of >2 / >4). Rationale: the 7b.2 baseline
    # was n=2 samples (median=0.5, max=1); tighter thresholds catch
    # over-firing faster given the weak baseline confidence. At 7o, with
    # ~100 cells of real data, re-evaluate whether to revert to >2 / >4
    # for stable rater-instruction language (see Chunk 7 plan's
    # rater-overwhelm risk row).
    if median_claims > 1 or max_claims > 3:
        lines.append(
            f"- Interpretive-claim heuristic surfacing too many candidates "
            f"(median={median_claims}, max={max_claims}). Early-warning "
            f"thresholds during 7c-7f: median > 1 OR max > 3 (tightened "
            f"from the post-real-data >2/>4 to catch over-firing faster "
            f"given the n=2 7b.2 baseline). Target at 7o: median ≤ 2, "
            f"max ≤ 4. Tighten Rule A's match threshold (require 2 "
            f"wordlist hits, not 1) OR narrow Rule B's relational "
            f"patterns BEFORE more sub-chunks accumulate."
        )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True, help="Path to smoke battery JSONL")
    parser.add_argument("--out", required=True, help="Path to output markdown summary")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    out_path = Path(args.out)
    rows = _load_jsonl(jsonl_path)
    logger.info(f"loaded {len(rows)} rows from {jsonl_path}")
    md = _render(rows, jsonl_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    logger.info(f"wrote summary to {out_path}")


if __name__ == "__main__":
    main()
