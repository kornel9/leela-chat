# Smoke battery — operational/regression half of the eval set

> This file describes the smoke battery introduced in Chunk 6.8. It is the MEASURE half of an eval set. The GRADING half (`tests/eval/eval_positions.yaml`, friends-authored with ground-truth commentaries) is separate; see `tests/eval/AUTHORING.md`.

---

## What the smoke battery is

A reproducible, automated run of all 3 handlers in both modes (where applicable) against 20 positions stratified from the Lichess Puzzle DB. Per (position, handler, mode) tuple we capture:

- Source (`llm` / `llm_retry` / `templated` / `CRASH`)
- Validation status (`validation_ok`, validation errors)
- Latency (`handler_s`, `translate_s`)
- LLM call count + token consumption
- Prose output verbatim (for eyeball review)
- Whether the focal move (puzzle solution) matches Leela's MCTS pick at 1000 nodes (Branch A vs Branch B signal)

20 positions × 5 runs per position = 100 calls. Pod time ~30-60 min wall clock.

## What it measures

- **Validation pass rate** across distributions. CLAUDE.md §2 Rule 2 satisfaction at scale.
- **Hallucinated-move count.** Rule 1 satisfaction — the validator catches these.
- **Latency distribution** including p95 / p99 tail.
- **Templated-fallback rate** per distribution slice.
- **Prose-pattern generalisation** by eyeball rubric (see below) — does v5's deliberation sentence work outside the walkthrough Rxd4 fixture? Does v2's substantive-finding pattern hold on harder puzzles?
- **Edge-case surfacing:** terminal positions, mate-in-N, very long PVs, opening book exits, known endgame-theory-gap positions.
- **Branch A vs Branch B coverage** per rating band (focal_uci == mcts_bestmove_uci vs ≠).

## What it does NOT measure

- **Prose quality / 2200+-player agreement.** That still needs G0.2 friends-authored commentaries in `eval_positions.yaml`.
- **Phase 1 exit gate compliance.** 20 positions ≠ the 30-position graded-eval gate in CLAUDE.md §11.

## How sampling works

`leela_chat/scripts/run_smoke_battery.py` calls `sample_stratified` from `leela_chat/eval/lichess_puzzle_db.py` with a fixed strata configuration (constant `STRATA` in the script). The strata cover four user-facing categories:

| Category | Count | Sub-strata (script-internal) | Filter highlights |
|---|---|---|---|
| Tactical puzzles | 8 | `tactical_fork` / `tactical_pin` / `tactical_sac` / `tactical_disc` / `tactical_skewer` / `tactical_mateInN` | Rating bands 1800-2200 through 3400+; specific theme per band; fallback_to_lower for the rare 3400+ band |
| Quiet middlegames | 5 | `quiet_middlegame` | Rating 1600-1900; theme=`middlegame`; excludes all tactical themes for genuine positional content |
| Endgames | 4 | `endgame` | Rating 1500-2200; theme=`endgame`; EXCLUDES `KNN_vs_K` material (project memory: endgame-theory gap) |
| Openings | 3 | `opening` | Rating 1700-2000; theme=`opening` |

Sampling is deterministic via `seed=0` — same corpus + same seed → same 20 positions. The generated fixture is committed to `tests/eval/smoke_positions.yaml` for reproducibility; subsequent runs read the file rather than re-sampling (when invoked with `--reuse-fixture`).

## How to extend

To add a new category or new strata:

1. Edit `STRATA` in `leela_chat/scripts/run_smoke_battery.py`. Add entries with `themes`, `n`, `rating_range`, and (optionally) `exclude_themes`, `exclude_material_signatures`, `fallback_to_lower`.
2. Edit `CATEGORY_OF` to map any new strata → user-facing category names.
3. Bump the chunk's plan numbering if the additions materially change the smoke-battery's scope.

To add a new derived-percentage class to the validator (per CLAUDE.md §8 implementation-pitfalls "Validator derivation"), see `leela_chat/validator/__init__.py::_collect_derived_percentages`.

## Eyeball rubric (post-run analysis)

For each (category × handler × mode) cell, pick the FIRST position's output deterministically and apply the 4-Y/N rubric:

1. **Position-character match** — tactical → tactical prose; quiet → positional language; endgame → endgame language.
2. **Grounding** — no claims missing from findings JSON.
3. **Forbidden-language check** — scan for "monotonic" / "recovered to" / "noticed" / "realized" / "discovered" / "saw" / "found" (per `INTERPRETABILITY_NOTES.md` §3 encoding-vs-use + trough-prose contracts).
4. **Over-claim check** — substantive-finding pattern only fires when `network_dismissal == "never_rose"` AND `peak_rank > 10`. Over-claim if it fires on a top-5 move.

Capture verbatim in `tests/eval/smoke_summaries/<YYYY-MM-DD>.md` with one row per cell. Future smoke runs reuse the same template for longitudinal comparison.

## Storage layout

- `tests/eval/smoke_positions.yaml` — fixture (git-tracked; reproducible from seed=0).
- `tests/eval/smoke_summaries/<YYYY-MM-DD>.md` — per-run dated summary (git-tracked).
- `logs/smoke_battery_<timestamp>.jsonl` — raw per-call telemetry (gitignored; lives on the pod).

## See also

- `leela_chat/eval/lichess_puzzle_db.py` — the sampling API.
- `leela_chat/scripts/download_lichess_puzzles.py` — corpus download.
- `tests/eval/AUTHORING.md` — graded-eval-set authoring guide (different artifact).
- `docs/INTERPRETABILITY_NOTES.md` §3 — prose contracts the eyeball rubric tests against.
- `docs/PROMPT_PATTERNS.md` — design patterns the smoke battery confirms or refutes.
