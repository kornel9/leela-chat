# Plan: Phase 2 Chunk 6 (v2) — `did_you_consider` with dual-timescale dismissal

> **Snapshot.** Frozen copy of
> `/home/kornel/.claude/plans/tell-me-and-explain-mutable-axolotl.md`
> (Claude Code plan-mode workspace), committed to the repo so the plan
> travels with the code in git history. The plan-mode workspace copy
> remains the live source updated by future plan-mode passes; this
> file is a snapshot. Updates land via manual recopy on plan
> revisions.

## Context

Two prior reviewer passes have refined this plan. The first pass (v0 → v1) fixed multi-search stitching, demoted linear logit-lens, and added epistemic-humility framing. The v1 implementation shipped:

- `8d17f36` — Chunk 6a v0: methodology doc + multi-budget Q-trajectory + tests.
- `452a4b1` — Chunk 6a v1: rewrote 6a to single-search live-stats + zero-ablation as the 6b default in the methodology doc + Revision-3 prose contract + monotonicity-assertion weakening.

A second-pass reviewer surfaced **nine** items (A1, B1, B2, C1, C2, C3, D1, D2, D3). All are valid. Three classes of work:

1. **Close the last outstanding item from prior review** — A1 divergent fixture was planned but not built. Make it a hard gate before 6c.
2. **Align doc with code** — B1 (§6.1 stale, describes linear logit-lens), B2 (§6.2 stale, describes multi-budget). Plus C1 (single-growing-doc pattern will calcify) and C3 (§6.5 mislabeled as a "limitation" rather than a methodological commitment).
3. **Strategic improvements driven by the cost-isn't-a-constraint clarification + product-goal clarification (coach, not analyst)** — C2 (denser MCTS-trajectory capture with summary statistics + inflection-point detection); D1 (chess-content extractors as a new chunk after 6.5); D2 (Phase 4 placeholder for coach-voice prompt migration); D3 (start the Phase 3 concept-candidates tracking doc now).

User clarifications confirmed in this session:

- **Doc structure (C1):** Option B — per-extractor module docstrings + thin `INTERPRETABILITY_NOTES.md` index. Methodology travels with code; cross-cutting principles (premise, two-timescale framework, prose contract) stay in the index.
- **Trajectory storage (C2):** keep ALL captured live-stats blocks plus summary statistics on the findings JSON. Prompt-facing serialization will down-sample to ~5-8 content-aware points; the full stream stays available for offline methodology validation and debug runs.

This revision inserts a **new sub-chunk 6a v2** between the shipped 6a v1 and 6b, plus adds a **Chunk 7** (chess-content extractors) after 6.5 and a Phase 4 placeholder. 6b and 6c are unchanged in scope, but they benefit from the work landed in 6a v2.

---

## What's shipped

- 6a v0 (`8d17f36`) — schema, multi-budget extractor (since replaced), methodology doc v0.
- 6a v1 (`452a4b1`) — single-search live-stats extractor, methodology doc rewritten for §3/§4/§5.3/§6.5, monotonicity assertions weakened, parser bug fixed. Full unit suite at 95 tests.

---

## The v2 revisions

### A1 (must-close gate before 6c). Divergent-timescale test fixture

The v1 plan included a divergent fixture in 6c verification, but 6a v1 didn't actually build it. Both walkthrough cases (Rxb1+, Rxd4) are network-keeps positions; we have no fixture for the network-low / MCTS-rescue quadrant — the architecturally most-justifying scenario for dual-timescale extraction.

**Action — land in 6a v2.** Hand-pick one position. Criteria:

- Black or White to move; non-trivial tactical content.
- Eventual MCTS bestmove has policy rank ≥ 5 at every BT4 layer.
- MCTS at 10k nodes commits ≥ 50% visit share to it.
- Confirmed by Stockfish at depth 20 as winning or substantially better than alternatives.

Sources to scan: the Jenner Look-Ahead paper's worked positions (in `update_research/pdfs/Evidence of Learned Look-Ahead in a Chess-Playing Neural Network.pdf`), or a Lichess puzzle with `quietMove` / `attraction` themes at 2200+. The leela-interp `interesting_puzzles.pkl` is not on disk and not needed for a single fixture — figshare download stays in the backlog as a future eval batch.

Add to fixtures:
- `tests/unit/fixtures/lc0/divergent_<short_name>.txt` — captured verbose-move-stats from a real Lc0 run on the position.
- Add to `FIXTURES` list in `tests/unit/test_mcts_parser.py` for the round-trip parser tests.
- Add `tests/unit/fixtures/divergent_position.yaml` (or similar) with the position metadata: FEN, expected MCTS-rescue move, Stockfish-depth-20 PV, side-to-move.

6c verification ADDS a third pod smoke using this fixture (already in the v1 plan). 6c's v1 prompt also adds a third worked example covering this quadrant. **These are still planned for 6c**; the new gate is that the fixture is **curated and committed** before 6c implementation begins.

### B1. Rewrite the linear-logit-lens limitations section

`docs/INTERPRETABILITY_NOTES.md` §6.1 currently describes linear logit-lens limitations with a mitigations list. The code now does zero-ablation; §6.1's mitigations describe risks we no longer take.

**Action — in 6a v2 (and the new module-docstring location, per C1):**

Move methodology specifics out of `INTERPRETABILITY_NOTES.md` §3 + §6.1 + §6.4 into `leela_chat/extractors/layer_emergence.py`'s module docstring (the file gets created scaffolded in 6a v2; full implementation lands in 6b). The new docstring section reads, in essence:

> *"We use zero-ablation logit-lens (Iterative Inference, arXiv 2508.21380). For each layer L, a forward pass with a hook returning `torch.zeros_like` on `blocks.{L}.hook_resid_post` masks layers L..N. The model's own `PolicyHead` reads the modified final residual. Cost ~15× one forward pass = ~750 ms on 4090. Linear logit-lens — apply the trained PolicyHead directly to cached intermediate residuals — is available as a future optimization (Chunk 6.5+) but is **not in the codebase**. It can produce plausible-looking but mechanistically meaningless output because the attention-based head's parameters are tuned to final-residual geometry; the final-layer top-1 match check cannot distinguish real signal from geometric noise. Equivalence to zero-ablation must be empirically demonstrated (concordant top-3 across ≥5 diverse positions, including non-trivial-depth ones) before linear is adopted. We do not claim early-layer distributions are calibrated probabilities; we use them as rank signals."*

This text is the SOURCE OF TRUTH for the methodology. `INTERPRETABILITY_NOTES.md` keeps only a one-paragraph cross-cutting summary + a pointer to the module.

### B2. Update the search-time stale text + extend with C2

`INTERPRETABILITY_NOTES.md` §6.2 still says "Our 5-point trajectory" — describing multi-budget. Code does live-stats now.

**Action — in 6a v2:**

Move methodology specifics out of `INTERPRETABILITY_NOTES.md` §4 + §6.2 into `leela_chat/extractors/mcts_trajectory.py`'s module docstring. The new docstring section reads, in essence:

> *"Within a single Lc0 search, we stream `--log-live-stats` and snapshot at every emit cycle. Cost is one search, not K. The data is genuinely continuous within one tree, not stitched across independent searches. We retain ALL captured blocks on `MCTSChild.trajectory_full` plus a `trajectory_summary` (peak-Q, trough-Q, q-stabilized-at, inflection points) and a content-aware down-sampled selection on `trajectory_selected` for prompt use. Real search-time dynamics happen across hundreds of visits; the dense stream captures fine oscillations, and the summary distills the prose-relevant facts (peak / trough / inflection). Five-point milestone snapshots, the v0/v1 default, are now one possible down-sample policy among several."*

The C2 code extension (dense capture + summary) is described next.

### C1. Methodology doc structure — per-extractor docstrings + thin index

Decision locked: **Option B**.

`docs/INTERPRETABILITY_NOTES.md` is reorganized to ~150 lines covering only cross-cutting concerns:

- §1 Premise — extracting deliberation, not just verdict (kept)
- §2 The two timescales — network-time + search-time (kept, shortened)
- §3 Prose contract (NEW heading, was §6.5 buried under "Limitations") — encoding-vs-use distinction; forbidden / allowed language lists; this is a **commitment**, not a limitation
- §4 Worked example (kept)
- §5 Per-extractor methodology pointers — one paragraph per extractor pointing at the module docstring as source of truth
- §6 Cross-cutting limitations (NOT per-extractor) — BT4-specificity, no causal claim, methodology-vs-code drift risk, BT4 weight pin lock
- §7 References (kept)
- §8 Future work (kept)

Methodology specifics MOVE to module docstrings:

- `leela_chat/extractors/mcts_trajectory.py` — search-time methodology (was §4 + §6.2 in doc).
- `leela_chat/extractors/layer_emergence.py` — network-time methodology (was §3 + §6.1 in doc). File doesn't exist yet; 6a v2 scaffolds a module docstring that 6b's implementation builds against.

This restructuring is the FIRST commit of 6a v2 — code changes in 6a v2 land against the restructured doc.

### C2. Dense MCTS-trajectory capture + summary statistics + content-aware down-sampling

The single-search live-stats infrastructure (6a v1) emits one verbose-move-stats block per info update — that's potentially 20-100+ blocks per 10k-node search. The v1 extractor down-samples to 5 milestone snapshots and discards the rest. With cost not a constraint, denser capture is essentially free and meaningfully improves prose quality.

**Action — in 6a v2:**

New Pydantic model in `leela_chat/extractors/mcts_parser.py`:

```python
class MCTSTrajectorySummary(BaseModel):
    q_first: float
    q_last: float
    q_peak: float
    q_peak_at_visits: int        # the candidate's visits when q hit peak
    q_trough: float
    q_trough_at_visits: int
    q_stabilized_at_visits: int | None   # first visit-count where subsequent q variation ≤ 0.05
    visit_growth_rate: float     # avg new visits per emit cycle
    q_crossed_zero: bool          # did q's sign change at any point?
    inflection_points: list[MCTSTrajectoryPoint]   # local maxima/minima of Q
    block_count: int              # total live-stats blocks for this candidate
    model_config = ConfigDict(frozen=True)
```

Extended `MCTSChild`:

```python
class MCTSChild(BaseModel):
    ... existing fields ...
    trajectory: list[MCTSTrajectoryPoint] = []          # v0 field; deprecated alias, populated == trajectory_selected
    trajectory_selected: list[MCTSTrajectoryPoint] = [] # content-aware down-sample for prompts (~5-8 points)
    trajectory_full: list[MCTSTrajectoryPoint] = []     # every captured block (per user pref C2 choice = "all")
    trajectory_summary: MCTSTrajectorySummary | None = None
```

`trajectory` keeps the existing field name and is set to `trajectory_selected` on populate, so existing handlers and prompts (the v0/v1 contract) continue to work unchanged. `trajectory_full` and `trajectory_summary` are new and opt-in.

**New helpers in `mcts_trajectory.py`:**

- `_compute_trajectory_summary(points: list[MCTSTrajectoryPoint]) -> MCTSTrajectorySummary` — single pass over the full per-candidate sequence; computes peak / trough / stabilized / inflection points.
- `_detect_inflection_points(points: list[MCTSTrajectoryPoint], *, q_window: float = 0.05) -> list[MCTSTrajectoryPoint]` — find local maxima/minima of Q over visit-progression after a smoothing window. Robust to MCTS noise.
- `_content_aware_downsample(points: list[MCTSTrajectoryPoint], *, target_n: int = 6) -> list[MCTSTrajectoryPoint]` — always include first, last, peak-Q, trough-Q, q-stabilized-point (if exists), then fill the remaining slots with snapshots near inflection points or by even spacing in the visit-domain. Dedupe.

**Extractor flow changes:**

`extract_mcts_trajectory` now:

1. Run one Lc0 search with live-stats (unchanged from v1).
2. Block-group + parse every block (unchanged).
3. Build per-candidate `trajectory_full` from every block.
4. Compute `trajectory_summary` per candidate.
5. Build `trajectory_selected` via content-aware down-sampling.
6. Populate the legacy `trajectory` field with `trajectory_selected` for backward compat.

The `budgets` argument becomes an optional override for down-sample milestones (kept for backward compatibility); when None or omitted, the content-aware down-sample is used. Existing v1 callers passing `budgets=[500, 1000, 2500, 5000, 10000]` get the closest-snapshot-to-each-milestone behaviour as a fallback for back-compat. New callers omit `budgets` and get content-aware sampling.

**Prompt-facing serialization (handler layer, not extractor):**

Handlers default to passing `trajectory_summary` + `trajectory_selected` to the LLM. `trajectory_full` is available on the findings JSON but is **not** serialized into the LLM prompt by default (it would dominate the prompt budget without proportional information gain). A `--debug-trajectory` CLI flag (Chunk 6.5 or later) can opt into full-trajectory rendering for offline inspection.

### C3. Reframe §6.5 as a methodological commitment, not a limitation

`§6.5` (encoding-vs-use, forbidden/allowed language lists) is currently sub-filed under §6 "Limitations and honest caveats." It's not a limitation — it's a **contract** with the prose. ADR-0007's behavioral-test requirement for probes is analogous: the requirement isn't a "limitation," it's how the project keeps interpretation honest.

**Action — in 6a v2:** Promote §6.5 to its own top-level section `§3. Prose contract for layer-emergence and trajectory outputs`. Make explicit: this contract is **permanent**, applies to outputs of all observational extractors, and is NOT relaxed at later phases (only causal extractors — Phase 6 counterfactuals — earn the right to use causal language, and only about their own outputs).

The forbidden/allowed lists stay identical. The framing changes.

### D1. Chess-content extractors as Chunk 7 (after 6.5)

User has clarified the product goal: a **usable chess coach**, not an analyst tool that narrates engine internals. Coach-voice prose needs chess content to ground in. Several cheap extractors are available without interp work:

- **Stockfish full PVs at `multipv=5, depth=20+`** — current `extract_stockfish_eval` returns shallower / fewer PVs; bumping both is a small parameter change.
- **Threat detection from python-chess** — hanging pieces, loose pieces (attacked, defended exactly once with worse defender), pinned, skewered. Pure Python helpers in a new `leela_chat/extractors/threats.py`. ~100-150 lines.
- **Material delta on resulting position** — `material_change(board, move)`. Trivial.
- **Simple tactical-motif detection** — forks (your piece attacks 2+ of opponent's pieces from one square), pins (you attack a sliding piece; an opponent piece behind it is more valuable), skewers (reverse of pins), discovered attacks (your move uncovers an attack from another piece). ~200-300 lines in `leela_chat/extractors/motifs.py`.
- **King safety scalar** — pawn shield count, opponent attacker count near king, open files near king. ~50-100 lines in `leela_chat/extractors/king_safety.py`. Heuristic, not learned.

**Action — reserve as Chunk 7** in the phase-roadmap. Detailed plan when 6c lands and we know what specific chess content the dual-timescale prompts actually need to ground in. The 6.5 retrofit may also surface specific gaps.

NOT in 6c (6c stays focused on dual-timescale methodology). NOT in 6a v2 (out of scope; 6a v2 is methodology-fix + trajectory-enrichment only).

### D2. Phase 4 placeholder: coach-voice prompt migration

Phase 1-2 prompts are deliberately telemetry-voice for safety while extractors are thin. By Phase 3 (probes) + Phase 4 (SAE features) + Chunk 7 (chess-content extractors), the findings JSON is rich enough to support coach-voice prose. That's a prompt-engineering project of its own, not a Phase 8 polish item.

**Action — add to PHASE_NOTES.md Phase 4 placeholder section.** Scope sketch:

- Rewrite each handler's prompt from "narrate the findings" to "give the player a coaching takeaway grounded in the findings."
- Voice-A/B harness for comparing telemetry-prompt vs coach-prompt on the same findings JSON (deterministic LLM responses where feasible; otherwise N-sample averages).
- Validation that the coach voice doesn't degrade Rule-1 grounding — the coach-voice prompts must still pass the validator on every output.
- Eval-set rerun comparing the two voices on the same eval positions.

No implementation now; just reserve the slot in the roadmap so it's not treated as polish.

### D3. Start the Phase 3 concept-candidates tracker now

The Phase 3 probe concept list is the single most product-defining decision in the project. Cold-starting it at Phase 3 entry leaves quality on the table.

**Action — in 6a v2:** Create `docs/phase3/concept_candidates.md`. Initial skeleton:

- Header explaining the doc's purpose: live tracker for probe-concept candidates, grouped by category. Cross-referenced against published curricula (Silman's *How to Reassess Your Chess*, Aagaard's *Grandmaster Prep* series, Watson's *Mastering the Chess Openings*) to pressure-test coverage. Nothing committed; the list grows over time and gets pruned at Phase 3 entry.
- Categories (seed with ~3-5 candidates each):
  - **Tactical motifs:** pins (absolute, relative), forks, skewers, discovered attacks, batteries, removal of defender, deflection, decoys, in-between moves (zwischenzug), overloads.
  - **Pawn structure:** isolated pawn (IQP), doubled pawns, hanging pawns, pawn chains (good/bad), passed pawns, backward pawns, candidate passers, pawn breaks, holes.
  - **Piece activity:** outposts, weak squares, knight on the rim, bishop pair, opposite-coloured bishops, good knight vs bad bishop, rook on the 7th, rook lifts.
  - **King safety:** pawn shield, attacker count, open files near king, exposed king, opposite-side castling, h-pawn pushes.
  - **Endgame patterns:** opposition, key squares, Lucena, Philidor, rook activity, king activation, fortress.
  - **Strategic concepts:** prophylaxis, restriction, two weaknesses, exchange to ease pressure, exchange to keep attacker.

This doc is **NOT a Phase 3 plan**. It's a candidate pool. Phase 3 entry brief will prune it based on what's actually probeable, dataset-available, and evaluation-meaningful.

---

## Revised sub-chunk structure

| # | What | Status |
|---|---|---|
| 6a v0 | Methodology doc + multi-search Q-trajectory | ✅ `8d17f36` |
| 6a v1 | Single-search live-stats; methodology rewrites for §3/§4/§5.3/§6.5 | ✅ `452a4b1` |
| **6a v2** | Doc restructure (Option B); §6.1+§6.2 rewrites (now in module docstrings); §6.5 promoted to §3 "Prose contract"; divergent fixture (A1); dense MCTS-trajectory capture + summary + inflection-points + content-aware down-sampling (C2); Phase 3 concept-candidates skeleton (D3) | **next** |
| 6b | Zero-ablation `extract_layer_emergence` | new |
| 6c | `did_you_consider` handler + v1 prompt (three worked examples covering all three signal quadrants) | new |
| 6.5 | Gated retrofit of Handlers 1 + 2 with opt-in `--detailed` flag | gated on 6c criteria |
| **7** | **Chess-content extractors** — Stockfish multi-PV, threats, material delta, motifs, king safety | new, post-6.5 |
| Phase 4 | (Placeholder) Coach-voice prompt migration | reserved |

---

## Critical files to create / modify in 6a v2

| Path | Change | Notes |
|---|---|---|
| `docs/INTERPRETABILITY_NOTES.md` | REWRITE — slim to ~150 lines: cross-cutting only. §1 Premise, §2 Two timescales, §3 Prose contract (was §6.5; promoted + renamed per C3), §4 Worked example, §5 Per-extractor pointers, §6 Cross-cutting limitations, §7 References, §8 Future work | Per-extractor methodology specifics MOVE to module docstrings (C1) |
| `leela_chat/extractors/mcts_trajectory.py` | EXPAND module docstring with full search-time methodology (was §4 + §6.2 in old doc). ADD `_compute_trajectory_summary`, `_detect_inflection_points`, `_content_aware_downsample`. EXTEND `extract_mcts_trajectory` to populate `trajectory_full`, `trajectory_summary`, `trajectory_selected` in addition to the legacy `trajectory` field. | C1 + C2 |
| `leela_chat/extractors/mcts_parser.py` | ADD `MCTSTrajectorySummary` Pydantic model. EXTEND `MCTSChild` with `trajectory_full`, `trajectory_selected`, `trajectory_summary` fields (all defaulting empty / None). Keep `trajectory` field for backward compat; populate it from `trajectory_selected`. | C2 schema |
| `leela_chat/extractors/layer_emergence.py` | NEW — scaffold ONLY (module docstring + placeholder function signature; full implementation lands in 6b). Module docstring carries the full network-time methodology (was §3 + §6.1 in old doc). | C1 (docstring lands in 6a v2 so 6b implementation has a written contract) |
| `tests/unit/fixtures/lc0/divergent_<name>.txt` | NEW — hand-picked divergent-timescale position's Lc0 verbose-move-stats capture. | A1 |
| `tests/unit/fixtures/lc0/divergent_<name>.meta.yaml` | NEW — FEN, expected MCTS-rescue move, Stockfish-depth-20 PV, side-to-move. | A1 |
| `tests/unit/test_mcts_parser.py` | EDIT — add the divergent fixture to `FIXTURES` for round-trip parser coverage. | A1 |
| `tests/unit/test_mcts_trajectory.py` | EDIT — add tests for the summary statistics, inflection-point detection, content-aware down-sample. Existing 18 tests preserved. | C2 |
| `docs/phase3/concept_candidates.md` | NEW — Phase 3 concept-candidates tracker skeleton (categories + seed entries). | D3 |
| `docs/PHASE_NOTES.md` | EDIT — add Phase 4 "coach-voice prompt migration" placeholder; add Chunk 7 "chess-content extractors" entry; update Active Chunk header for 6a v2. | D2 |

---

## Design details — C2 in depth

### Data model

```python
class MCTSTrajectoryPoint(BaseModel):
    nodes_searched: int
    visits: int
    q: float
    model_config = ConfigDict(frozen=True)


class MCTSTrajectorySummary(BaseModel):
    """One-shot summary of a candidate's full intra-search trajectory."""
    q_first: float                  # Q at the first emit cycle the candidate appeared in
    q_last: float                   # Q at search end
    q_peak: float
    q_peak_at_visits: int           # this candidate's visit count when q_peak happened
    q_trough: float
    q_trough_at_visits: int
    q_stabilized_at_visits: int | None   # first visit count after which q's variation ≤ 0.05 for the remainder; None if never
    visit_growth_rate: float        # avg visit-increment per emit cycle the candidate was present in
    q_crossed_zero: bool            # True if q changed sign at any point
    inflection_points: list[MCTSTrajectoryPoint]   # smoothed local maxima/minima
    block_count: int                # how many live-stats blocks this candidate was present in
    model_config = ConfigDict(frozen=True)


class MCTSChild(BaseModel):
    ... existing fields ...
    trajectory: list[MCTSTrajectoryPoint] = []          # legacy field; equal to trajectory_selected when populated
    trajectory_selected: list[MCTSTrajectoryPoint] = [] # 5-8 content-aware points for prompt use
    trajectory_full: list[MCTSTrajectoryPoint] = []     # every captured block (per user preference: store all)
    trajectory_summary: MCTSTrajectorySummary | None = None
```

### Down-sample policy (per call)

Two paths into `extract_mcts_trajectory`:

1. **Caller passes `budgets=[...]`** (legacy v1 contract). Pick closest-snapshot-to-each-milestone, as in v1. Populate `trajectory_selected` from those picks, and `trajectory` for backward compat.
2. **Caller omits `budgets`** (new v2 default). Use `_content_aware_downsample(points=trajectory_full, target_n=6)`. Populate `trajectory_selected` from the content-aware picks.

Either way, `trajectory_full` and `trajectory_summary` are always populated.

### Inflection-point detection

Trivial first version, robust to noise:

1. Smooth Q with a 3-point moving average over the candidate's visit-ordered trajectory.
2. Find local maxima / minima of smoothed Q (a point where smoothed_q[i-1] < smoothed_q[i] > smoothed_q[i+1] is a local max; reverse for min).
3. Filter: keep only inflections where the magnitude of the change from previous inflection (or start) is ≥ 0.05.
4. Return up to ~5 inflection points (more than 5 indicates noise dominating signal; cap and log).

### Content-aware down-sample policy

Build `target_n = 6` snapshots in this priority order, dedup-by-`nodes_searched`:

1. First-visit snapshot (smallest `nodes_searched` where candidate had visits > 0).
2. Last snapshot (deepest `nodes_searched`).
3. q-peak snapshot.
4. q-trough snapshot.
5. q-stabilized snapshot, if it exists.
6. If slots remain, fill with snapshots evenly-spaced through the inflection-point list.
7. If still slots remain, fill with even spacing through the full trajectory in visit-domain.

If `target_n` < 4, drop priorities 5-7. If `target_n` > 6, fill remaining slots via even spacing.

### Test coverage additions

In `tests/unit/test_mcts_trajectory.py`:

- Summary computation: build a synthetic trajectory of 20 points spanning visit counts [10, 1000] with a Q curve that peaks at visit 200, troughs at visit 600, stabilizes after visit 700. Assert summary correctly identifies all four marker points.
- Inflection-point detection: build a trajectory with two clear inflections + noise; assert exactly two inflections returned.
- Content-aware down-sample: build a 30-point trajectory; assert the 6-point down-sample includes first, last, peak, trough, stabilized.
- Backward compat: a caller passing `budgets=[500, 1000, 2500, 5000, 10000]` still gets a `trajectory` field with 5 closest-milestone snapshots (v1 behaviour preserved).

---

## Reused functions and modules (carried over from v1)

- `leela_chat/extractors/mcts.py` — `extract_mcts` reused in the fallback path.
- `leela_chat/extractors/mcts_parser.py` — `parse_verbose_move_stats` reused per block.
- `leela_chat/engine/lc0_engine.py` — `analyse_verbose(..., log_live_stats=True)`.
- `leela_chat/_vendor/transformer_lens/hook_points.py:417` — `run_with_hooks` for zero-ablation (6b).
- `leela_chat/handlers/why_this_move.py` — `WhyThisMoveFindings` shape mirrored for `DidYouConsiderFindings` (6c).
- `leela_chat/cli/__main__.py` — `_answer_one` dispatch (6c).

---

## Out of scope (for 6a v2)

- 6b implementation (next chunk; 6a v2 only scaffolds the module docstring).
- 6c handler + prompt.
- 6.5 retrofit.
- Chunk 7 chess-content extractors (post-6.5; planned separately).
- Phase 4 coach-voice migration (just a placeholder roadmap entry).
- Phase 3 concept-candidates beyond the initial skeleton.
- leela-interp figshare download (still backlog; one hand-picked fixture suffices).
- Linear logit-lens equivalence study (still 6.5+ backlog).
- Maia / "practical-best" backlog.
- A "thinking timeline" composite visualisation.

---

## Verification — 6a v2

### Methodology doc restructure (C1 + B1 + B2 + C3)

1. `docs/INTERPRETABILITY_NOTES.md` is now ~150 lines, contains only cross-cutting content. §3 is "Prose contract for layer-emergence and trajectory outputs" (was §6.5, promoted). §6 has BT4-specific limitations, no causal claim, methodology-vs-code drift risk — and only those.
2. `leela_chat/extractors/mcts_trajectory.py` module docstring carries the full search-time methodology (was §4 + §6.2). Includes the C2 summary-statistics methodology.
3. `leela_chat/extractors/layer_emergence.py` exists as a scaffold (module docstring + placeholder function signature). Module docstring carries the full network-time methodology (was §3 + §6.1), zero-ablation only. Linear logit-lens explicitly not in the codebase, with the equivalence-criterion documented.
4. No `noticed / realized / discovered / saw / found` language anywhere in the methodology doc or any module docstring (grep -E grep -ri to confirm).

### Divergent fixture (A1)

1. `tests/unit/fixtures/lc0/divergent_<name>.txt` committed; a `.meta.yaml` (or similar) sibling records FEN + expected MCTS-rescue move + Stockfish PV.
2. `tests/unit/test_mcts_parser.py` round-trip test passes on the new fixture.
3. Manual eyeball: in the fixture's Lc0 capture, the MCTS-rescued move has high visits AND low policy prior (the divergence signature). If the picked position doesn't show this signature in real Lc0 output, **pick another** before committing the fixture.

### Dense MCTS-trajectory capture (C2)

1. `MCTSTrajectorySummary` Pydantic model + extended `MCTSChild` fields validate cleanly under `pytest tests/unit/test_mcts_parser.py`.
2. New unit tests in `tests/unit/test_mcts_trajectory.py` (summary, inflection-point detection, content-aware down-sample, back-compat) all pass.
3. Full unit suite green: target ~100+ tests (was 95 in v1).
4. Pod integration (separate from this chunk's verification — runs whenever Lc0 + BT4 are present): the existing `test_rxd4_q_confirms_within_one_search` and `test_rxb1_check_refuted_within_one_search` still pass; new `trajectory_summary` fields populate sensibly (peak / trough / stabilized values match manual eyeball of the printed trajectory).

### Phase 3 concept-candidates tracker (D3)

1. `docs/phase3/concept_candidates.md` exists with the structure described in §D3 above and seed entries in each category.
2. PHASE_NOTES.md notes the doc's existence and that it grows over time.

### Phase 4 placeholder (D2)

1. PHASE_NOTES.md has a Phase 4 entry for "coach-voice prompt migration" with the sketched scope.

### Commit + push

One commit for the doc restructure (small diff, large semantic change); a second commit for the C2 code work; a third for the divergent fixture + D3 + D2 docs. Three commits keeps the diff reviewable.

---

## Verification — 6b and 6c (unchanged from v1 except where noted)

### 6b

Unchanged from v1 plan. Zero-ablation logit-lens; pod hook-wiring sanity (layer-N ablation = no ablation = baseline matches `extract_network_inference`); methodology check on Rxd4 walkthrough position (per-layer rank trajectory printed; tag derivation consistent with intuitive read; pivot to mean-ablation if degenerate). The methodology doc for 6b ALREADY EXISTS as the `layer_emergence.py` module docstring from 6a v2.

### 6c

Unchanged from v1 plan except: the **third worked example** (network-low / MCTS-rescue) now has a real fixture (committed in 6a v2). The third pod smoke uses that fixture directly. The v1 prompt's third example anchor IS the fixture's FEN + move.

---

## Backlog

- **Linear logit-lens equivalence study** — 6.5+ candidate, gated on demonstrated zero-ablation cost bottleneck.
- **leela-interp `interesting_puzzles.pkl` figshare download** — future eval batch.
- **Q-trajectory visit-by-visit precision** — would need Lc0 source patch; not currently planned.
- **Methodology doc consolidation per-phase** — if Option B module-docstring approach has cross-cutting reading gaps as more chunks land, revisit at Phase 4.
- **Maia / "practical-best" backlog** — still tracked; Chunk 7 candidate.
- **`prior_pct` ≠ `policy_pct` divergence** in `network_inference` — still orthogonal.
- **`setup_pod.sh BT4.pt` idempotency** — separate hygiene fix.
- **Voice-A/B harness for coach-voice migration (D2)** — Phase 4 work item.

---

## What this revision does NOT change (from v1)

- Sub-chunk decomposition for the core trilogy: 6a → 6b → 6c → 6.5.
- Design-doc-first discipline (methodology precedes code).
- Zero-ablation as 6b default; linear logit-lens stays out of the codebase.
- Single-search live-stats for trajectories.
- Pre-computed dismissal tags (`rose_and_stayed`, `explored_and_refuted`).
- Five categorical eval buckets from Chunk 5.
- Opt-in `--detailed` pattern for 6.5.
- `MCTSTrajectoryPoint` schema.
- The Revision-3 prose contract — only its FRAMING changes (commitment, not limitation); the forbidden / allowed language lists stay byte-identical.
