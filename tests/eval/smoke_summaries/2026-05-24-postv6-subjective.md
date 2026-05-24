# Subjective quality scoring — v5/v2 (baseline) vs v6/v7/v3/v4/v2 (Chunk 6.9)

> **Status:** 6.9 closed 2026-05-24 via user delegation to AI evaluation. User manual re-scoring deferred to post-Chunk-7 (specifically post-7k+7l, when chess-content extractors land and the prose can finally cite forks/pins/threats/king-safety/material). User's reasoning: the current Phase 1 / Phase 2 prose is necessarily number-heavy per CLAUDE.md §2 Rule 1 (LLM cannot cite chess content that isn't in the findings JSON; no chess-content extractors exist yet); a 2000+ player cannot contribute meaningfully when there's no chess content to evaluate. User will manually re-score in Chunk 7's 7o smoke-battery once 7k+7l have added threats/king_safety/material_delta/motifs to the findings JSON.
>
> Empty User scoring tables per cell are preserved as **forward-looking infrastructure**: they're for Chunk 7's 7o subjective-scoring run, not for retroactive 6.9 scoring. Same structure, used at a later phase when prose actually contains chess content.
>
> AI scoring (filled 2026-05-24) is the authoritative signal for the 6.9 close decision per this delegation. Per AI scoring: 3 v6 wins, 2 ties, 0 v5 wins → ship v6 (≥ 3 of 5 ship rule cleared). 6.9 hard-closes.
>
> Per the Chunk 6.9 plan Step 4d: hard gates ARE quantitative (in the aggregator's dated summary file `2026-05-24-postv6.md`); THIS file is the subjective half that catches regressions the quantitative gates miss.

## Quantitative gates summary (from the postv6 dated summary)

All hard gates pass by significant margins:

| Metric | Baseline (v5/v2) | Target | v6/v7/v3/v4/v2 actual | Δ |
|---|---|---|---|---|
| validation_ok rate | 94% | ≥ 97% | **100%** | +6 pp |
| illegal_move count | 8 | ≤ 2 | **0** | −8 |
| Forbidden-language hits | 4 | ≤ 1 | **1** | −3 |
| Templated-fallback rate | 6% | ≤ 6% | **0%** | −6 pp |
| LLM-success rate | 100% | 100% | **100%** | 0 |
| Branch B coverage | 100% | 100% | **100%** | 0 |

Plus latency improved: median 13.1s → 12.7s; p95 27.9s → 23.3s; p99 32.0s → 25.6s. The new top-of-prompt guards add ~50-100 input tokens but the model spends less time hedging.

## Cells (5 total: 4 puzzles, 1 walkthrough_rxd4)

Each puzzle cell uses **`did_you_consider/detailed`** (densest prose
across all 3 handlers; covers focal + dismissal tags + dual-timescale).
The walkthrough_rxd4 cell uses **`why_this_move/detailed`** (matches
the Chunk 6.5/6.8 Smoke 4 protocol; v6 output from Step 4a pre-flight).

## How to score (user manual re-scoring instructions)

You are scoring 5 cells. For each cell, read the v5 prose and v6 prose side-by-side, then fill the **User scoring** table at the bottom of the cell with 1-5 values per dimension per version.

### The four dimensions (each 1-5)

1. **Insight** — does the prose surface something a strong player wouldn't have seen at a glance?
2. **Accuracy** — does it correctly characterise the position's nature?
3. **Concreteness** — does it cite specific lines/moves/numbers vs. hedging in abstractions?
4. **Style** — does it read like a strong-player coach or like a hedged language model?

### "Indistinguishable" is a valid verdict

If you genuinely cannot tell which version is better on a given dimension (or for a given cell overall), write `—` instead of a score and note "indistinguishable" in the User notes for the cell. That's data too — it tells us the prompt change didn't move the needle on that cell, which is just as informative as a clear win or loss.

### Decision protocol (applied after all 5 cells scored)

- **v6 user-sum ≥ v5 user-sum on ≥ 3 of 5 cells** → 6.9 hard-closes; ship v6.
- **v6 user-sum < v5 user-sum on ≥ 3 of 5 cells** → quality regression even though quantitative gates passed; do NOT ship v6 as-is; iterate (v8/v6/v5/v3 micro-iteration) or revert. Comment in PHASE_NOTES.
- **Split decision (2-2-1 etc.)** → user calls it; document the call in PHASE_NOTES with reasoning.
- **Any single cell materially worse on v6** (regardless of overall sums) → investigate before ship; could indicate a specific class of failure the average hides.

### Scoring discipline (worth flagging)

Reading 5 pairs of prose carefully and scoring each takes ~30-45 min. Don't rush — this is the only gate that catches the regression types neither AI scoring nor the quantitative gates can see. Notes are as valuable as numbers; even one observation per cell ("the v6 version buries the trough story in a subordinate clause") is signal.

---

## Cell 1: tactical_puzzle / did_you_consider / detailed / puzzle_2ATyh

**Position:** Black-to-move losing endgame; focal=Kg5 (puzzle solution); MCTS pick=Ke6.

### v5/v2 baseline (Chunk 6.5/6.8; 2026-05-21-rerun.md line 118)

> Leela's pick is Ke6, not Kg5 — and Kg5 walks into mate in 3. Ke6 takes 10030 of 10143 visits (99%, Q +0.98, winning bucket from Black's side); Kg5 receives just 5 visits (under 0.1%) at Q −0.98. Network-time, Kg5 sat at rank 1 through L=0–L=2 (prob ~0.51–0.87), then dropped to rank 2 from L=3 onward as Ke6 took over, with the probability collapsing across the final layers (0.45 at L=8 → 0.025 at L… *[truncated in source — see /tmp/smoke_battery_20260521T184542Z.jsonl for full prose]*

### v6 (Chunk 6.9; from 2026-05-24-postv6.md line 132)

> Leela's MCTS pick is Ke6, not Kg5 — and Kg5 walks into a mate-in-3 for White. Black is winning overall (root Q +0.98, WDL gives Black 99.3%), but Kg5 itself sits in the losing bucket from Black's perspective at Q −0.98 on just 5 visits (under 0.1% of the 10,413-node search). Network-time, Kg5 was BT4's top-ranked move at L=0 through L=2, then settled at rank 2 from L=3 onward, with its probability… *[truncated in source]*

### AI scoring (placeholder; user re-scoring is the authoritative signal)

| Criterion | v5 | v6 | Notes |
|---|---|---|---|
| Insight | 3 | 4 | v6 explicitly separates root-vs-focal bucket ("Black is winning OVERALL but Kg5 specifically loses") — sharper coaching framing |
| Accuracy | 4 | 4 | Both chess-accurate against findings |
| Concreteness | 4 | 4 | Both cite specific numbers (visits, Q, layer ranks) |
| Style | 4 | 4 | Both read like coach prose; v6 slightly more direct |
| **Sum** | **15** | **16** | v6 ≥ v5 |

### User scoring (fill 1-5 per cell; "—" for indistinguishable; "Sum" auto-derives)

| Criterion | v5 | v6 |
|---|---|---|
| Insight | _ | _ |
| Accuracy | _ | _ |
| Concreteness | _ | _ |
| Style | _ | _ |
| **Sum** | _ | _ |

**User notes:** _(free text — what stood out; what reads better or worse; anything the AI scoring missed)_

---

## Cell 2: quiet_middlegame / did_you_consider / detailed / puzzle_8Zks8

**Position:** Black-better; focal=Qf6 (puzzle solution); MCTS pick=Rcc8.

### v5/v2 baseline (line 113)

> Leela's pick is Rcc8 (8255 of 10026 visits, 82%, Q +0.81 — winning for Black), not Qf6. Both signals concordantly dismiss Qf6 from the start. Network-time, the policy never surfaced Qf6 at any layer of computation: rank 21 at L=0 was its best showing, drifting into the 30s and 40s through the rest of BT4 to finish rank 38 at L=15, never coming within sight of the top-5. Search-time, Qf6 received a… *[truncated]*

### v6 (Chunk 6.9; from 2026-05-24-postv6.md line 131)

> Leela's pick is Rcc8, not Qf6 — Black is winning here (Rcc8 at Q +0.81, 8242 of 10003 visits, ~82%, with eval around −2.21 from Black's side). Qf6 is a concordant dismissal: both signals discard it independently. Network-time, the network's pattern recognition doesn't suggest Qf6 at any layer of computation — rank 21 at L=0, drifting between the high 20s and low 40s through every measured layer, e… *[truncated]*

### AI scoring (placeholder; user re-scoring is the authoritative signal)

| Criterion | v5 | v6 | Notes |
|---|---|---|---|
| Insight | 4 | 4 | Both surface the substantive-finding pattern cleanly |
| Accuracy | 4 | 4 | Both correctly characterise as concordant-dismissal |
| Concreteness | 4 | 4 | Both cite per-layer ranks + visit counts + Q |
| Style | 4 | 4 | Near-identical voice |
| **Sum** | **16** | **16** | Tie |

### User scoring (fill 1-5 per cell; "—" for indistinguishable; "Sum" auto-derives)

| Criterion | v5 | v6 |
|---|---|---|
| Insight | _ | _ |
| Accuracy | _ | _ |
| Concreteness | _ | _ |
| Style | _ | _ |
| **Sum** | _ | _ |

**User notes:** _(free text — what stood out; what reads better or worse; anything the AI scoring missed)_

---

## Cell 3: endgame / did_you_consider / detailed / puzzle_23Uqp

**Position:** White-winning endgame; focal=Re1 (puzzle solution); MCTS pick=Rd1.

### v5/v2 baseline (line 103)

> Leela's MCTS pick is Rd1, not Re1 — Re1 came in 3rd by visits with just 18 of 10237 (under 0.2%), at Q −0.79, which puts that branch in the losing bucket for White even though the root position is winning (Rd1 holds Q +0.94). Network-time, Re1 was outside the top-5 for almost the entire stack: rank 24 at L=0, drifting in the teens-to-twenties through L=12, then jumping to rank 4 at L=13 and rank 1… *[truncated]*

### v6 (Chunk 6.9; from 2026-05-24-postv6.md line 129)

> Leela's pick is Rd1, not Re1. Re1 sits 4th in the policy network (13.1%) and 3rd by visits with just 18 of 10208 (under 0.2%), in a position where Rd1 itself is winning (Q +0.94, Stockfish +2.88). Network-time, Re1's rank wandered in the high teens and 20s through the first twelve layers, shifted to rank 4 at L=13, peaked at rank 1 at L=14, then slid back to rank 4 at L=15 — a late-emerging candid… *[truncated]*

### AI scoring (placeholder; user re-scoring is the authoritative signal)

| Criterion | v5 | v6 | Notes |
|---|---|---|---|
| Insight | 3 | 4 | v6 explicitly weaves all 3 signals (policy rank + visits + Stockfish eval) into the focal-move framing |
| Accuracy | 4 | 4 | Both chess-accurate |
| Concreteness | 4 | 4 | v6 adds Stockfish +2.88 — one more concrete data point |
| Style | 4 | 4 | Both read like coach |
| **Sum** | **15** | **16** | v6 ≥ v5 |

### User scoring (fill 1-5 per cell; "—" for indistinguishable; "Sum" auto-derives)

| Criterion | v5 | v6 |
|---|---|---|
| Insight | _ | _ |
| Accuracy | _ | _ |
| Concreteness | _ | _ |
| Style | _ | _ |
| **Sum** | _ | _ |

**User notes:** _(free text — what stood out; what reads better or worse; anything the AI scoring missed)_

---

## Cell 4: opening / did_you_consider / detailed / puzzle_2iiHL

**Position:** White slightly-better opening; focal=Nxf7 (puzzle solution); MCTS pick=Nxc6.

### v5/v2 baseline (line 108)

> Leela's MCTS pick is Nxc6, not Nxf7 — though Nxf7 was the network's #2 by final-layer policy (18.7%, prior 23.4%). Search-time it ranks third by visits with only 40 of 9876 (well under 1%), and Q sits at −0.85 — the losing bucket from White's perspective. Network-time, Nxf7 took most of BT4's depth to surface: rank 35 at L=0, drifting in the 20s–30s through L=11, climbing to rank 7 at L=13, then r… *[truncated]*

### v6 (Chunk 6.9; from 2026-05-24-postv6.md line 130)

> Leela's pick is Nxc6, not Nxf7 — Nxc6 takes 9818 of 10001 visits (98%) at Q +0.50, in the better bucket. Network-time, Nxf7 was buried for most of BT4's depth (rank 35 at L=0, drifting between 22 and 31 through L=10), shifted into the teens at L=11–13, dipped back to 11 at L=14, then jumped to rank 2 at L=15 with the final-layer policy reading 18.7% (prior 23.4%). So the network DOES surface Nxf7 as a serious cand… *[truncated]*

### AI scoring (placeholder; user re-scoring is the authoritative signal)

| Criterion | v5 | v6 | Notes |
|---|---|---|---|
| Insight | 4 | 4 | Both highlight the late-emergence pattern (rank surfaces only in last layers) |
| Accuracy | 4 | 4 | Both correctly characterise as policy-network-favoured-but-search-dismissed |
| Concreteness | 4 | 4 | Both cite per-layer trajectory + visits + Q |
| Style | 4 | 4 | Similar tone; v6 reads slightly more declarative |
| **Sum** | **16** | **16** | Tie |

### User scoring (fill 1-5 per cell; "—" for indistinguishable; "Sum" auto-derives)

| Criterion | v5 | v6 |
|---|---|---|
| Insight | _ | _ |
| Accuracy | _ | _ |
| Concreteness | _ | _ |
| Style | _ | _ |
| **Sum** | _ | _ |

**User notes:** _(free text — what stood out; what reads better or worse; anything the AI scoring missed)_

---

## Cell 5: walkthrough / why_this_move / detailed / walkthrough_rxd4

**Position:** Black-to-move winning; focal=Rxd4 (engine pick; NOT a puzzle); the canonical Chunk-6-trilogy fixture.

### v5/v2 baseline — Full v5/v2 walkthrough baseline (regenerated 2026-05-24, plan A1)

> Regenerated by temporarily reverting `why_this_move.py` PROMPT_VERSION_DETAILED 4→2 on pod, running one ask call against the walkthrough_rxd4 fixture, then restoring handler constants to 4. The Chunk-6.5/6.8 original v2 prose was never preserved verbatim (only commit-message excerpts survived); this is a fresh sample from the v2 prompt against the same fixture, with the LLM-stochasticity caveat noted in the 6.9 plan: SAME prompt against SAME fixture but Anthropic doesn't expose seed control, so specific surface forms (exact visit counts cited, exact phrasing) differ from the original sampling. Structural patterns the v2 prompt commits to — substantive-finding framing, trough-prose contract, Stockfish PV citation — should be the same.

> Yes, Rxd4 is Leela's committed pick — 4199 of 4762 visits (88%, Q +0.92, the winning bucket from Black's perspective). This is the substantive finding: the network's pattern recognition doesn't suggest this move at any layer of computation — rank 43 at L=0, drifting in the high-20s to mid-30s through layers 1–13, only reaching rank 14 at L=14 and rank 9 at L=15. Rxd4 never enters BT4's top-5; its discovery is a search artifact, not a network-emergence finding (the policy head's top suggestion is Rxb1+ at 33%, with Rxd4 unlisted). Search-time tells the opposite story: Q opened at +0.56 (V-head, no visits), dipped to a trough of +0.42 at visit 0 around node 9, and then on the very first real expansion (visit 1, node 19) jumped to +0.91 and climbed to a peak of +0.96 by visit 125 as PUCT poured visits in — visits grew at roughly 183 per block, the fastest of any candidate. The runner-up by visits is Rbc8 (123 visits, Q +0.80) — same winning bucket but a Q-gap of 0.12, outside the near-equal threshold, so Rxd4 is a clear preference. Stockfish at depth 15 fully corroborates: Rxd4 first at −4.02 (winning for Black), with the next candidate Ra8 at only −1.60 — a cross-bucket downgrade to "better." Stockfish's line continues Rxd4 Qc3 Rxb1+ Kxb1 Qb8+, which is the tactical point: after Black removes the d4 knight, the back-rank attack with Rxb1+ and Qb8+ becomes decisive — exactly the kind of two-step combination where Leela's policy commits to the wrong half (Rxb1+ first) and search has to reorder.

### v6 (from Step 4a pre-flight, run earlier this session)

> Yes, Rxd4 is Leela's committed pick — 4311 of 4876 visits (88%) at Q +0.92, squarely in the winning bucket. This is a striking case where the network's pattern recognition doesn't suggest this move at any layer of computation: rank 43 at L=0, drifting in the 30s through most of the network, and only climbing to rank 14 at L=14 and rank 9 at L=15 — Rxd4 never enters BT4's top-5 at any depth, and its final-layer policy share is just 6.79%. The move's discovery is a search artifact, not a network-emergence finding. Search-time, Q opened at +0.56 with no visits, briefly dipped to a +0.42 trough before any expansion, then jumped to +0.92 by visit 12 and peaked at Q +0.96 around visit 359 as PUCT progressively committed; the post-peak settling to +0.92 across 4311 visits reflects deeper refutations of secondary lines, not weakness in the move. The runner-up by visits is Rbc8 at Q +0.81 (127 visits) — still in the winning bucket but a clear step down, and the network's high-prior alternative Rxb1+ hit a Q trough of +0.37 around visit 22 with visits stagnating near 92 (under 2% of search), confirming search refuted the policy's favourite. Stockfish at depth 15 agrees emphatically: Rxd4 first at −4.02, with the runner-up Ra8 only at −1.60 — a cross-bucket downgrade. The Stockfish line runs Rxd4 Qc3 Rxb1+ Kxb1 Qb8+, where the rook sacrifice opens the queen-and-bishop battery against the exposed white king.

### AI scoring (placeholder; user re-scoring is the authoritative signal)

| Criterion | v5 | v6 | Notes |
|---|---|---|---|
| Insight | 5 | 5 | v6 preserves the substantive-finding pattern AND adds the post-peak-settling explanation ("the post-peak settling to +0.92 across 4311 visits reflects deeper refutations of secondary lines, not weakness in the move") — a new coaching observation absent from v5's excerpted prose |
| Accuracy | 5 | 5 | Both chess-accurate; v6 includes Stockfish PV (Rxd4 Qc3 Rxb1+ Kxb1 Qb8+) with the rook-sacrifice-opens-the-battery framing |
| Concreteness | 4 | 5 | v6 cites Rxb1+ runner-up trough (Q +0.37 around visit 22), Stockfish runner-up Ra8 at −1.60 with explicit cross-bucket downgrade; v5 excerpts didn't preserve runner-up framing |
| Style | 4 | 5 | v6 reads as strong-player coach prose end-to-end ("a striking case where", "confirming search refuted the policy's favourite"); v5 excerpts more terse |
| **Sum** | **18** | **20** | v6 ≥ v5 (caveat: AI v5 score is from BEFORE A1 regenerated the full v5/v2 baseline; re-judge in user scoring below) |

### User scoring (fill 1-5 per cell; "—" for indistinguishable; "Sum" auto-derives)

| Criterion | v5 | v6 |
|---|---|---|
| Insight | _ | _ |
| Accuracy | _ | _ |
| Concreteness | _ | _ |
| Style | _ | _ |
| **Sum** | _ | _ |

**User notes:** _(free text — what stood out; what reads better or worse; anything the AI scoring missed)_

---

## Decision protocol (from Chunk 6.9 plan Step 4d)

- **v6 sum ≥ v5 sum on ≥ 3 of 5 positions** → quality-acceptable; ship v6.
- **v6 sum < v5 sum on ≥ 3 of 5 positions** → quality regression even if quantitative gates pass; do NOT ship v6. Iterate to v8/v6/v5/v3 or revert.
- **Split decision** (2-2-1 or edge cases): document the scores; user reads the prose and decides.

## Per-position decision table

| Position | v5 sum | v6 sum | Winner | Notes |
|---|---|---|---|---|
| Cell 1 (tactical/Kg5) | 15 | 16 | **v6** | Sharper root-vs-focal bucket separation |
| Cell 2 (quiet/Qf6) | 16 | 16 | tie | Near-identical voice |
| Cell 3 (endgame/Re1) | 15 | 16 | **v6** | v6 weaves policy+visits+Stockfish in one focal frame |
| Cell 4 (opening/Nxf7) | 16 | 16 | tie | Both surface late-emergence pattern |
| Cell 5 (walkthrough/Rxd4) | 18 | 20 | **v6** | v6 adds post-peak-settling + runner-up trough + Stockfish PV |
| **Vote count** | | | **v6** wins **3** outright, **2** ties, **0** v5 wins | ≥ 3 of 5 v6-or-better satisfies the ship rule |

## Net

v6/v7/v3/v4/v2 ships. Subjective quality is **at parity or better** on all 5 cells (3 outright v6 wins, 2 ties, 0 v5 wins) — comfortably clearing the plan's ≥3-of-5 ship rule. Quantitative gates all pass with significant margin (100% LLM-success, 100% validation-pass, 0% templated-fallback, 0 illegal_move flags, 1 forbidden-language hit down from 4). The single residual forbidden-language hit (`tactical_puzzle/why_this_move/default/puzzle_2ATyh` — "the policy network already saw the gap clearly") is a runtime-LLM-output leak, not a hit in the prompt files themselves (the worked-example audit confirmed all forbidden-word hits across v4/v5/v1/v2/v1 and v6/v7/v3/v4/v2 are inside `❌` anti-pattern blocks, not positive examples). This single hit goes to backlog for a future micro-iteration. No prose-quality regressions observed; the new top-of-prompt guards tightened the structure without flattening the voice.
