# Smoke battery summary — 2026-05-24

Raw data: `/tmp/smoke_battery_20260524T095535Z.jsonl`

## Headline (three-rate classification)

**⚠️ LLM-success rate is 93.0%** (93/100 calls returned LLM content). 
The validation-pass numbers below reflect only the LLM-success calls and may be unreliable until LLM health is restored.

- **LLM-success rate:** 93.0% (93/100)
- **Validation-pass among LLM-success:** 100.0% (93/93) — unreliable; see warning above
- **Templated-fallback rate:** 7.0% (target ≤ 10%)

## Aggregate (legacy single-rate + latency)

- **Runs:** 100 (expected 100; missing = crashes / fixture issues)
- **Validation_ok rate (legacy single-rate):** 93.0% (target ≥ 90%)
- **Sources:** {'llm': 82, 'llm_retry': 11, 'templated': 7}
- **Latency** (handler_s + translate_s):
  - mean 14.38s
  - median 12.58s
  - p95 26.88s
  - p99 30.43s
- **Token consumption:** 754,247 in / 58,073 out

## illegal_move classification

- **Genuine hallucinations** (Rule 1 violations the validator caught): 1
- **Validator parser false-positives** (prose-context regex misreads): 5
- **Validator gaps** (span IS in findings, validator missed it): 1

| Position | Handler/mode | Span | In findings? | Classification | Failed LLM prose excerpt |
|---|---|---|---|---|---|
| puzzle_SGLK2 | did_you_consider/detailed | `Bxg7` | ✓ | validator_gap | ...ee (Rh1 −3.70, Be2 −3.72, Kg1 −4.19), and its Be2 line continues Be2 f4 Bxe5 f3 Bxg7+ — i.e. taking on e5 only works *after* Be2 f4 has been inserted, not immediate... |
| puzzle_WvZao | did_you_consider/detailed | `Bd2` | ✗ | parser_fp | ... evaluates so catastrophically is that it walks the queen onto the same file as Bd2 while still leaving the b8-knight untangled, but I'd need a PV from the engines... |
| puzzle_zzpwy | did_you_consider/detailed | `Qxh2+` | ✗ | parser_fp | ...cal refutation rather than a slow positional reassessment — almost certainly ...Qxh2+ winning the knight on h2, which is undefended after the a1-rook leaves. Stockfi... |
| puzzle_23Uqp | why_this_move/default | `Rf2` | ✗ | hallucination | ...age. The takeaway: Re1 walks the rook onto the e-file where Black's e3-pawn and Rf2 create concrete problems, while Rd1 keeps the rook active and out of the line o... |
| puzzle_cmQVa | did_you_consider/detailed | `Kxf3` | ✗ | parser_fp | ... straightforward — Qxf3+ trades Black's only piece for White's queen, and after Kxf3 the resulting K+P vs K endgame with White's h5- and g3-pawns is winning for Whi... |
| puzzle_Xpxds | why_this_move/detailed | `Qxg4` | ✗ | parser_fp | ...4 and the f3-knight defending d4, taking the d4-pawn directly loses material to Qxg4, which is what the −0.87 Q is reflecting. |
| puzzle_Xpxds | did_you_consider/detailed | `Bg4` | ✗ | parser_fp | ...oved the defender of d4. Played immediately, Nxd4 hangs the knight to Nxd4 with Bg4 still en prise to the queen — which is why both signals dismiss it. |

## Per-category breakdown

| Category | N | validation_ok | templated | median latency (s) |
|---|---|---|---|---|
| tactical_puzzle | 40 | 95% | 5% | 12.28 |
| quiet_middlegame | 25 | 96% | 4% | 12.57 |
| endgame | 20 | 90% | 10% | 13.49 |
| opening | 15 | 87% | 13% | 13.04 |

## Per-theme breakdown (tactical only)

| Theme | N runs | validation_ok | prose-mentions-theme |
|---|---|---|---|
| discoveredAttack | 5 | 100% | 0% |
| fork | 15 | 93% | 0% |
| mateIn3 | 15 | 93% | 0% |
| pin | 10 | 90% | 20% |
| sacrifice | 10 | 100% | 0% |
| skewer | 5 | 100% | 0% |

## Per-difficulty-band breakdown (tactical only)

| Band | N runs | validation_ok | Branch B rate |
|---|---|---|---|
| 1800-2200 | 15 | 93% | 60% |
| 2200-2600 | 10 | 90% | 60% |
| 2600-3000 | 10 | 100% | 60% |
| 3000-3400 | 5 | 100% | 60% |

Branch A (focal == MCTS pick): 0 of 60 H2+H3 runs
Branch B (focal ≠ MCTS pick):  60 of 60 H2+H3 runs

## Eyeball rubric (first output per category × handler × mode)

Per-cell rubric (Y/N):
1. **Position-character match** — judge manually per Chunk 6.8 plan.
2. **Grounding** — auto-Y for validation_ok=True; manually re-check the others.
3. **Forbidden-language check** — auto-fail if the auto-scan column shows hits.
4. **Over-claim check** — judge manually per the plan.

| Cell | Position | val_ok | Forbidden hits | Prose (first 400 chars) |
|---|---|---|---|---|
| endgame / did_you_consider / detailed | puzzle_23Uqp | True | (none) | Leela's MCTS pick is Rd1, not Re1 — Rd1 took 10031 of 10318 visits (97%) at Q +0.94, in the winning bucket. Re1 is the policy network's #4 candidate at 13.1% but got only 18 visits (under 1% of search) with Q −0.78. Network-time, Re1's rank stayed in the teens-and-twenties from L=0 through L=12, then climbed sharply: rank 4 at L=13, rank 1 at L=14 with prob 33.2%, settling back to rank 4 at L=15 —… |
| endgame / what_is_leela_seeing / default | puzzle_23Uqp | True | (none) | The three signals tell different stories at the root but converge on the move. Leela's policy network sees only a flat +12 cp with WDL heavily weighted to draw (84%), Stockfish at depth 15 gives White +2.60, and Leela's MCTS after 4,704 nodes lands at Q +0.91 — essentially a winning position from White's side. So the policy first impression badly under-rates this, and search corrects it.  The comm… |
| endgame / what_is_leela_seeing / detailed | puzzle_23Uqp | True | (none) | Leela's policy head reads the position as roughly balanced (+0.12, WDL heavily weighted to draw at 84%), but MCTS and Stockfish both see it as winning for White: MCTS Q is +0.94 from White's perspective and Stockfish gives +2.73 at depth 15 — a sharp split where one forward pass undersells the position and search rescues the evaluation. Leela's MCTS commits decisively to Rd1, the same move Stockfi… |
| endgame / why_this_move / default | puzzle_23Uqp | False | (none) | Asked about: Re1 (uci c1e1). This is NOT the engine's pick — see below. Engine's actual pick: Rd1. Position eval (policy net): +12 cp (+0.12); side to move: white; WDL 13% W / 84% D / 4% B. Focal move stats — MCTS: rank 3 by visits, N=14 (+0), Q=-0.72, prior 13.0%; Policy: rank 4 at 13.1%; Stockfish: outside its reported top moves..… |
| endgame / why_this_move / detailed | puzzle_23Uqp | True | (none) | Leela's MCTS strongly prefers Rd1, not Re1 — 9230 of 9505 visits (97%) at Q +0.94, squarely in the winning bucket. Network-time, Re1 had a quirky path through BT4: rank 24 at L=0, drifting in the high teens and twenties through the middle layers, then climbing to rank 8 by L=12, rank 4 by L=13, and briefly to rank 1 at L=14 (prob 33%) before settling back to rank 4 at the final layer with 13.1% po… |
| opening / did_you_consider / detailed | puzzle_2iiHL | True | (none) | Leela's pick is Nxc6, not Nxf7 — Nxc6 takes 9757 of 10006 visits (97%) at Q +0.51 in the better bucket, while Nxf7 stalls at 38 visits (under 1%) with Q −0.84, a losing evaluation. Network-time, Nxf7 was a high-prior candidate that the final layer ranks 2nd at 18.7%, but the per-layer trajectory is striking: rank 35 at L=0, drifting through the 20s and 30s, reaching rank 7 at L=13, then jumping to… |
| opening / what_is_leela_seeing / default | puzzle_2iiHL | True | (none) | All three signals point the same direction with Nxc6 as the clear pick, though they differ on magnitude: Leela's policy network gives +45 cp, MCTS settles at Q +0.48 (root, White's perspective), and Stockfish at depth 15 is more bullish at +1.14. Leela's MCTS committed 4408 of 4659 visits (95%) to Nxc6, matching both the policy first-impression top choice (45.6%) and Stockfish's pick. Policy and s… |
| opening / what_is_leela_seeing / detailed | puzzle_2iiHL | True | (none) | All three signals point the same direction, with Stockfish more bullish than Leela: the policy head gives +0.45 (eval_cp 45), MCTS settles at Q +0.52 from White's perspective after 6143 nodes, and Stockfish at depth 15 returns +1.15. The committed move is Nxc6, taking 5869 of 6143 visits (≈96%) — a near-unanimous search verdict.  Network-time, Nxc6 is a striking late-emergence case: it sat at rank… |
| opening / why_this_move / default | puzzle_2iiHL | True | (none) | Leela's MCTS strongly prefers Nxc6, not Nxf7: Nxc6 took 7205 of 7468 visits (96%) at Q +0.52, landing in the better bucket, while Nxf7 collected just 31 visits and crashed to Q −0.81 — that's a losing-bucket evaluation, a dramatic cross-bucket downgrade. The policy network was tempted (Nxf7 was the #2 first impression at 18.7%, and the prior was 23.4%), but as soon as search looked at the replies … |
| opening / why_this_move / detailed | puzzle_2iiHL | True | (none) | Leela's MCTS prefers Nxc6 (8456 of 8754 visits, 97%, Q +0.51 — in the better bucket), not Nxf7. Network-time, Nxf7 was a late-emerging policy candidate: rank 35 at L=0, drifting in the 20s–30s through the middle layers, climbing to rank 7 by L=13, and only entering the top-5 at the final layer L=15 at rank 2 with 18.7% policy — a sharp last-layer jump consistent with the policy head pattern-matchi… |
| quiet_middlegame / did_you_consider / detailed | puzzle_8Zks8 | True | `saw` | Leela's pick is Rcc8, not Qf6 — Rcc8 takes 8381 of 10106 visits (83%, Q +0.80, winning bucket for Black), while Qf6 is ranked 40th by visits and got exactly one visit. Both signals concordantly dismiss Qf6. Network-time, Qf6 never rose: its best showing was rank 21 at L=0, and it drifted between ranks 27 and 41 through every subsequent layer, ending at rank 38 by L=15 — BT4's pattern recognition d… |
| quiet_middlegame / what_is_leela_seeing / default | puzzle_8Zks8 | True | (none) | All three signals agree that Black is clearly better, though they don't quite agree on how much: Leela's policy network gives −2.21 with a 79% Black-win share, MCTS lands harder at Q +0.76 from Black's view (roughly −3 in White-perspective pawn terms), while Stockfish at depth 15 is more measured at −1.84. The committed move is Rcc8 — Leela's MCTS poured 4414 of 6210 visits (71%) into it at Q +0.7… |
| quiet_middlegame / what_is_leela_seeing / detailed | puzzle_8Zks8 | True | `monotonic` | All three signals agree Black is winning and converge on the same family of moves: Leela's policy gives −2.21 (WDL 78.5% Black wins), MCTS root Q is +0.77 from Black's perspective, and Stockfish at depth 15 reads −1.84 cp — a roughly 40 cp spread, with Leela's search the most optimistic. The committed move is Rcc8, taking 5927 of 7753 visits (76%) at Q +0.79, defending the rook while leaving the f… |
| quiet_middlegame / why_this_move / default | puzzle_8Zks8 | True | (none) | Leela's MCTS prefers Rcc8, not Qf6: Rcc8 took 7559 of 9413 visits (80%) with Q +0.80, firmly in the winning bucket from Black's perspective, while Qf6 collected just 1 visit at Q −0.96 — that's a catastrophic cross-bucket collapse from winning all the way to losing. Qf6 didn't even register in Leela's policy top-5 (prior just 0.51%), and it ranks 40th by visits in MCTS; the move essentially hangs … |
| quiet_middlegame / why_this_move / detailed | puzzle_8Zks8 | True | (none) | Leela's MCTS prefers Rcc8 (8330 of 10106 visits, 82%, Q +0.80 — winning for Black), not Qf6. Network-time, Qf6 never entered BT4's consideration at any depth: rank 21 at L=0 was its best showing, and from L=1 onward it drifted between ranks 27 and 41, finishing at rank 38 at L=15. The network's pattern recognition does not suggest this move at any layer of computation. Search-time, Qf6 received ex… |
| tactical_puzzle / did_you_consider / detailed | puzzle_2ATyh | True | (none) | Leela's pick is Ke6, not Kg5 — Ke6 takes 10195 of 10273 visits (99%) at Q +0.98, while Kg5 sits at rank 2 with just 5 visits, Q −0.98 (losing bucket from Black's perspective). Network-time, Kg5 was BT4's L=0–L=2 favorite at rank 1, then settled at rank 2 from L=3 onward; its probability mass eroded sharply in the late layers (0.45 at L=8, 0.23 at L=12, 0.012 at the final layer), consistent with th… |
| tactical_puzzle / what_is_leela_seeing / default | puzzle_2ATyh | True | (none) | Leela sees this as completely winning for Black, though it sees more than Stockfish does: Leela's policy puts the position at −45.12 with WDL 99% Black, MCTS confirms with root Q +0.98 from Black's view, while Stockfish reads it as only −4.78 at depth 15. All three sources unanimously pick Ke6 — Leela's policy gives it 98.8%, MCTS commits 4349 of 4516 visits (96%) at Q +0.98, and Stockfish ranks i… |
| tactical_puzzle / what_is_leela_seeing / detailed | puzzle_2ATyh | True | (none) | Black is winning decisively and all three signals agree on Ke6: Leela's policy gives it 98.8%, MCTS commits 5940 of 6108 visits (97%) with Q +0.98 from Black's perspective, and Stockfish concurs at −4.67 cp. The critical detail is that Kg5 walks into mate-in-3 (Rg7+ Kh5 Nf6+ Kh6 and Stockfish's line continues to mate), so Ke6 is essentially the only sane king move — the rest of the position is jus… |
| tactical_puzzle / why_this_move / default | puzzle_2ATyh | True | `saw` | Leela's MCTS overwhelmingly prefers Ke6, not Kg5: Ke6 took 7564 of 7732 visits with Q +0.98 (winning for Black), while Kg5 was visited just 3 times at Q −0.96 — a flip from winning to losing, the maximum possible bucket gap. Leela's policy network already saw this clearly, giving Ke6 98.8% to Kg5's 1.2%. Stockfish confirms the catastrophe: Kg5 walks into mate in 3, with the line Kg5 Rg7+ Kh5 Nf6+ … |
| tactical_puzzle / why_this_move / detailed | puzzle_2ATyh | True | (none) | Leela's MCTS prefers Ke6, not Kg5 — 9288 of 9457 visits (98%) at Q +0.98, in the winning bucket from Black's perspective. Network-time, Kg5 is the strange case: it was actually rank 1 at L=0 through L=2 (probability ~0.51–0.87), then settled to rank 2 from L=3 through L=15 as Ke6 took over, with the final-layer probability collapsing to 1.2% — the rank stayed in the top-2 throughout, but the netwo… |

## Outliers (validation_ok=False, source != templated)

None — every non-templated run validated cleanly.

## Backlog candidates (to be filed in PHASE_NOTES)

Pre-filled from automatic analysis; human review may add more:
- 1 genuine hallucination(s) — Rule 1 violation; halt-and-investigate per CLAUDE.md §8.
- 5 validator parser false-positive(s) — file as validator backlog item.
- 3 eyeball cells have forbidden-language hits — file as prompt v6/v3 iteration target.
