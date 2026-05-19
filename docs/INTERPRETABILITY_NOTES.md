# INTERPRETABILITY_NOTES.md — Methodology index for Leela-deliberation extractors

> **Thin index.** This file covers the cross-cutting concepts and the
> prose contract that govern every observational extractor in the
> project. Per-extractor methodology specifics live in module
> docstrings (see §5 for pointers). Decision recorded in Chunk 6a v2
> reviewer pass: methodology travels with code; this file stays slim.
>
> Status: **draft, Chunk 6a v2**. Sister documents:
> - `BIBLE.md` — strategic project vision
> - `PRIOR_WORK.md` — curated index of papers and repos
> - `DECISIONS.md` — architecture decision records
> - `docs/phase3/concept_candidates.md` — Phase 3 probe-concept tracker

---

## 1. Premise

Leela Chess Zero is a black box that, given a position, ultimately
commits to one move. The user-visible output is shallow: top-N policy
percentages, a value-head Q, an MCTS visit distribution, a chosen
bestmove.

A 2200+ player asking *"why this move?"* wants more than that. They
want the answer to **"how did Leela arrive at this move?"** — a
question about the engine's deliberation, not just its verdict.

The verdict is the outcome. The deliberation is everything that
produced it. The extractors in this project measure slices of that
deliberation. Phase 1 and Phase 2 Chunks 1-5 extracted only the
outcome. From Chunk 6 onward we extract slices of the deliberation
itself.

---

## 2. The two timescales

A given Lc0 analysis happens on two completely separate clocks:

**Network-time (one BT4 forward pass).** The residual stream passes
through ~15 transformer blocks; the final block's output goes through
the policy head to produce the reported move distribution. A given
candidate move has a rank/probability *at each layer* — by reading
intermediate residuals we can build a **per-layer rank trajectory** for
any candidate. See `leela_chat/extractors/layer_emergence.py` (created
in Chunk 6b) for the methodology.

**Search-time (one MCTS run).** PUCT walks the tree, expanding nodes
and backing values up. A given candidate move has a (visits, Q) pair
*at every point in the search* — by streaming Lc0's verbose-move-stats
during the search we capture a **per-visit-milestone trajectory** for
every candidate. See `leela_chat/extractors/mcts_trajectory.py` for the
methodology.

These are **independent signals**. The four interpretive cases:

| | network keeps | network drops |
|---|---|---|
| MCTS keeps | concordant: both promote — the engine's pick | divergent: MCTS rescue (search > policy alone) |
| MCTS drops | divergent: search refutes (policy > what search confirms) | concordant: both dismiss |

Each case tells a genuinely different story; the prompts and prose
must not blur them. The forbidden / allowed language lists in §3 are
how we enforce that.

---

## 3. Prose contract for observational extractors

This is a **methodological commitment**, not a limitation. It applies
to every extractor whose data is observational (everything in Chunks
6-7; Phase 3 probes; Phase 4 SAE features). Causal extractors —
Phase 6 counterfactual ablation — earn the right to use causal
language, but only about their own outputs.

### 3.1 Why the contract exists

Per-layer rank change and per-visit Q change are **correlations**. They
do not establish that a specific layer "computed" the change or that
search "decided" the move was bad. The same observed trajectory is
consistent with multiple interpretations:

1. *Causal:* the layer's computation actively re-evaluated the move.
2. *Architectural:* the network refines a representation; rank
   changes are a side-effect of geometric rotation with no semantic
   notion of consideration.

Without probes (Phase 3, to name *what feature* a layer detects) or
counterfactual ablations (Phase 6, to test whether removing a specific
component causes the rank change), we cannot distinguish (1) from (2).
The contract keeps the prose exactly as strong as the data warrants —
no stronger.

This is the **encoding-vs-use distinction** (ADR-0007, Lovering et al.,
"Predicting Inductive Biases of Pre-Trained Models") re-emerging one
layer down: that a feature is decodable from the residual doesn't mean
the network *uses* it; that a move's rank changes between layers
doesn't mean any layer *re-evaluated* it.

### 3.2 Forbidden language

Do not write — in handler prompts, in module docstrings, in
methodology docs, anywhere observational-extractor outputs are
described:

- *"the network noticed / realized / discovered / saw / found ..."*
- *"by layer N, the model recognized ..."*
- *"layers 8-10 worked out that ..."*
- *"the network changed its mind ..."*
- *"search decided / concluded / determined ..."*

### 3.3 Allowed language

Describe **what changed** — rank, probability, visit share, Q value —
without claiming what the network or search "did" semantically:

- *"the rank had shifted to N-th by layer K"*
- *"rank stayed at #1 across all measured layers"*
- *"Q dropped from +0.74 at 1 000 visits to +0.40 at 10 000 visits
  in the same tree"*
- *"consistent with the Kxb1 response Stockfish identifies as the
  refutation"* (Stockfish is the content authority; rank / Q
  trajectories covary but do not establish causation)
- *"the trajectory alone does not establish which layer, if any,
  computed the change"*

### 3.4 Permanence

This contract does **not** relax at later phases for observational
extractors. Phase 6 counterfactual ablations can use causal language
*about their own outputs* — "ablating head 9.3 restores Rxb1+ to
rank 1, evidence that the head computes the Kxb1 refutation" — but
the layer-emergence prose stays observational forever.

---

## 4. Worked example

Walkthrough position, Black to move:
`1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1`

Lc0 plays **Rxd4** (`d8d4`). Network policy ranks Rxb1+ first (~27%),
Rxd4 only fourth (~7%). MCTS at 10 000 visits inverts this: Rxd4
collects ~840 visits, Rxb1+ ~20.

### 4.1 Network-time story (predicted)

- **Rxb1+**: rank stays #1 across every measured BT4 layer. Its
  `network_dismissal` tag is **never_rose** — except it *started* at
  top-1, so technically the right tag is **rose_and_stayed**. The
  network alone, even at layer 15, ranks it first. The mechanism that
  catches Kxb1 lies beyond the per-position forward pass; search is
  what reveals it.
- **Rxd4**: enters top-10 around mid-network, stays there at the final
  layer (~4th). Tag: **rose_and_stayed**.

### 4.2 Search-time story (predicted)

- **Rxb1+**: early in the tree (~500 visits across all candidates), Q
  is roughly +0.7 — matches the optimistic policy prior. By 10 000
  visits in the same tree, Q has dropped to ~+0.4 as the refutation
  propagates and PUCT redirects visits elsewhere. Tag:
  **explored_and_refuted**.
- **Rxd4**: ~50 visits early because the prior is low; Q ~+0.85. By
  10 000 visits in the same tree, visits dominate (~840), Q has grown
  to ~+0.95. Tag: **explored_and_confirmed**.

### 4.3 What the LLM should say (Chunk 6c v1 prompt target)

> *"Rxb1+'s rank emerged only at the very end of the network — outside
> the top-10 through layer 13, then rising to rank 2 at layer 14 and
> rank 1 at layer 15. Within the same MCTS tree, Q started at +0.74
> (one early visit, V-head optimistic), reached a trough of +0.36
> around 22 visits as PUCT explored White's responses, then partially
> recovered to +0.53 by the deepest live-stats emit (~93 visits). The
> trough is the signature of search-time refutation; the rank
> trajectory alone doesn't tell us which layer, if any, computed the
> refutation — but Stockfish identifies Kxb1 as the move that holds.
> Rxd4 surfaced only at search-time: rank stayed >= 9 at every BT4
> layer (the network never finds it), but MCTS Q rose from a V-head
> estimate of +0.56 to a peak of +0.96 by 125 visits, with visit
> share growing to ~84% of the 10 000-node search."*

**Empirical (pod 2026-05-19, BT4 + Lc0 v0.32.1):**
- Rxb1+ summary: `q_first=+0.74 q_peak=+0.75@v2 q_trough=+0.36@v22 q_last=+0.52 blocks=25`
- Rxd4 summary: `q_first=+0.56 q_peak=+0.96@v125 q_trough=+0.42@v0 q_last=+0.92 blocks=25`
- Layer-emergence: Rxb1+ rank 41→...→1 (final), peak rank 1 at L=15. Rxd4 rank 43→...→9 (final), peak rank 9 at L=15.

The v0 prediction ("Q drops monotonically toward end-of-search")
turned out wrong on the magnitude: the dip-then-partial-recovery
pattern is the real refutation signature. The `q_trough` summary
field is the load-bearing data point, not `q_last`. Methodology
locked accordingly in Chunk 6a v2.

This is the v1 worked example. A second case — the divergent
*network-low / MCTS-rescue* quadrant — is gated on a hand-picked
fixture (deferred to pod session before Chunk 6c implementation).

---

## 5. Per-extractor methodology pointers

Methodology specifics live in module docstrings — read those for
implementation detail. This index points to where each lives.

| Extractor | Module | Methodology covers |
|---|---|---|
| Search-time Q-trajectory | `leela_chat/extractors/mcts_trajectory.py` | Single-search live-stats via `--log-live-stats`; block-grouping; per-candidate `MCTSTrajectorySummary`; content-aware down-sampling; inflection-point detection |
| Network-time layer emergence | `leela_chat/extractors/layer_emergence.py` (scaffolded in 6a v2; impl in 6b) | Zero-ablation logit-lens (Iterative Inference); per-layer rank trajectory; `CandidateLayerTrajectory`; why we don't use linear logit-lens |
| Position survey | `leela_chat/extractors/network_inference.py`, `leela_chat/extractors/stockfish.py`, `leela_chat/extractors/mcts.py` | Phase-1/2 scope; not deliberation extractors |

The prose contract in §3 applies to all observational outputs of these
extractors.

---

## 6. Cross-cutting limitations

### 6.1 BT4-specific

The methodology relies on BT4's specific architecture (15 transformer
blocks; attention-based policy head; smolgen attention). Different Lc0
networks would need re-validation. T82 in particular has a different
policy head; the zero-ablation hook attachment points would need
re-derivation. The BT4 network pin (ADR-0001) is load-bearing — see
DECISIONS.md.

### 6.2 No causal claim from observational extractors

The two timescales we measure (network-time, search-time) are both
observational. We read off correlates of deliberation, not interventions
on it. Causal claims wait for Phase 6 (counterfactual ablations) and
Phase 7 (reasoning pathways). See §3.4 for what this means for prose.

### 6.3 Methodology-vs-code drift risk

By keeping methodology specifics in module docstrings (per the C1
decision in Chunk 6a v2), we accept that the methodology travels with
the code it implements. The risk is that someone reads the code and
skips the docstring. Mitigations:

- Code review during chunk merges must include a docstring read.
- Phase Entry Briefs (CLAUDE.md §10) explicitly include reading the
  relevant per-extractor docstrings.
- This file's §5 lists every extractor's docstring location so it's
  hard to miss.

If the trade-off proves wrong by Phase 4, we revisit (per the C1
decision's exit criterion).

---

## 7. References

- **Karvonen et al. 2025**, "Iterative Inference in a Chess-Playing
  Neural Network" (arXiv 2508.21380). Source of the zero-ablation
  logit-lens methodology now in `layer_emergence.py`.
- **Alain & Bengio 2016**, "Understanding intermediate layers using
  linear classifier probes" (arXiv 1610.01644). Original logit-lens
  methodology in language modeling.
- **Lovering et al. 2021**, "Predicting Inductive Biases of
  Pre-Trained Models" (cited via ADR-0007). Source of the
  encoding-vs-use distinction underlying §3.
- **Jenner et al. 2024**, "Evidence of Learned Look-Ahead in a
  Chess-Playing Neural Network" (arXiv 2406.00877). Motivates why a
  high-rank policy move can still be search-refuted.
- **Lin et al. 2024**, "Tracing the Thought of a Grandmaster-level
  Chess-Playing Transformer" (arXiv 2604.10158). The project's North
  Star; Phase 7 reasoning-pathway extraction builds on this.
- **Monroe 2024**, "Mastering Chess with a Transformer Model"
  (arXiv 2409.12272). BT4 architecture, policy head, smolgen
  specifics.
- **McGrath et al. 2022**, "Acquisition of Chess Knowledge in
  AlphaZero" (arXiv 2111.09259). The concept-probe methodology for
  Phase 3.

Full PDFs under `update_research/pdfs/`; curated annotations in
`docs/PRIOR_WORK.md`.

---

## 8. Future work

- **Phase 3 — concept probes.** When a candidate's `network_dismissal`
  is `rose_and_fell`, name *what feature* the dismissing layer
  detects. Linear probes (Alain & Bengio methodology) trained on
  labeled positions. Concept candidates already accumulating in
  `docs/phase3/concept_candidates.md`.
- **Phase 4 — sparse features.** SAEs at each layer give a
  fine-grained feature dictionary. "Which sparse feature at layer 9
  has the highest activation when Rxb1+ drops?"
- **Phase 6 — counterfactuals.** Ablate suspected components and
  re-run the forward pass. If ablating head 9.3 brings Rxb1+ back to
  top-1, we have *causal* evidence that this head computes the
  refutation. This is the phase at which causal language about
  observational data unlocks (see §3.4 — and even then, causal claims
  are scoped to the ablation's own outputs).
- **Phase 7 — reasoning pathways.** The Fudan methodology — stitch
  per-position circuit attributions into a graph of "the network's
  thought process." Builds on the Chunk 6 layer-emergence data plus
  Phase 6 ablations.
- **Phase 4 — coach-voice prompt migration** (D2 from Chunk 6a v2
  reviewer pass). Rewrite handler prompts from telemetry-voice
  ("narrate the findings") to coach-voice ("give the player a
  coaching takeaway"), with voice-A/B harness for comparison.
  Scoped in PHASE_NOTES.md.
