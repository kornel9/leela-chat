# INTERPRETABILITY_NOTES.md — Methodology for extracting Leela's deliberation

> Living methodology document. Reviewed before each new interpretability
> extractor lands. Status: **draft, Chunk 6a v1**. Owner: the project
> (this is the "why" behind the layer-emergence + Q-trajectory
> extractors). v1 revisions integrated reviewer feedback on multi-search
> stitching (now single-search live-stats), zero-ablation as the 6b
> default (not a fallback), and encoding-vs-use epistemic humility about
> layer-rank evolution.
>
> Sister documents:
> - `BIBLE.md` — strategic project vision
> - `PRIOR_WORK.md` — the curated index of papers and repos
> - `DECISIONS.md` — short architecture-decision records
> - This file — the *methodology* book: what we're extracting, how, and why
>   we chose this technique over the alternatives. Grows with each phase.

---

## 1. Premise

Leela Chess Zero is a black box that, given a position, ultimately commits
to one move. The user-visible output is shallow:

- a top-N policy list with percentages
- a value head Q in [-1, +1]
- an MCTS visit distribution
- a chosen bestmove

A 2200+ player asking *"why this move?"* wants more than that. They want
the answer to **"how did Leela arrive at this move?"** — which is a
question about the engine's deliberation, not just its verdict.

The verdict (final policy, final visits, final Q) is the *outcome*. The
deliberation is everything that produced it. This document is about
extracting the deliberation.

Phase 1 + Phase 2 (Chunks 1-5) extracted only the outcome. From Chunk 6
onward we extract slices of the deliberation itself.

---

## 2. Two timescales of deliberation

A given Lc0 analysis happens on two completely separate clocks.

### 2.1 Network-time (forward-pass, fast)

In one forward pass through BT4, the residual stream passes through
~15 transformer blocks. The final layer's output is run through the
policy head to produce the 1858-dim move distribution that Lc0 reports
as "policy."

But *every* intermediate layer also has a residual stream, and that
residual stream is the same dimensionality as the final one. We can
apply the (frozen) policy head to any intermediate layer's residual and
read off **what the policy would be if the network stopped thinking
right there**.

This is **logit-lens / iterative-inference methodology** — applied to
chess in Karvonen et al. ("Iterative Inference in a Chess-Playing
Neural Network," arXiv 2508.21380), and originally in Alain & Bengio
(arXiv 1610.01644) for language models.

The findings of the chess paper are striking and load-bearing for us:

- For most positions, the network's eventual top-1 move enters the top-5
  by layer ~4 and stays. The middle layers are mostly *refining*, not
  changing the answer.
- For positions involving short-horizon tactics, a *candidate move can
  rise and fall* across layers. The early layers see a flashy candidate
  (a check, a sacrifice); a middle layer "notices" the refutation and
  the move's rank drops; the final layer commits elsewhere.
- Conversely: positional moves that depend on subtle features sometimes
  *emerge late*, ranked outside the top-10 until layer 8 or 10, then
  rising as later layers integrate global features.

So the *layer-wise emergence trajectory of a candidate move* is a
window into the network's silent deliberation: did the network briefly
consider this move and then dismiss it, or did it never consider it at
all? We call this **network-time emergence**.

### 2.2 Search-time (MCTS, slow)

After the network finishes thinking, MCTS uses the policy and value
heads as inputs to a tree search. Inside one search call the engine
expands ~10,000 nodes (at our default budget), and on each node it:

1. Picks a leaf to expand by walking the tree using PUCT (which trades
   off Q vs. policy prior vs. visit count).
2. Runs the network on that leaf.
3. Backs the value up the tree.

This means each candidate move's Q evolves over the search:

- **Early in the search** (first ~100-500 nodes), PUCT explores moves
  roughly in policy-prior order. A move with a high policy prior gets
  visits even if it's actually bad.
- **As the search deepens**, those early visits propagate value back up
  the tree. If a move's "best response" is, in fact, a strong refutation,
  the move's Q crashes from the optimistic policy-prior anchor toward
  the refutation's true value.
- **Late in the search** (5k+ nodes), the visit distribution should be
  dominated by moves whose Q has *survived* deeper exploration.

So the *Q-trajectory of a candidate move across search budgets* is a
window into search-time refutation: did MCTS try this move and find a
refutation, or did it simply never have the budget to look? We call
this **search-time refutation** (or **Q-trajectory**).

### 2.3 The two timescales are independent signals

A move can be:

- **Dismissed at both timescales.** Network's later layers downgrade it
  AND MCTS visits show its Q crashing. *Strongest dismissal evidence.*
- **Dismissed by the network only.** Early policy puts it on the board,
  later layers drop it, but MCTS never even gave it enough visits to
  develop a Q. *The network's later layers found the refutation; search
  didn't need to.*
- **Dismissed by search only.** Network's policy ranks it high
  throughout all layers — but during MCTS, the moves it leads to get
  refuted and the move's Q drops. *The network missed something search
  found.*
- **Confirmed at both timescales.** Network's later layers reinforce it
  AND visits accumulate as Q stays healthy. *The eventual bestmove,
  usually.*

These four cases tell genuinely different stories, and our prose should
reflect that — not collapse them into "Leela likes / doesn't like" this
move.

---

## 3. Methodology — network-time

### 3.1 Goal

For a board position, run BT4 once and produce a per-layer per-move
view: at layer L, what is the policy distribution implied by *stopping
the forward pass at L and applying the (final) policy head*?

### 3.2 Strategy: zero-ablation logit-lens (default)

For each layer index L ∈ [0..N−1], run a forward pass with a hook at
`blocks.{L}.hook_resid_post` (or equivalent) that **replaces the layer's
output with `torch.zeros_like(activation)`**, then read the model's
final policy. Layer N's "ablation" is the baseline pass (no hooks). The
per-layer-L policy is "what BT4 would predict if it only had layers
0..L−1; everything beyond was zeroed."

This is the methodology of the Iterative Inference paper (arXiv
2508.21380) and we adopt it as **6b's default**, not a fallback.

- Cost: ~(N+1)× a single forward pass. For BT4 with N=15, that's ~16
  passes × ~50 ms on a 4090 ≈ 800 ms. Well within Phase 1's 30 s
  end-to-end ceiling. Cost is not the binding constraint.
- The vendored `HookedRootModule.run_with_hooks`
  (`leela_chat/_vendor/transformer_lens/hook_points.py:417`) supports
  this directly: a hook can return a tensor that replaces the activation
  downstream. No modifications to the vendored module needed.
- We let the *real* `PolicyHead` see the (modified) final residual.
  We do **not** invoke the head on intermediate residuals. This sidesteps
  the geometric-mismatch failure mode below.

### 3.3 Why we don't start with linear logit-lens

A naive alternative — call **linear logit-lens** — caches each
intermediate residual and applies the trained `PolicyHead` *directly*
to those caches. It's faster (~1.5× a single forward pass) but
unsound in this setting.

BT4's `PolicyHead` (`leela_chat/_vendor/transformer_lens/components/leela_policyhead.py`)
is an **attention-based head** with learned key/query projections per
square, plus separate promotion logits. Its parameters are tuned to
the geometry of the *final* residual stream. Applied to intermediate
residuals, the head can produce *plausible-looking but mechanistically
meaningless* output: sane-shaped distributions, sensible-looking top-N
orderings, all artefacts of geometric mismatch rather than of the
layer's computation.

The seductive failure mode here is that a "final-layer top-1 matches
the engine's policy" sanity check **does not** distinguish "early-
layer projection is real" from "early-layer projection is coincidentally
plausible noise." Subtle rank-order errors at intermediate layers
would slip through, and our prose would describe a layer's "thinking"
based on noise. Per CLAUDE.md §2 Rule 1, that is the failure mode we
exist to prevent.

Linear logit-lens stays out of the codebase until equivalence with
zero-ablation is empirically demonstrated. That's a future
optimization (deferred to Chunk 6.5 or later), gated on a positive
side-by-side study: concordant top-3 ranks at every layer for ≥5
diverse positions, including positions where the eventual top-1 differs
from the layer-1 top-1.

### 3.4 What zero-ablation produces — and what it doesn't

Under zero-ablation we get, for each layer L, a *real* BT4 policy
distribution — the model's actual `PolicyHead` output, run on a
modified residual stream. That distribution is interpretable in the
network's native units (probabilities over the 1858 move vocabulary).
What changes between L=0 and L=N is *only how much of the network's
computation was used*; the head's contribution is constant.

This eliminates the geometric-mismatch failure mode of §3.3. The
remaining caveat — that we cannot causally claim a specific layer
"computed" the rank change — is the topic of §6.5.

### 3.5 Data we extract per candidate per layer

```python
LayerPolicyPoint(layer: int, rank: int, prob: float)
```

For each candidate of interest, we record one of these per layer. The
candidates we care about for `did_you_consider`:

- The focal move the user asked about (always, even if rank is high).
- The final top-N policy moves (so we have coverage of the eventual
  winners).
- The mid-search top-N MCTS moves (so we have coverage of the search
  story).

### 3.6 Summary tags

The raw trajectory is rich but verbose. To make the LLM's job easier we
pre-compute a categorical tag per candidate:

```
network_dismissal ∈ {rose_and_stayed, rose_and_fell, never_rose}
```

- **rose_and_stayed**: entered top-5 by layer L_in ≤ 8 and remained in
  top-10 through the final layer.
- **rose_and_fell**: entered top-5 at some L_in and *exited* top-10 by
  the final layer. This is the interesting "considered then dismissed"
  case.
- **never_rose**: never entered top-10 at any layer. The network simply
  didn't see this move as a candidate.

These tags are derived from the trajectory; the prompt receives both,
so the LLM can quote the tag *and* cite the specific layer numbers
("by layer 6 it was top-1, by layer 12 it had dropped to 4th").

---

## 4. Methodology — search-time

### 4.1 Goal

For a board position, run Lc0 MCTS **once** to a deep budget and
record, for every candidate of interest, how its (visits, Q) evolved
within that **same continuous tree** as the search progressed. The
trajectory is a true intra-search history, not a stitched composite of
independent searches.

### 4.2 Why not stitch independent searches?

A first-cut implementation runs Lc0 K times at budgets [500, 1000,
2500, 5000, 10000] and stitches per-candidate (visits, Q) across the K
independent search results. We tried this in Chunk 6a v0 and rejected
it after reviewer feedback:

- **The K trees are not the same tree.** MCTS expansion order depends
  on backend stochasticity, exploration-bonus tiebreaks, and network
  noise. Two 10 000-node runs can produce different visit
  distributions; *different-budget* runs produce *different-shape*
  trees that happened to have ended at different sizes.
- **Prose built on stitched data over-claims.** A sentence like *"Q
  dropped from +0.74 at 1000 nodes to +0.53 at 10 000"* implies
  single-run continuity that the data doesn't have. With stitching,
  the true statement is *"Q in a 1 000-node search vs Q in a separately-
  run 10 000-node search"* — a different (and much weaker) claim.

### 4.3 Implementation — Lc0 `--log-live-stats`

Lc0 v0.32.1 supports the `--log-live-stats` flag (UCI option
`LogLiveStats`). Per `update_research/repos/lc0-master/src/search/classic/params.cc:294`:

> *"Do VerboseMoveStats on every info update."*

When the option is on, Lc0 emits the **full verbose-move-stats block**
— every candidate's line plus the `node` summary — on every UCI info
update during the search. The format is identical to end-of-search
verbose-move-stats; the existing `parse_verbose_move_stats` reuses
cleanly.

```python
# One call. Live-stats stream collected via the wrapper's existing
# info-string capture in leela_chat/engine/lc0_engine.py.
result = engine.analyse_verbose(board, nodes=10_000, log_live_stats=True)

# result.info_string_lines now contains many blocks of candidate lines,
# each terminated by a `node ...` summary line.
blocks = group_live_stats_blocks(result.info_string_lines)
findings_per_block = [parse_verbose_move_stats(b, ...) for b in blocks]

# Pick the snapshots closest to our requested milestones.
selected = select_milestone_snapshots(
    findings_per_block,
    milestones=[500, 1000, 2500, 5000, 10000],
)

# Join across snapshots by UCI; each MCTSChild.trajectory is one point
# per selected snapshot, all from the SAME tree.
```

Block boundary detection is mechanical: a line whose header parses to
the literal token `node` ends the current block. The next non-`node`
candidate line opens a new block. Stray diagnostic info-string lines
are filtered out by the parser.

### 4.4 Milestone selection

The shipped API accepts a list of target visit milestones (default
`[500, 1000, 2500, 5000, 10000]`). For each target M, the extractor
picks the live-stats snapshot whose `total_nodes_searched` is **closest
to M**. If Lc0's emission cadence is coarser than the requested
milestones (e.g. Lc0 emits at 800, 2000, 4500, ...), multiple
milestones may resolve to the same snapshot; the trajectory then has
fewer *unique* points than milestones requested. That's an honest
representation of the data.

A snapshot's *actual* `nodes_searched` (not the requested milestone
value) is recorded on each `MCTSTrajectoryPoint`. Prose downstream
uses the actual values, never the requested ones.

### 4.5 Cost

One Lc0 search to the deepest milestone, plus per-snapshot parse of
the streamed info-string lines (cheap, ~µs per block). On the pod
GPU, ~3-4 s warm for a 10 000-node search. *Faster* than the prior
multi-search approach.

### 4.6 What this gives us that multi-search did not

- **Single-tree continuity.** Every snapshot is the same tree at a
  different point in its growth. "Q at this candidate at 1 000 visits"
  is the same tree as "Q at this candidate at 10 000 visits" — just
  later.
- **Honest prose.** Sentences like *"Q dropped from +0.74 to +0.53
  within the same search"* are now true. The reader's natural
  interpretation matches the data.
- **Cheaper.** One search instead of K, but with richer information.

### 4.7 Summary tags

Mirroring §3.5 for search-time:

```
search_dismissal ∈ {explored_and_refuted, explored_and_confirmed,
                    never_explored, unknown}
```

- **explored_and_refuted**: visits accumulated in early budgets, Q
  dropped by >0.2 across the trajectory.
- **explored_and_confirmed**: visits grew across budgets, Q stayed
  within 0.1 (or rose).
- **never_explored**: visits ≤ 2 at the final budget. (Search budget
  simply didn't reach this move.)
- **unknown**: trajectory data missing or insufficient (fewer than 3
  checkpoints with this candidate present).

---

## 5. Worked example — the walkthrough position

> Black to move. FEN: `1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1`

In this position Lc0 plays **Rxd4**. The policy network ranks Rxb1+
first (~27%) and Rxd4 only fourth (~7%) — but MCTS visits invert this
hierarchy: Rxd4 collects ~840 visits, Rxb1+ only ~20.

This is *the* canonical test case for our methodology because it's a
clean policy-vs-search disagreement.

### 5.1 Network-time story (predicted)

We *expect* (will validate in 6b integration test). Phrasings below
describe what changed; they do not claim that any specific layer
"computed" the change — see §6.5 for why that distinction matters.

- **Rxb1+**: ranked top-1 at every layer we measure (matching its
  high final policy). Its `network_dismissal` tag is **never** — the
  network alone, even at layer 15, ranks it #1. The mechanism that
  catches Kxb1 lies beyond the network's per-position forward pass;
  search is what reveals it.
- **Rxd4**: enters the top-10 around mid-network and stays in top-10
  at the final layer (~4th). Its `network_dismissal` is
  **rose_and_stayed**.

If layer-emergence shows Rxb1+ ranked top-1 throughout all 15 layers,
that *itself* is informative: it tells us the per-position forward
pass alone is insufficient for this case, and search is doing
load-bearing work the network couldn't.

### 5.2 Search-time story (predicted)

- **Rxb1+**: early in the tree (~500 visits) likely shows healthy Q
  (~+0.7) matching the optimistic policy prior; by 10 000 visits in
  the same tree, Q drops to ~+0.4 as the refutation propagates and
  PUCT redirects visits elsewhere. Its `search_dismissal` is
  **explored_and_refuted**.
- **Rxd4**: ~50 visits early because the prior is low. Q is healthy
  (~+0.85), so PUCT progressively commits to it. By 10 000 visits in
  the same tree, visits dominate (~840) and Q has grown to ~+0.95.
  Its `search_dismissal` is **explored_and_confirmed**.

### 5.3 What the LLM should say (Chunk 6c v1 prompt target)

> *"Rxb1+ never dropped in the network — its rank stayed top-1 across
> all 15 layers. The refutation came from search within the same tree:
> at ~1 000 visits, Q was +0.7; by ~10 000 visits in that same tree, Q
> had fallen to +0.4 while visits stagnated at ~20. The rank trajectory
> alone doesn't tell us *which* layer or head, if any, computed the
> refutation — but Stockfish identifies Kxb1 as the move that holds.
> Rxd4 surfaced at both timescales: rank entered the top-10 by the
> middle layers and stayed; its MCTS Q grew from +0.85 at ~1 000 visits
> to +0.95 at ~10 000 visits in the same tree, with visits accumulating
> as PUCT committed."*

Note the deliberate phrasings:
- *"never dropped in the network"* not *"the network never noticed"*.
- *"the refutation came from search"* (compatible with both encoding
  and use interpretations of the rank data — see §6.5).
- *"rank entered the top-10 ... and stayed"* not *"the middle layers
  pulled it in"*.
- Stockfish is the source of the *content* claim (which move refutes);
  layer-rank data is the source of the *trajectory* claim (rank
  changed).

---

## 6. Limitations and honest caveats

### 6.1 Linear logit-lens is approximate

The policy head was trained against the *final* residual. There is no
guarantee that the same head, applied to layer 4's residual, produces a
meaningful distribution. We mitigate this by:

- Sanity-checking the final-layer match against `extract_network_inference`.
- Comparing "easy" positions (starting position, simple endgames) across
  layers — the top-1 should be stable, since the answer doesn't depend
  on depth.
- Falling back to zero-ablation logit-lens if linear is degenerate.

We do *not* claim the early-layer distributions are calibrated
probabilities. We use them as **rank signals** and explicitly frame
prose around rank, not probability.

### 6.2 The five MCTS budgets are coarse

Real search-time dynamics happen across hundreds of visits per
checkpoint. Our 5-point trajectory captures monotonic trends well but
won't catch fine oscillations. This is fine for the project's goals
(narrative prose, not visualization), but a researcher reproducing our
work should be aware.

### 6.3 No causal claim

Both timescales are observational. We are reading off correlates of
deliberation, not intervening on them. Phase 6 (counterfactuals) and
Phase 7 (reasoning pathways) will add causal extensions: ablate layer
L's contribution and re-run, see if Rxb1+ still gets visits. But Chunk
6 is observational only.

### 6.4 BT4-specific

This methodology relies on BT4's specific architecture (15 layers,
attention-based policy head, smolgen attention). Different Lc0
networks would need re-validation. T82 in particular has a different
policy head and the zero-ablation hook attachment points would need
to be re-derived.

### 6.5 What layer-emergence does and does not tell us

We measure **per-layer policy-rank trajectories** for candidate moves.
We sometimes describe this in prose using language like "the move was
considered then dismissed" because that phrasing is the most readable
chess-content gloss on the data. But the data does not establish a
semantic act of consideration on any particular layer's part.

Per-layer rank change is consistent with at least two interpretations:

1. **Causal — the layer's computation actively re-evaluates the move.**
   Some attention head at layer L attends to the squares that matter
   for the refutation; its output rotates the residual so that the
   move's logit drops.

2. **Architectural — the network refines a representation; rank
   changes are a side-effect of that refinement with no semantic
   notion of consideration.** The residual progresses through layers
   doing many distributed computations; the projection of the residual
   onto move-logit space happens to *rotate* between layers, and a
   given move's rank can rise and fall purely because of how the
   geometry shifts, with no layer's contribution being "about" that
   move specifically.

The methodology in this document cannot distinguish (1) from (2). It
takes probes (Phase 3, to name *what feature* a layer detects) and
counterfactual ablations (Phase 6, to test whether removing a specific
component causes the rank change) to push from descriptive
("rank changed at layer L") to causal ("layer L computed the
refutation").

This is the **encoding-vs-use distinction** (ADR-0007, Lovering et al.,
"Predicting Inductive Biases of Pre-Trained Models") re-emerging one
layer down. ADR-0007 applies it to probes: that a feature is decodable
from the residual doesn't mean the network *uses* it. The same logic
applies here: that a move's rank changes between layers doesn't mean
any layer *re-evaluated* it.

**Practical consequence for prose.** In handler prompts and methodology
text, describe **what changed** (rank, probability, visit share)
without invoking causal verbs about the network's "thinking." Forbidden
language for `did_you_consider` v1 prompt and beyond:

- *"the network noticed / realized / discovered / saw / found ..."*
- *"by layer N, the model recognized ..."*
- *"layers 8-10 worked out that ..."*
- *"the network changed its mind ..."*

Allowed language:

- *"the rank had shifted to N-th by layer K"*
- *"rank stayed at #1 across all measured layers"*
- *"consistent with the Kxb1 response Stockfish identifies as the
  refutation"* (Stockfish is the content authority; the rank
  trajectory just covaries)
- *"the trajectory alone does not establish which layer, if any,
  computed the change"*

The point is not to make the prose stilted; it's to keep the prose
exactly as strong as the data warrants — no stronger.

---

## 7. References

The methodology in this document draws on:

- **Karvonen et al. 2025**, "Iterative Inference in a Chess-Playing
  Neural Network" (arXiv 2508.21380). Source of the layer-wise emergence
  framing; the zero-ablation logit-lens fallback comes from their
  Algorithm 1.
- **Alain & Bengio 2016**, "Understanding intermediate layers using
  linear classifier probes" (arXiv 1610.01644). Original logit-lens
  methodology in language modeling.
- **Jenner et al. 2024**, "Evidence of Learned Look-Ahead in a
  Chess-Playing Neural Network" (arXiv 2406.00877). Demonstrates that
  Lc0 can pre-compute multi-ply responses inside one forward pass —
  motivates why a high-rank policy move can still be search-refuted
  (the network's look-ahead can miss specific defenses).
- **Lin et al. 2024**, "Tracing the Thought of a Grandmaster-level
  Chess-Playing Transformer" (arXiv 2604.10158). The project's North
  Star; reasoning-pathway extraction in Phase 7 will build on this.
- **Monroe 2024**, "Mastering Chess with a Transformer Model"
  (arXiv 2409.12272). The BT4 architecture paper; policy head + smolgen
  specifics live here.
- **McGrath et al. 2022**, "Acquisition of Chess Knowledge in
  AlphaZero" (arXiv 2111.09259). The concept-probe methodology that
  Phase 3 will build on; complementary to the layer-emergence work
  described here.

Full PDFs of these are under `update_research/pdfs/`; the curated index
in `docs/PRIOR_WORK.md` annotates each.

---

## 8. Future work

This document is the seed of the project's interpretability methodology.
Subsequent phases will extend it:

- **Phase 3 — concept probes.** When a candidate's `network_dismissal`
  is `rose_and_fell`, *what* feature does the dismissing layer detect?
  Linear probes (Alain & Bengio methodology) trained on labeled
  positions will let us name the feature: "layer 9 detects 'the back
  rank is defended' and that's what kills Rxb1+ in the layer-9 logits."
- **Phase 4 — sparse features.** SAEs at each layer give us a fine-grained
  feature dictionary. The "what fires when Rxb1+ drops" question
  becomes "which sparse feature at layer 9 has the highest activation
  on this position?"
- **Phase 6 — counterfactuals.** Ablate the candidate features /
  attention heads / layers and re-run the forward pass. If ablating
  layer 9 head 3 brings Rxb1+ back to top-1, we have *causal* evidence
  that this head computes the refutation.
- **Phase 7 — reasoning pathways.** The full Fudan methodology — stitch
  per-position circuit attributions into a graph of "the network's
  thought process." Builds directly on the Chunk 6 layer-emergence data.

The framing established in this document — *two independent timescales
of deliberation, both observable, both narratable* — should survive
through these later phases. The extractors grow richer; the storytelling
grammar stays the same.
