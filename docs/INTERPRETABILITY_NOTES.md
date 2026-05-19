# INTERPRETABILITY_NOTES.md — Methodology for extracting Leela's deliberation

> Living methodology document. Reviewed before each new interpretability
> extractor lands. Status: **draft, Chunk 6a**. Owner: the project (this is
> the "why" behind the layer-emergence + Q-trajectory extractors).
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

### 3.2 Two implementation strategies

**Strategy A — Linear logit-lens.** Single forward pass with hooks at
every `hook_resid_post.{L}`. For each captured residual stream tensor,
invoke the *same* policy head (frozen, learned on the final layer)
directly on the cached residual. Read off the 1858-dim distribution.

- Cost: ~1.5× a single forward pass.
- Caveat: the policy head was trained against the *final* residual.
  Early-layer residuals may be in a different geometric subspace; the
  resulting distribution can be degenerate.
- Soundness check: the final layer's logit-lens output must exactly
  match the engine's reported policy (otherwise we wired the hooks
  wrong). If layer 0's top-1 is nonsense (e.g. random / illegal) then
  the linear logit-lens is degenerate at early layers and we need
  Strategy B.

**Strategy B — Zero-ablation logit-lens.** For each layer L, run a
forward pass with layers L..N's transformer blocks masked out (set
contributions to zero, or replaced with mean activations). The final
policy from that pass is then "what BT4 would say if it only had
layers 0..L−1."

This is the methodology of the Iterative Inference paper. It's more
expensive (~N× forward passes), but it's the one the paper validates.

**Our plan.** Start with Strategy A. It's fast, and the final-layer
sanity check is cheap. If the layer-0 / layer-1 outputs are degenerate
on the BT4 starting position (where the right answer is well-defined and
shouldn't depend much on depth), pivot to Strategy B.

### 3.3 The BT4-specific wrinkle: policy head reapplication

BT4's policy head is not a linear projection of the final residual. From
`leela_chat/_vendor/transformer_lens/components/leela_policyhead.py`:

- Linear "key" and "query" projections per square (8×8 = 64 tokens).
- 64×64 logit grid produced by Q·Kᵀ.
- Promotion logits added separately (for 8th-rank pawn moves).
- 1858 final move logits assembled from the grid + promotion adjustments.

This means we can't shortcut the head with `final_W_unembed @ residual`.
We must invoke the PolicyHead module (or its functional equivalent) on
the cached residual stream. Concretely, for each layer L:

```
residual_L = cache["blocks.{L}.hook_resid_post"]   # (batch=1, 64, d_model)
policy_logits_L = model.policy_head(residual_L)    # (batch=1, 1858)
policy_probs_L = softmax(policy_logits_L, dim=-1)  # (batch=1, 1858)
```

The vendored `PolicyHead` is just a `nn.Module`; we can call it
directly on intermediate residuals. (HookedTransformer's `decompose_qk`
infrastructure isn't required.)

### 3.4 Data we extract per candidate per layer

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

### 3.5 Summary tags

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

For a board position, run Lc0 MCTS at multiple budgets (e.g. 500, 1000,
2500, 5000, 10000 nodes) and record, for every candidate, its (visits,
Q) at each checkpoint.

### 4.2 Implementation

Lc0's UCI `go nodes N` runs a search and reports verbose-move-stats at
the end. We already parse this in `extractors/mcts.py` and
`extractors/mcts_parser.py`. Multi-budget trajectory is the same
extractor called K times with different `nodes`:

```python
budgets = [500, 1000, 2500, 5000, 10000]
checkpoints = [extract_mcts(board, engine=engine, nodes=N) for N in budgets]
```

Then for each candidate UCI present in the final-budget findings, we
join across checkpoints by UCI:

```python
trajectory[uci] = [
    MCTSTrajectoryPoint(
        nodes_searched=ck.total_nodes_searched,
        visits=child.visits if child else 0,
        q=child.q if child else 0.0,
    )
    for ck, child in zip(checkpoints, lookups)
]
```

The result is attached as `MCTSChild.trajectory: list[MCTSTrajectoryPoint]`
on the final-budget findings. Existing callers see the same shape as
before (the field defaults to `[]`); new callers can introspect.

### 4.3 Why checkpoint resolution, not visit-by-visit?

A true visit-by-visit trajectory would require Lc0 to emit progress
during search, which is non-trivial (would need either `--show-tree`
periodically or a custom build). Checkpoints at logarithmically-spaced
budgets capture the same qualitative story:

- Visit growth rate is informative (visits doubling between budgets =
  PUCT still favoring this move; visits stagnating = PUCT has moved on).
- Q evolution is informative (Q dropping monotonically across budgets =
  search progressively finding the refutation; Q rising = search
  confirming early optimism).

Five checkpoints over [500, 10000] is enough resolution for the prose
to make a meaningful statement. We can scale up if needed.

### 4.4 Cost

Each `extract_mcts(board, nodes=N)` costs roughly N microseconds-per-node
on the pod GPU. Running 5 budgets sums to ~19k nodes total — *slower
than just running 10k once*, but only modestly (the first 4 are warm
network passes and short).

In practice on the pod: warm BT4 + 5 budgets [500..10000] ≈ 6-8 s.
Versus a single 10k MCTS ≈ 2-3 s. So Q-trajectory roughly doubles MCTS
latency — acceptable for `did_you_consider`, gated behind a `--detailed`
flag for the retrofitted Handler 1 and Handler 2.

### 4.5 Summary tags

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

We *expect* (will validate in 6b integration test):

- **Rxb1+**: enters the top-1 by some early layer (matches its high
  final policy), then *stays high* at the final layer (because policy
  has no concept of search-deeper). Its `network_dismissal` is
  **never** — the network alone, even at layer 15, ranks it #1. The
  network *cannot* find this refutation. Search must.
- **Rxd4**: enters the top-10 around mid-network as positional features
  integrate. Stays in top-10 at the final layer (ranked ~4th). Its
  `network_dismissal` is **rose_and_stayed**.

If our methodology shows Rxb1+ ranked top-1 throughout all 15 layers,
that *itself* is the story: "The network never saw Kxb1's refutation;
this is why MCTS is essential here."

### 5.2 Search-time story (predicted)

- **Rxb1+**: early budget (500 nodes) likely shows 30+ visits and Q
  around +0.7 (matching the optimistic policy prior). By budget 10000,
  visits stagnate near 20 and Q crashes toward +0.4. Its
  `search_dismissal` is **explored_and_refuted**.
- **Rxd4**: early budget might show only ~50 visits because the prior
  is low. But Q is healthy (+0.85+), so PUCT progressively redirects
  visits to it. By budget 10000, visits dominate (~840) and Q has risen
  toward +0.95. Its `search_dismissal` is **explored_and_confirmed**.

### 5.3 What the LLM should say (Chunk 6c v1 prompt target)

> *"Rxb1+ wasn't dismissed by the network — it stayed top-1 through all
> 15 layers. The refutation came from search: MCTS gave it 30 visits at
> 1000 nodes when Q looked like +0.7, but by 10000 nodes Q had crashed
> to +0.4 and visits had stagnated at 20. The network missed Kxb1's
> defense; search caught it. Rxd4, conversely, was a slow-burn from both
> sides: it entered the top-10 in the network's middle layers and never
> left, and its MCTS Q rose from +0.85 at 1000 nodes to +0.95 at 10000
> as visits accumulated."*

This is what we are unable to write today (Phase 2 Chunk 5 has no
trajectory data); it is what we can write after Chunk 6 lands.

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
policy head and the linear logit-lens math may not transfer.

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
