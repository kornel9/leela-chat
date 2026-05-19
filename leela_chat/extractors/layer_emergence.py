"""Network-time layer emergence via zero-ablation logit-lens (scaffold).

Source of truth for the **network-time methodology** in the dual-
timescale dismissal framework. Cross-cutting framing in
``docs/INTERPRETABILITY_NOTES.md``; prose contract in §3 of that file
applies to every consumer of this module's output.

**Status: scaffold only.** Chunk 6a v2 lands the methodology docstring
plus a placeholder function signature. Full implementation lands in
Chunk 6b. The docstring is the *written contract* the 6b implementation
must satisfy.

Companion: :mod:`leela_chat.extractors.mcts_trajectory` is the
*search-time* half of the dual-timescale methodology.

================================================================
Methodology — zero-ablation logit-lens
================================================================

**Goal.** For each layer L ∈ [0..N-1] of BT4, measure what the network
would predict if its computation **stopped after layer L−1**. The
sequence of per-layer policy distributions, read in order, is the
network's "thought trajectory" for the position. We track per-candidate
rank across this trajectory.

**Method.** For each layer index L, run a forward pass with a hook at
``blocks.{L}.hook_resid_post`` (or equivalent) whose return value is
``torch.zeros_like(activation)``. Downstream blocks then see a zeroed
contribution from layer L. The model's own ``PolicyHead`` reads the
final (now-modified) residual stream. The output is the policy
distribution under the ablation. Layer N's "ablation" is the baseline
pass with no hooks.

The vendored ``HookedRootModule.run_with_hooks`` at
``leela_chat/_vendor/transformer_lens/hook_points.py:417`` supports
this directly — a hook can return a tensor that replaces the
activation. No modifications to the vendored module needed.

Cost: ~(N+1)× one forward pass. For BT4 (N=15) on a 4090, ~16 passes ×
~50 ms ≈ 800 ms. Well inside the Phase 1 30 s end-to-end ceiling. Cost
is not a binding constraint.

**Why not linear logit-lens?**

A naive alternative caches each intermediate residual and applies the
trained ``PolicyHead`` *directly* to those caches. It's faster (~1.5×
one forward pass) but **unsound for BT4**.

BT4's ``PolicyHead``
(``leela_chat/_vendor/transformer_lens/components/leela_policyhead.py``)
is an attention-based head with learned key/query projections per
square plus separate promotion logits. Its parameters are tuned to the
geometry of the *final* residual stream. Applied to intermediate
residuals, the head can produce **plausible-looking but mechanistically
meaningless** output: sane-shaped distributions, sensible-looking
top-N orderings, all artefacts of geometric mismatch rather than of
the layer's computation.

The seductive failure mode: a "final-layer top-1 matches the engine's
policy" sanity check verifies hook wiring but does **not** distinguish
"early-layer projection is real" from "early-layer projection is
coincidentally plausible noise." Subtle rank-order errors at
intermediate layers would slip through, and our prose would describe a
layer's "computation" based on noise. Per CLAUDE.md §2 Rule 1, that
is the failure mode the project exists to prevent.

Linear logit-lens stays out of the codebase until equivalence with
zero-ablation is **empirically demonstrated**:

1. Concordant top-3 ranks at every layer for ≥5 diverse positions.
2. Positions include cases where the eventual top-1 differs from the
   layer-1 top-1 (the cases where linear logit-lens is most likely to
   diverge).
3. Side-by-side correlation on per-layer rank trajectories above some
   threshold (TBD, but rough target: Spearman rank correlation > 0.85
   per layer).

A future Chunk 6.5-or-later optimization can land linear logit-lens
behind these gates.

**What zero-ablation produces and what it doesn't.**

Under zero-ablation we get, for each layer L, a *real* BT4 policy
distribution — the model's actual ``PolicyHead`` output, run on a
modified residual stream. The distribution is interpretable in the
network's native units (probabilities over the 1858-move vocabulary).
What changes between L=0 and L=N is *only how much of the network's
computation was used*; the head's contribution is constant.

This eliminates the geometric-mismatch failure mode of linear logit-
lens.

The remaining caveat — that we cannot causally claim a specific layer
"computed" the rank change — is the encoding-vs-use distinction. See
``docs/INTERPRETABILITY_NOTES.md`` §3 for the prose contract that
enforces this.

================================================================
Output data model (lands in 6b)
================================================================

::

    class LayerPolicyPoint(BaseModel):
        layer: int            # 0 (most-ablated) through N (baseline)
        rank: int             # 1-based rank of this move at this layer
        prob: float           # softmax probability at this layer
        model_config = ConfigDict(frozen=True)

    class CandidateLayerTrajectory(BaseModel):
        uci: str
        san: str
        layer_trajectory: list[LayerPolicyPoint]   # one entry per layer
        entered_top5_at_layer: int | None
        exited_top5_at_layer: int | None
        peak_rank: int
        peak_layer: int
        network_dismissal: Literal[
            "rose_and_stayed", "rose_and_fell", "never_rose", "unknown",
        ]
        model_config = ConfigDict(frozen=True)

    class LayerEmergenceResult(BaseModel):
        candidates: list[CandidateLayerTrajectory]
        n_layers: int           # BT4 = 15
        model_config = ConfigDict(frozen=True)

================================================================
6b implementation contract
================================================================

The 6b implementation MUST:

1. Use zero-ablation only. No linear logit-lens.
2. Hook attachment: ``blocks.{L}.hook_resid_post`` (or the BT4-specific
   equivalent if the vendored HookedTransformer uses a different name).
3. Replace activation with ``torch.zeros_like(activation)`` (not mean
   ablation, not zero per dimension). The simplest, most-replicable
   choice; if it produces degenerate per-layer outputs in the
   integration test, *pivot to mean ablation* and document the
   decision in DECISIONS.md, NOT switch to linear logit-lens.
4. Run N+1 passes per call (one per layer, plus baseline).
5. Reuse the model's own ``PolicyHead`` — do not re-implement.
6. Return per-layer rank by reading the modified-final-residual policy
   output; rank is computed against the same 1858-move vocabulary the
   network uses.

================================================================
Limitations
================================================================

- **BT4-specific.** Different Lc0 networks (T82, T80, future networks)
  have different policy head architectures and would need
  re-validation of the methodology.
- **Observational.** See ``docs/INTERPRETABILITY_NOTES.md`` §6.2 and
  §3.4. No causal claims.
- **Single-position.** Each call is one position; batching across
  positions is a future optimization.
- **Hook-wiring failure mode.** If layer-N (baseline) ablation policy
  doesn't match ``extract_network_inference``'s policy top-1, the
  hooks are wired wrong. The integration test for 6b makes this the
  first thing it asserts.
- **Mean-ablation pivot path.** If zero-ablation produces degenerate
  outputs (e.g. uniform random at early layers on easy positions), we
  pivot to mean ablation (replace activation with the activation's
  mean over a calibration batch). Document the decision in
  DECISIONS.md if/when it happens.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import chess


def extract_layer_emergence(
    board: "chess.Board",
    *,
    model: object,
    focal_uci: str | None = None,
    top_n_per_layer: int = 10,
) -> "object":
    """Per-layer policy-rank trajectory for candidate moves.

    NOT YET IMPLEMENTED. Lands in Chunk 6b.

    The scaffold exists so that:

    - Module docstring (the methodology contract) lands in 6a v2 and
      is reviewable before 6b implementation begins.
    - Callers can reference the function signature in 6c planning
      without waiting on 6b.

    The 6b implementation must satisfy the contract in this module's
    docstring (§ "6b implementation contract").

    Args:
        board: Position to analyse.
        model: A HookedTransformer-wrapped BT4 model instance.
        focal_uci: If provided, the trajectory for this UCI is
            guaranteed to be returned even if the move's final-layer
            rank is outside ``top_n_per_layer``. Used by
            ``did_you_consider`` to track an arbitrary user-asked move.
        top_n_per_layer: Number of candidates to track per layer.
            Default 10.

    Returns:
        ``LayerEmergenceResult`` (model defined in 6b).

    Raises:
        NotImplementedError: always, in 6a v2.
    """
    raise NotImplementedError(
        "extract_layer_emergence lands in Chunk 6b. The module docstring "
        "is the written methodology contract; the implementation must "
        "satisfy it."
    )
