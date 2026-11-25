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

**Method.** For each layer index L ∈ [0..N], run a forward pass that
masks the *contributions* of layers L..N-1. A layer's contribution to
the residual is its attention output (``hook_attn_out``) plus its
MLP output (``hook_mlp_out``); both are exposed as named HookPoints on
``EncoderLayer`` (see
``leela_chat/_vendor/transformer_lens/components/leela_encoder.py``
lines 161-165). Setting both to ``torch.zeros_like(activation)`` makes
the layer a no-op for the residual stream (modulo its layer-norms,
which still run — they're not what carries information across blocks).

For each pass:

- L=0: hooks zero ``hook_attn_out`` and ``hook_mlp_out`` at every block
  0..N-1. Result: pure embedding-through-layer-norms (no transformer
  blocks contribute).
- L=k for k ∈ [1..N-1]: hooks zero ``hook_attn_out`` and ``hook_mlp_out``
  at every block k..N-1. Result: residual contains contributions only
  from blocks 0..k-1.
- L=N: no hooks. Baseline pass; matches what
  :func:`leela_chat.extractors.network_inference.extract_network_inference`
  produces.

The model's own ``PolicyHead`` reads the final residual at every pass.
We do **not** invoke the head on intermediate residuals. This sidesteps
the geometric-mismatch failure mode that disqualified linear logit-lens.

**Critical clarification.** Zero-ablating the *output residual*
(``hook_resid_post`` or ``resid_post_after_ln``) at layer L would zero
EVERYTHING in the residual at that point, not just layer L's
contribution. That's why we hook the per-layer *contributions*
(``hook_attn_out`` + ``hook_mlp_out``) instead. The Iterative Inference
paper's "layer ablation" is equivalent to this.

The vendored ``HookedRootModule.run_with_hooks`` at
``leela_chat/_vendor/transformer_lens/hook_points.py:417`` supports
this directly — a hook can return a tensor that replaces the
activation. No modifications to the vendored module needed.

Cost: N+1 forward passes per call. For BT4 (N=15) on a 4090,
~16 passes × ~50 ms ≈ 800 ms. Well inside the Phase 1 30 s end-to-end
ceiling. Cost is not a binding constraint.

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
2. Hook attachment: ``blocks.{L}.hook_attn_out`` AND
   ``blocks.{L}.hook_mlp_out`` for each layer being ablated. Both must
   be zeroed; zeroing only one leaves residual contributions from the
   other half of the block.
3. Replace activation with ``torch.zeros_like(activation)`` (not mean
   ablation, not zero per dimension). The simplest, most-replicable
   choice; if it produces degenerate per-layer outputs in the
   integration test, *pivot to mean ablation* and document the
   decision in DECISIONS.md, NOT switch to linear logit-lens.
4. Run N+1 passes per call: L=k zeroes layers k..N-1; L=N is baseline
   (no hooks).
5. Reuse the model's own ``PolicyHead`` (and the rest of the network's
   forward path) — do not re-implement.
6. Return per-layer rank by reading the modified-final-residual policy
   output; rank is computed against legal moves only (masked +
   re-normalised softmax), matching
   ``extract_network_inference``'s convention.

================================================================
External-tool pitfall reference (CLAUDE.md §8)
================================================================

This extractor uses HookedTransformer hookpoints by name (e.g.
``blocks.{L}.hook_attn_out``). Hookpoint name strings are NOT
validated at attach-time — passing an unknown name silently no-ops
the hook. Same class of pitfall as Lc0's ``LogLiveStats`` UCI
``setoption`` silently-no-op (CLAUDE.md §8 "External-tool / protocol
pitfalls"). Verification pattern: integration tests must assert
behavior depends on the hook, not just that the call doesn't raise.
The 6b integration test's hook-wiring sanity gate
(``test_baseline_top1_matches_network_inference_starting_position``)
is this project's canonical defense against silent-hook-no-op for
this extractor. Re-validate it whenever the HookedTransformer
vendored module is bumped or BT4 weights are changed.

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

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import chess


# === data models ============================================================


class LayerPolicyPoint(BaseModel):
    """One candidate's (rank, probability) at a specific BT4 layer.

    A layer index ``layer == L`` means: "what BT4 predicts when its
    residual contains contributions only from blocks 0..L-1." Layer
    ``L=N`` (where N is the total block count) is the baseline pass —
    matches :func:`extract_network_inference`.
    """

    layer: int = Field(
        ..., ge=0,
        description="0 = most-ablated (no blocks contribute); N = baseline.",
    )
    rank: int = Field(
        ..., ge=1,
        description=(
            "1-based rank of this move under the legal-move-masked + "
            "renormalised policy at this layer."
        ),
    )
    prob: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Renormalised softmax probability over legal moves at this "
            "layer. Use as a rank signal; not calibrated as probability."
        ),
    )

    model_config = ConfigDict(frozen=True)


# Pre-computed dismissal tags. See module docstring §"6b implementation
# contract" and INTERPRETABILITY_NOTES.md §3 for the prose contract that
# governs how these tags surface in handler prompts.
NetworkDismissalTag = Literal[
    "rose_and_stayed", "rose_and_fell", "never_rose", "unknown",
]


class CandidateLayerTrajectory(BaseModel):
    """Per-layer rank trajectory for one candidate move across all layers.

    Pre-computes ``network_dismissal`` and a few peak markers so the
    handler / LLM doesn't have to derive them from raw trajectory data.
    """

    uci: str = Field(..., description="Move in UCI form.")
    san: str = Field(..., description="Move in SAN.")
    layer_trajectory: list[LayerPolicyPoint] = Field(
        ...,
        description=(
            "One entry per layer, ordered by ascending layer index "
            "(L=0 first, L=N last)."
        ),
    )
    entered_top5_at_layer: int | None = Field(
        None,
        description=(
            "First layer L at which this candidate's rank ≤ 5, or None "
            "if it never entered the top 5."
        ),
    )
    exited_top5_at_layer: int | None = Field(
        None,
        description=(
            "Last layer L at which rank ≤ 5 IF the candidate exited "
            "the top 5 at any later layer. None if it stayed in top 5 "
            "through the final layer, OR if it never entered top 5."
        ),
    )
    peak_rank: int = Field(
        ..., ge=1,
        description="Best (lowest) rank reached across all layers.",
    )
    peak_layer: int = Field(
        ..., ge=0,
        description="The layer at which ``peak_rank`` was reached.",
    )
    network_dismissal: NetworkDismissalTag = Field(
        ...,
        description=(
            "Pre-computed categorical tag. See module docstring + "
            "INTERPRETABILITY_NOTES.md §3 for the prose contract."
        ),
    )

    model_config = ConfigDict(frozen=True)


class LayerEmergenceResult(BaseModel):
    """Result of one :func:`extract_layer_emergence` call.

    Holds per-candidate layer trajectories plus metadata about the
    forward-pass set that produced them.
    """

    candidates: list[CandidateLayerTrajectory] = Field(
        ...,
        description=(
            "Per-candidate trajectories. Includes the top-N candidates "
            "at the final (baseline) layer plus any focal move passed "
            "in via ``focal_uci`` even if outside the final-layer top-N."
        ),
    )
    n_layers: int = Field(
        ..., ge=1,
        description="Total block count (BT4 = 15).",
    )

    model_config = ConfigDict(frozen=True)


# === implementation =========================================================


def _zero_hook(tensor: "torch.Tensor", *, hook) -> "torch.Tensor":  # noqa: ANN001
    """Return zeros to ablate a layer's contribution.

    Hook protocol per the vendored HookedTransformer
    (``hook_points.py:51-54``): hook receives ``(tensor, *, hook)`` and
    returns a tensor (replaces the activation) or None (leaves it
    unchanged).
    """
    return torch.zeros_like(tensor)


def _build_ablation_hooks_for_layer(L: int, n_layers: int) -> list[tuple[str, callable]]:
    """Hook list to ablate layers L..n_layers-1.

    For each ablated block, BOTH ``hook_attn_out`` AND ``hook_mlp_out``
    must be zeroed — these are the two contributions a block adds to
    the residual stream (see leela_encoder.py:172-198). Zeroing only
    one leaves the other half's contribution in the residual.

    L == n_layers → empty list (the baseline pass).
    """
    hooks: list[tuple[str, callable]] = []
    for L_ablate in range(L, n_layers):
        hooks.append((f"blocks.{L_ablate}.hook_attn_out", _zero_hook))
        hooks.append((f"blocks.{L_ablate}.hook_mlp_out", _zero_hook))
    return hooks


def _legal_move_probs(
    policy_logits: np.ndarray,
    legal_indices: np.ndarray,
) -> np.ndarray:
    """Legal-move-masked softmax + renormalise to a probability vector
    over the legal moves. Mirrors network_inference.py's convention so
    per-layer ranks are comparable to the baseline extractor.
    """
    legal_logits = policy_logits[legal_indices]
    # Numerically-stable softmax.
    legal_logits = legal_logits - legal_logits.max()
    probs = np.exp(legal_logits)
    probs = probs / probs.sum()
    return probs


def _classify_dismissal(
    layer_trajectory: list[LayerPolicyPoint],
) -> NetworkDismissalTag:
    """Derive the categorical dismissal tag from a candidate's trajectory.

    Rules:
    - ``rose_and_stayed``: rank ≤ 5 at SOME layer AND rank ≤ 10 at the
      final (baseline) layer.
    - ``rose_and_fell``: rank ≤ 5 at SOME earlier layer AND rank > 10
      at the final layer.
    - ``never_rose``: rank > 5 at every layer.
    - ``unknown``: fewer than 2 trajectory points (can't tell a story).
    """
    if len(layer_trajectory) < 2:
        return "unknown"
    ranks = [p.rank for p in layer_trajectory]
    final_rank = ranks[-1]
    entered_top5 = any(r <= 5 for r in ranks)
    if not entered_top5:
        return "never_rose"
    if final_rank <= 10:
        return "rose_and_stayed"
    return "rose_and_fell"


def _build_candidate_trajectory(
    uci: str,
    san: str,
    per_layer_probs: list[np.ndarray],
    per_layer_legal_uci_order: list[list[str]],
) -> CandidateLayerTrajectory:
    """Assemble a CandidateLayerTrajectory from per-layer policy
    distributions over legal moves.

    Args:
        uci: The candidate move's UCI string.
        san: The candidate move's SAN.
        per_layer_probs: One (legal-only) probability vector per layer,
            in ascending layer order.
        per_layer_legal_uci_order: Per layer, the UCI of each entry in
            the corresponding probability vector. Same length / order
            as the probability vectors; passed once per layer so we can
            look up the candidate's index per layer.
    """
    points: list[LayerPolicyPoint] = []
    for layer_idx, (probs, ucis) in enumerate(
        zip(per_layer_probs, per_layer_legal_uci_order)
    ):
        if uci not in ucis:
            # Position is terminal at this layer (no legal moves) or the
            # candidate isn't legal in this position. Should be rare;
            # report rank=len(ucis)+1 and prob=0 to signal "outside the
            # measured set."
            points.append(LayerPolicyPoint(
                layer=layer_idx,
                rank=max(1, len(ucis) + 1),
                prob=0.0,
            ))
            continue
        i = ucis.index(uci)
        # Rank: 1-based index in the descending-prob order.
        order = np.argsort(-probs)
        rank = int(np.where(order == i)[0][0]) + 1
        points.append(LayerPolicyPoint(
            layer=layer_idx,
            rank=rank,
            prob=float(probs[i]),
        ))

    # Peak rank (= minimum rank value).
    peak_rank = min(p.rank for p in points)
    peak_layer = next(p.layer for p in points if p.rank == peak_rank)

    # entered_top5_at_layer.
    entered = next((p.layer for p in points if p.rank <= 5), None)

    # exited_top5_at_layer: only meaningful if entered AND the final
    # layer has rank > 5.
    exited: int | None = None
    if entered is not None and points[-1].rank > 5:
        # Walk backward from the final layer to find the LAST layer
        # where rank ≤ 5 — that's where it last exited.
        for p in reversed(points):
            if p.rank <= 5:
                exited = p.layer
                break

    return CandidateLayerTrajectory(
        uci=uci, san=san,
        layer_trajectory=points,
        entered_top5_at_layer=entered,
        exited_top5_at_layer=exited,
        peak_rank=peak_rank,
        peak_layer=peak_layer,
        network_dismissal=_classify_dismissal(points),
    )


def extract_layer_emergence(
    board: "chess.Board",
    *,
    model,  # HookedTransformer — typed loose to avoid forcing the heavy import
    focal_uci: str | None = None,
    top_n_per_layer: int = 10,
) -> LayerEmergenceResult:
    """Per-layer rank trajectory for top-N candidates (plus optional focal).

    See module docstring for the full methodology contract. Behaviour
    summary: N+1 forward passes through BT4, each ablating layers k..N-1
    (for k = 0..N) by zeroing ``hook_attn_out`` + ``hook_mlp_out`` at
    every block in that range. Per-pass policy is read off the model's
    own PolicyHead. Per-candidate ranks are computed against legal
    moves only (masked + renormalised softmax) so they're directly
    comparable to :func:`extract_network_inference`'s rank-1.

    Args:
        board: Position to analyse. Caller-owned; not mutated.
        model: Loaded BT4 ``HookedTransformer`` (from
            :func:`leela_chat.network.load_bt4`). Already on the target
            device.
        focal_uci: If provided, this UCI's trajectory is included in
            the result even when it's outside the final-layer top-N.
            Used by ``did_you_consider`` (Chunk 6c) to track an
            arbitrary user-asked move.
        top_n_per_layer: How many of the final-layer top candidates to
            track. Default 10.

    Returns:
        :class:`LayerEmergenceResult` with one trajectory per tracked
        candidate. Empty ``candidates`` list only on terminal positions
        (no legal moves).
    """
    from leela_chat import _vendor  # noqa: F401  (sys.path hack side effect)
    from transformer_lens.components.leela_features import LeelaBoard

    from leela_chat.network import encode_legal_move

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return LayerEmergenceResult(candidates=[], n_layers=model.cfg.n_layers)

    legal_indices = np.array([encode_legal_move(board, mv) for mv in legal_moves])
    legal_uci_order: list[str] = [mv.uci() for mv in legal_moves]

    # Build the BT4 input.
    lb = LeelaBoard.from_fen(board.fen(), history_synthesis=True)
    feats = lb.lcz_features(no_history=False)
    device = next(model.parameters()).device
    x = torch.from_numpy(feats).to(dtype=torch.float32, device=device).unsqueeze(0)

    n_layers = int(model.cfg.n_layers)

    # N+1 forward passes; collect per-layer legal-move probabilities.
    per_layer_probs: list[np.ndarray] = []
    with torch.no_grad():
        for L in range(n_layers + 1):
            hooks = _build_ablation_hooks_for_layer(L, n_layers)
            if hooks:
                outputs = model.run_with_hooks(x, fwd_hooks=hooks)
            else:
                outputs = model(x)
            # BT4's forward returns a list [policy_logits, wdl, mlh].
            policy_logits = outputs[0].cpu().numpy()[0]
            probs = _legal_move_probs(policy_logits, legal_indices)
            per_layer_probs.append(probs)

    # Determine which candidates to track: final-layer top-N, plus the
    # focal move if not in the top-N.
    baseline_probs = per_layer_probs[-1]
    baseline_order = np.argsort(-baseline_probs)[:top_n_per_layer]
    tracked_indices = set(int(i) for i in baseline_order)
    if focal_uci is not None and focal_uci in legal_uci_order:
        tracked_indices.add(legal_uci_order.index(focal_uci))

    # Build per-layer-legal-uci-order list (constant: legal moves don't
    # change between forward passes for the same position).
    per_layer_legal_uci_order = [legal_uci_order] * (n_layers + 1)

    candidates: list[CandidateLayerTrajectory] = []
    for idx in sorted(
        tracked_indices, key=lambda i: -baseline_probs[i],
    ):  # final-layer rank order
        mv = legal_moves[idx]
        uci = mv.uci()
        san = board.san(mv)
        candidates.append(_build_candidate_trajectory(
            uci, san, per_layer_probs, per_layer_legal_uci_order,
        ))

    logger.debug(
        f"layer_emergence: n_layers={n_layers} "
        f"tracked={len(candidates)} "
        f"focal={focal_uci} "
        f"baseline_top1={candidates[0].san if candidates else None}"
    )

    return LayerEmergenceResult(candidates=candidates, n_layers=n_layers)
