"""Adaptive MultiPV selection based on policy distribution character.

Per Chunk 7c.8 + ADR-0015. Replaces the hardcoded ``multipv=5`` default
from Chunk 7b with a policy-aware mapping: a position whose policy is
dominated by one move (top_prob > 70%, low entropy) wastes search
budget on multipv=5; a position with naturally flat policy (top_prob
< 30%, high entropy) benefits from multipv=10+ to surface real
candidates.

The mapping is contract-grade: Phase 3+ probes / SAEs / labels read
the chosen multipv from ``MCTSFindings.multipv_used``. Phase 3
training data must use a stable mapping; changing this function's
behavior requires an ADR amendment per ADR-0015's reversal contract.

Inputs are CHEAP to compute from existing forward-pass output:
- ``top_prob``: ``top_moves[0].policy_pct / 100.0`` (single field read).
- ``policy_entropy``: shannon entropy in nats over the top_moves'
  policy_pct distribution. One-line computation; no new extractor.
  7e (network confidence) may centralize this into a helper later;
  for now each handler computes inline.

Per-mode caps (post-mapping; calibrated against synthetic findings
token counts via anthropic.messages.count_tokens, 7c.8 calibration):
- ``"quick"``: max 3. Conservative to fit the <30s quick-mode gate.
- ``"deep"``: max 8. Calibration found multipv=15 is **2.67x** the
  token cost of multipv=5 — above the plan's 2x trigger for tightening.
  multipv=10 is 1.85x (acceptable on its own) but the policy was to
  cap at 8 once the 2x trigger fired on the higher value.
  Generous enough to accommodate contested positions under the
  super-GM quality target (ADR-0011) + the 300s cap (ADR-0012
  post-7c.6); tight enough to bound the prompt-input growth.
- ``"deepest"``: max 15 — design intent only; deepest mode is
  EXPLICITLY OUT-OF-SCOPE per ADR-0013. Documented for if/when
  deepest lands.

See ADR-0015 in ``docs/DECISIONS.md`` for the full mapping table
+ reversal cost.
"""
from __future__ import annotations

import math
from typing import Sequence

from leela_chat.handlers import Mode


# Per-mode caps applied after the policy-character mapping.
# "deepest" is OOS per ADR-0013; documented here as design intent only.
PER_MODE_CAP: dict[str, int] = {
    "quick": 3,
    "deep": 8,  # 7c.8 calibration: multipv=15 is 2.67x cost vs multipv=5;
                # dropped from initial 10 to 8 per plan's "drop if 2x" trigger.
    "deepest": 15,  # OOS per ADR-0013; design intent if deepest ever lands.
}


def policy_entropy_normalized(policy_pcts: Sequence[float]) -> float:
    """Shannon entropy over the policy distribution, normalized to [0, 1].

    Normalization divides raw entropy (nats) by the maximum possible
    entropy for the given number of candidates (``ln(N)``), making the
    metric comparable across different ``top_n`` values. This matters
    because :func:`extract_network_inference` truncates to ``top_moves``
    (typically top-5) — raw entropy is bounded by ``ln(5) ≈ 1.609``
    and can't reach the plan's original 2.5+ threshold without N >= 10+
    candidates. Normalized entropy of 1.0 means uniform distribution
    over the given candidates; 0.0 means one candidate has all the mass.

    Args:
        policy_pcts: Per-move policy probabilities IN PERCENT (0-100).
            Typically ``[m.policy_pct for m in network_inference.top_moves]``.
            Entries that round to 0 are skipped (don't contribute to
            entropy; convention: 0 * log(0) = 0).

    Returns:
        Normalized entropy in [0, 1] over the renormalized distribution.
        0.0 on empty input or single-candidate input (the entropy
        denominator ``ln(1) = 0`` is undefined; one-candidate distributions
        are by definition fully concentrated).
    """
    fractions = [pct / 100.0 for pct in policy_pcts if pct > 0]
    total = sum(fractions)
    if total <= 0 or len(fractions) <= 1:
        return 0.0
    normalized = [f / total for f in fractions]
    raw_entropy = -sum(p * math.log(p) for p in normalized if p > 0)
    max_entropy = math.log(len(fractions))
    return raw_entropy / max_entropy


def compute_adaptive_multipv(
    *,
    top_prob: float,
    policy_entropy: float,
    mode: Mode,
) -> int:
    """Map policy distribution character to an MCTS multipv value.

    Args:
        top_prob: The policy probability of the top move (0.0-1.0).
            ``network_inference.top_moves[0].policy_pct / 100.0``.
        policy_entropy: NORMALIZED Shannon entropy in [0, 1] over the
            policy distribution. Computed via
            :func:`policy_entropy_normalized`. 7e may centralize this
            into a network-confidence helper; for now handlers compute
            inline.
        mode: ``"quick"`` or ``"deep"`` (per ADR-0012). Selects the
            post-mapping cap from :data:`PER_MODE_CAP`.

    Returns:
        Integer multipv value to pass to ``extract_mcts_trajectory``
        / ``extract_mcts``. Always >= 1.

    Mapping rationale (per ADR-0015):

    +-----------------------+-----------+-------------------+------------------+
    | Policy character      | top_prob  | normalized entropy| Base multipv     |
    +=======================+===========+===================+==================+
    | Dominant top move     | > 0.70    | < 0.60            | 2                |
    +-----------------------+-----------+-------------------+------------------+
    | Moderate top move     | 0.30-0.70 | 0.60-0.90         | 5 (7b default)   |
    +-----------------------+-----------+-------------------+------------------+
    | Contested             | < 0.30    | 0.85-0.95         | 10               |
    +-----------------------+-----------+-------------------+------------------+
    | Naturally flat        | < 0.20    | > 0.95            | 15               |
    +-----------------------+-----------+-------------------+------------------+

    Thresholds calibrated against synthetic distributions (see
    Chunk 7c.8 PHASE_NOTES entry). With top-5 candidates:
    - very dominant ([90, 3, 3, 2, 2]): top=0.90, norm_ent ≈ 0.25
    - dominant ([80, 8, 6, 3, 2]): top=0.80, norm_ent ≈ 0.45
    - moderate ([50, 25, 15, 7, 3]): top=0.50, norm_ent ≈ 0.82
    - contested ([25, 22, 20, 18, 15]): top=0.25, norm_ent ≈ 0.99
    - flat (many similar): top<0.20, norm_ent → 1.00

    Both inputs are checked because they're partially redundant but
    NOT identical signals: top_prob is a single-move read, entropy
    is distribution-shape. A position with top_prob=0.40 and a sharp
    second-best (0.35) has lower entropy than one with top_prob=0.40
    and flat 0.05 across many — both warrant different multipv values.

    The cap is applied AFTER mapping. e.g., a naturally-flat position
    in quick mode picks base 15 but caps to 3.
    """
    # Dominant top move: one obvious candidate; budget spent on others
    # is largely wasted. multipv=2 captures the second-best for
    # completeness (the prompt cites both); going higher just adds
    # zero-visit PVs.
    if top_prob > 0.70 and policy_entropy < 0.60:
        base = 2
    # Naturally flat: many candidates with similar policy weight;
    # multipv=15 surfaces enough lines to make the contest visible
    # in prose. Capped per-mode below.
    elif top_prob < 0.20 and policy_entropy > 0.95:
        base = 15
    # Contested: clear group of contenders without a single dominant
    # move. multipv=10 lets the prompt cite the top-tier candidates
    # without arbitrary cutoffs.
    elif top_prob < 0.30 and policy_entropy > 0.85:
        base = 10
    # Moderate (the 7b default case): top move is favored but not
    # overwhelming; second/third lines are worth showing.
    else:
        base = 5

    cap = PER_MODE_CAP.get(mode, 5)
    return min(base, cap)
