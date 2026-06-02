# Maia-2: A Unified Model for Human-AI Alignment in Chess

**Summary**: Single skill-conditioned ResNet that predicts human moves across the full rating spectrum, replacing 9 independent Maia-1 models with coherent skill-aware attention.

**Source**: arXiv:2409.20553 (Tang/Tan et al., NeurIPS 2024) + CSSLab/maia2@16c1383

**Verified**: paper-derived, not reproduced

**Last updated**: 2026-06-02

---

## Contribution
Maia-2 is a unified neural network that predicts human chess moves across the entire skill spectrum (roughly 1100–2600+ Lichess rapid rating) using a single model conditioned on both the active player's and opponent's skill levels via categorical embeddings injected into a skill-aware attention mechanism. Unlike its predecessor Maia-1, which used 9 entirely independent models each targeting a 100-point rating band, Maia-2 produces coherent predictions where the probability assigned to the correct move changes smoothly and often monotonically as skill increases. Trained on 169M Lichess rapid games (9.1B positions, Jan 2013–Nov 2023), it surpasses Maia-1 by approximately 2 percentage points in average human-move prediction accuracy and reduces perplexity from 4.67 to 4.07 bits — using only 23.3M parameters versus Maia-1's effective 92M. This makes Maia-2 the strongest available behavioral oracle for what a human at a given rating would play in a position, which is directly useful as a grounded signal source when explaining human move choices.

## Method
Position is encoded as a single-timestep multi-channel 8×8 tensor (no 6-board history). Skill levels are mapped to categorical embeddings that are injected into the query vectors of multi-head self-attention operating over channel-wise patches of the ResNet feature maps, treating each feature-map channel as a sequence token. Auxiliary classification heads for move meta-information (piece moved, capture, origin/destination square, check) are trained jointly with the policy and value heads using balanced cross-entropy objectives; games with mismatched skill levels are over-sampled to cover the full skill-combination matrix. Linear probes on residual-stream activations before and after the skill-aware attention blocks are used to verify which chess concepts do and do not respond to the injected skill level.

## Key findings
- Maia-2 surpasses all baselines in average human-move prediction accuracy with 53.25% vs. 51.39% for Maia-1500, ~2 pp gain. (source: arXiv:2409.20553, Table 1; Section 4.1)
- Maia-2 reduces average perplexity from 4.67 bits (Maia-1900) to 4.07 bits, the most striking gain on Advanced and Master players. (source: arXiv:2409.20553, Table 2; Section 4.1)
- 27% of test positions are treated monotonically by Maia-2 (probability of correct move increases with skill) vs. 1% for Maia-1. (source: arXiv:2409.20553, Table 4; Section 4.2)
- Changing own skill while fixing opponent can alter Maia-2's top move prediction up to 22% of the time; changing opponent skill alone changes it only up to 6%. (source: arXiv:2409.20553, Figure 3(B); Section 4.2)
- Architecture alone accounts for ~73% of the accuracy improvement; the larger training dataset accounts for the remaining ~27%. (source: arXiv:2409.20553, Section 4.1 (Maia-2subset comparison))
- Maia-2 uses 23.3M parameters vs. Maia-1's effective 92M (9 × 10.3M), achieving better accuracy with far fewer parameters. (source: arXiv:2409.20553, Section 4.1)
- Linear probes confirm that skill-dependent concepts (board eval, piece bonuses) vary after skill-aware attention, while skill-independent concepts (bishop pair, queen-capture legality) do not. (source: arXiv:2409.20553, Figure 4; Section 4.2 (Chess concept understanding))
- Training used 169M Lichess rapid games (9.1B positions, Jan 2013–Nov 2023) with cross-skill balancing; a single-board-state input (no 6-board history) is sufficient for state-of-the-art results. (source: arXiv:2409.20553, Section 4 (opening paragraph); Section 3.2)

## Relevance to Leela Chat
For Phase 6 (differential/counterfactual analysis), Maia-2's skill-conditioned policy distribution is a valuable human-behavioral baseline: when Leela and Maia-2 disagree on a move's probability across skill bands, that divergence is grounded signal for the `why_is_my_move_wrong` handler rather than an LLM confabulation (directly supporting Rule 1). The pip-installable inference API (`maia2.inference.inference_each`) accepts a FEN plus two ELO values and returns move probabilities, making it straightforward to add as an extractor alongside Lc0 and Stockfish without a subprocess wrapper. The [[linear-probes]] methodology used in Figure 4 to validate skill-aware attention is directly reusable in Phase 3 (concept probes) as a sanity-check pattern: train probes on activations before and after a targeted architectural boundary to confirm a concept responds to the variable of interest. Note that Maia-2 uses a non-BT4 network and single-board-state input (no 6-board history), so its [[residual-stream]] activations are not interoperable with Lc0/BT4 probes or SAEs; treat it as a black-box behavioral oracle, not a mechanistic interpretability target.

## Related pages
- [[maia-chess]]
- [[maia-human-alignment]]
- [[maia-aligning-superhuman-ai]]
- [[skill-adaptation-saes]]
