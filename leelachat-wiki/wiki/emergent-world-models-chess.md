# Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models

**Summary**: A standard GPT trained on 16 million real Lichess games — with no a priori chess knowledge — develops a linear internal representation of board state recoverable with 99.6% accuracy, and also learns a latent Elo-estimation vector that can be steered to improve win rate by up to 2.6x. (source: Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models.pdf)

**Sources**: Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models.pdf

**Last updated**: 2026-06-02

---

## Citation
Karvonen. "Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models." COLM 2024.  (arXiv:2403.15498)

## Contribution
Karvonen trains 8- and 16-layer GPT models (25M and 50M parameters) on 16 million Lichess games using only next-character prediction, giving the model no a priori chess knowledge. The paper establishes that world models emerge from non-synthetic, human-game data — contradicting earlier results that attributed this capability to synthetic datasets. It also introduces latent Elo estimation as a probe target, and validates both the board-state and skill representations causally via residual-stream interventions. (source: Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models.pdf)

## Method
Linear probes (512×13 softmax classifiers) are trained per board square per layer on 10k held-out games, predicting one of 13 piece-type states using a (Mine/Yours/Empty) framing. A second probe family targets Elo classification (below 1550 vs. above 2050). Interventions add or subtract probe-derived or contrastive-activation vectors directly to the residual stream, with a scaling coefficient α; a dynamic-scale variant targets a specific probe logit of −10. Skill vectors are derived both from probe weight differences and from mean-activation contrasts over 2,000 high-skill vs. 2,000 low-skill games. The models are evaluated on legal move rate and win rate against Stockfish level 0. (source: Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models.pdf)

## Key findings
- A standard GPT trained on real human chess games (not synthetic data) develops a linear internal representation of the board state recoverable by linear probes, contradicting the earlier finding that world models only emerge from synthetic datasets. (source: Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models.pdf, §1 Intro / §3.1)
- The 16-layer model's board-state peak accuracy occurs at layer 12, while the 8-layer model peaks at layer 6 — the deeper model uses proportionally more layers to form its ultimately more accurate board state. (source: Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models.pdf, §3.1 / Fig. 3)
- Linear probes on the 16-layer residual stream classify board squares with 99.6% accuracy and F1 > 0.98 on all 13 piece-type states; randomly initialized models score ~74.7% accuracy and F1 in the 0.5–0.6 range. (source: Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models.pdf, Table 1 / Table 9)
- The model learns to estimate player Elo as a latent variable: a probe trained on the trained model achieves 90.5% Elo classification accuracy vs. 69.3% for a random-weight baseline, demonstrating unsupervised skill estimation. (source: Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models.pdf, §3.2 / Table 1)
- Subtracting a piece's probe vector from the residual stream causes the model to treat that piece as absent; legal move rate on the hypothetical modified board rises from 40.5% (no intervention) to 92%. (source: Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models.pdf, §4.2 / Table 2)
- Adding a contrastive skill vector to the 16-layer model's activations improves its win rate against Stockfish level 0 by 2.6x on randomly initialized boards (16.7% → 43.2%), validating a causal role for the Elo representation. (source: Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models.pdf, §4.3 / Table 3)
- Elo-probe accuracy peaks at the final layer (unlike board-state accuracy which peaks mid-to-late), suggesting skill estimation occurs later in the computation than board-state tracking, possibly gating move selection. (source: Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models.pdf, §3.2 / Fig. 3)
- The Elo probe classifies 99.4% of games starting from 20 random moves as low-skill, explaining why a positive skill intervention restores so much win-rate on random boards. (source: Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models.pdf, §4.3)

## Relevance to Leela Chat
**Phase 3 (linear concept probes):** this paper is the direct methodological template for training per-square, per-layer linear probes on the residual stream of BT4, including the crucial (Mine/Yours/Empty) framing and the finding that peak accuracy occurs in mid-to-late layers rather than immediately. Leela Chat should replicate the per-layer sweep to identify which layer(s) of BT4's 15-layer architecture to hook for board-state extraction in the `board_state` extractor — note that absolute layer indices from this paper do not transfer to BT4 (the paper's 16-layer model peaks at layer 12; BT4 layer-sweep is needed separately). (source: Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models.pdf)

**Phase 6 (differential/counterfactual):** the vector-subtraction intervention technique — subtracting a probe direction from the residual stream to causally erase a piece — is a direct prototype for Leela Chat's counterfactual handlers. The 92% board-intervention success rate and the documented unintended side-effects on neighboring pieces (Fig. 5) are important calibration data for what the `why_is_my_move_wrong` and counterfactual handlers can promise users. The contrastive-activation derivation method (mean contrast over high/low game groups) is reusable for any latent variable Leela Chat wants to steer, even though Lc0 is not an Elo-conditioned model. (source: Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models.pdf)

## Related pages
- [[world-models]]
- [[linear-probes]]
- [[board-state-probing]]
- [[residual-stream]]
- [[activation-patching]]
- [[transformer-chess-architecture]]
- [[measuring-progress-dictionary-learning]]
- [[chess-testbed-lm-state-tracking]]
- [[acquisition-chess-knowledge-alphazero]]
- [[limits-of-probes]]
- [[contrastive-sae-planning]]
