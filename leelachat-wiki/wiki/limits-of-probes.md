# Exploring the Limits of Probes for Latent Representation Edits in GPT Models

**Summary**: Linear probes on raw residual stream beat SAE-based probes for chess board-state editing: 88% vs 41% legal-move success rate. The paper introduces LMPM (Legal Move Probability Mass) as an objective intervention-quality metric and confirms the residual stream encodes Markovian board-state representations in deeper layers.

**Sources**: Exploring_the_Limits_of_Probes_for_Latent_Represen.pdf

**Last updated**: 2026-06-02

---

## Citation
Davis et al.. "Exploring the Limits of Probes for Latent Representation Edits in GPT Models." MDPI AI 2026.

## Contribution
Davis et al. train a 12-layer chess-playing GPT-2 on UCI move sequences and compare linear probes vs. Sparse Autoencoder (SAE)-based probes for classifying and editing the model's internal board-state representation. The paper introduces Legal Move Probability Mass (LMPM) as a metric for measuring how well a post-intervention forward pass respects chess rules. The central result — linear probes achieve 88.8% LMPM vs. ~41% for SAE-based probes — is a strong empirical prior for prioritising probe-based approaches over SAE-based ones for targeted representation editing. The paper also validates that the chess transformer's residual stream encodes a Markovian world model, with the same board position producing strongly aligned hidden states regardless of the move sequence that reached it.

## Method
A 12-layer GPT-2 (d=768) is trained from scratch on 94M Lichess games (June 2023) using UCI tokenization (72-token vocabulary). One Gated SAE per layer (16× expansion, d_lat=12,288) is trained on 1B tokens from residual stream activations at each layer. Linear probes (one per square × layer × phase) are trained to classify piece type and colour from cached hidden states. Five intervention types are compared: Random, Patch (time-shifted hidden state), Probe (negated probe weight column), SAE-Probe (probe on SAE latents, decoded back), and CD-Probe (SAE contrastive difference). Interventions are injected via NNsight hooks with a LayerNorm-matched scaling factor η=0.5. LMPM is computed over ~498k (linear) and ~404k (SAE) intervention samples. (source: Exploring_the_Limits_of_Probes_for_Latent_Represen.pdf)

## Key findings
- Linear probes trained on raw residual stream significantly outperform SAE-based probes: 88% LMPM on the target board vs. 41% for SAE probes. (source: Exploring_the_Limits_of_Probes_for_Latent_Represen.pdf, Abstract)
- Linear probe classification accuracy peaks at 95% at layers 7-8, while SAE-based probes peak at 88% at layer 10 — matching only the performance of a layer-3 linear probe. (source: Exploring_the_Limits_of_Probes_for_Latent_Represen.pdf, §4.3 / §4.4 / Table 4 / Table 5)
- SAE reconstruction is high (cross-entropy loss score 0.95, cosine similarity 0.97) yet SAE-based editing fails, suggesting reconstruction quality does not imply usable decomposition for intervention. (source: Exploring_the_Limits_of_Probes_for_Latent_Represen.pdf, §4.2 / Table 3 / §4.8)
- The chess GPT residual stream respects the Markovian property: same board positions from different move sequences yield significantly more aligned hidden states than random pairs at all layers (p<1e-3, Cohen's d up to 7.6 at layer 12). (source: Exploring_the_Limits_of_Probes_for_Latent_Represen.pdf, §4.5 / Table 6)
- SAEs fail not primarily because of information loss from sparsity but because unsupervised SAE training lacks a mechanism to enforce ontologically coherent decomposition aligned with chess concepts. (source: Exploring_the_Limits_of_Probes_for_Latent_Represen.pdf, §4.8)
- Linear probe weights are compositionally interpretable: a joint piece-type-and-color probe is approximated by the sum of separate type and color probes. (source: Exploring_the_Limits_of_Probes_for_Latent_Represen.pdf, §1 (Introduction))
- The patch intervention (using a valid hidden state from 6 tokens later) disrupts legal move generation: mean LMPM against the future board is only 0.620, confirming that raw time-shifted activations cannot serve as surgical edits. (source: Exploring_the_Limits_of_Probes_for_Latent_Represen.pdf, §4.7)
- The new LMPM metric quantifies how much probability mass a post-intervention model assigns to legal moves given the target board, providing an objective measure of intervention success beyond classification accuracy. (source: Exploring_the_Limits_of_Probes_for_Latent_Represen.pdf, §3.6 / §4.7)

## Relevance to Leela Chat
This paper directly informs Phase 3 (linear concept probes) and Phase 4 (SAEs) of Leela Chat. The central finding — linear probes on raw residual stream achieve 88% LMPM vs 41% for SAE-based probes — is a strong prior for investing in Phase 3 probes before spending Phase 4 effort on SAEs, and for not expecting SAE features to serve as surgical write targets. (source: Exploring_the_Limits_of_Probes_for_Latent_Represen.pdf)

The Markovian world-model result validates using a single board-state extraction per FEN across layers rather than requiring full move-history context, simplifying extractor design: a single FEN input to the extractor layer is sufficient, without needing to replay the full game sequence. (source: Exploring_the_Limits_of_Probes_for_Latent_Represen.pdf, §4.5 / Table 6)

The LMPM metric is a concrete template for Leela Chat's own eval framework: measuring what fraction of LLM probability mass over moves falls on moves that are legal given the extracted board state gives an objective grounding-quality signal beyond classification accuracy alone. This slots directly into the validator layer — a natural extension to the existing legal-move-string check. (source: Exploring_the_Limits_of_Probes_for_Latent_Represen.pdf, §3.6)

The failure mode identified — high SAE reconstruction quality coexisting with poor intervention effectiveness — is a pitfall to explicitly track if SAEs are adopted in Phase 4 for Leela Chat's feature-extraction pipeline. High reconstruction score should not be treated as a sufficient quality gate for Phase 4 SAE features; LMPM-style functional testing is needed. (source: Exploring_the_Limits_of_Probes_for_Latent_Represen.pdf, §4.8)

Note: the model studied is a 12-layer GPT-2 trained on UCI move sequences, not Lc0/BT4. Findings are directly relevant for methodology but results may differ on BT4 given its smolgen attention and policy head differences. (source: Exploring_the_Limits_of_Probes_for_Latent_Represen.pdf)

## Related pages
- [[linear-probes]]
- [[sparse-autoencoders]]
- [[board-state-probing]]
- [[residual-stream]]
- [[world-models]]
- [[transformer-chess-architecture]]
- [[measuring-progress-dictionary-learning]]
- [[chess-testbed-lm-state-tracking]]
- [[emergent-world-models-chess]]
- [[decoding-chess-mastery]]
- [[contrastive-sae-planning]]
