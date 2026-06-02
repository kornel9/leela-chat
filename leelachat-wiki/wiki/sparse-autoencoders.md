# Sparse autoencoders & dictionary learning

**Summary**: Sparse autoencoders (SAEs) and related dictionary-learning methods decompose a chess network's dense internal activations (e.g. the Lc0/BT4 residual stream) into a larger set of mostly-inactive, individually interpretable features — the unsupervised route to turning Leela's hidden state into named chess concepts that downstream tools can cite.

**Sources**: Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents.pdf, Contrastive Sparse Autoencoders for Interpreting.pdf, Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models.pdf, Scaling and evaluating sparse autoencoders.pdf, Sparse Feature Circuits.pdf, UNDERSTANDING SKILL ADAPTATION IN TRANSFORMERS USING SPARSE AUTOENCODERS.pdf, Exploring_the_Limits_of_Probes_for_Latent_Represen.pdf, Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf

**Last updated**: 2026-06-02

---

## What it is

A sparse autoencoder learns an over-complete dictionary that reconstructs a layer's activation vector from a sparse linear combination of feature directions, so each feature ideally corresponds to one human-interpretable concept rather than a polysemantic mixture. This matters for Leela Chat because the BT4 residual stream is dense and entangled; a good SAE turns it into a firing list of chess-meaningful units (king-safety corridors, threatened squares, rook threats) that the extractor layer can hand to the translator as grounded findings. Training revolves around a reconstruction-vs-sparsity tradeoff, and the open research problem is how to train and evaluate dictionaries whose features are both faithful (low reconstruction error) and genuinely usable for interpretation and intervention.

## In the literature

- **TopK recipe and scaling laws.** Gao et al. (OpenAI) replace the ReLU+L1 penalty with a TopK activation that keeps only the k largest latents, directly controlling sparsity, removing activation shrinkage, and (with tied init + an AuxK auxiliary loss) cutting dead latents to ~7% even in a 16M-latent autoencoder on GPT-4; reconstruction MSE follows clean power laws in both autoencoder size and sparsity (source: Scaling and evaluating sparse autoencoders.pdf, §2.3, §2.4, §3.1.4). Notably, chess PGN notation surfaces as its own recurring dense feature (~20%) in their GPT-2 SAEs, evidence that structured chess symbols cluster into features (source: Scaling and evaluating sparse autoencoders.pdf, §E.6). See [[scaling-evaluating-saes]].

- **Evaluating dictionaries with known board-state ground truth.** Karvonen et al. exploit chess/Othello models that have formally specifiable features to define supervised SAE metrics — coverage and board reconstruction — and introduce p-annealing (a sparsity penalty transitioning from convex L1 toward L0) that matches Gated SAEs at 33% less compute; their best chess SAE reaches board-reconstruction F1=0.85 vs 0.01 for a random-weight model, but linear probes still beat it (0.98 vs 0.85), and complex tactics like fork (0.13–0.38) and pin (0.20–0.33) are poorly covered (source: Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models.pdf, Table 1, Table 4, §5). See [[measuring-progress-dictionary-learning]].

- **Contrastive SAEs for planning in Leela.** Poupart's CSAE trains directly on Leela Chess Zero (a ResNet-based Lc0 variant), pairing the MCTS root state with an optimal vs suboptimal trajectory state and partitioning the dictionary into common (root-state) c-features and differentiating (planning) d-features; it hits R²=0.81 reconstruction with ~73 active features of 2048 (3.5%), and d-features carry higher trajectory entropy (3.24 vs 2.81) and are slightly more discriminative of optimality (F1 0.566 vs 0.537) (source: Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents.pdf, Table 4, Table 1). The sibling write-up adds the "blinking problem": Lc0 flips the board for side-to-move, so paired boards must share a colour for correct interpretation (source: Contrastive Sparse Autoencoders for Interpreting.pdf, §5.1). See [[contrastive-sae-planning]] and [[contrastive-sae-interpreting]].

- **SAE features as causal circuit nodes.** Marks et al. use SAE features (plus SAE error terms kept as first-class nodes) as the units of a causal computation graph, scoring importance with attribution / integrated-gradients patching; sparse feature circuits need far fewer nodes than neuron circuits for the same faithfulness (~100 features vs ~1500 neurons on Pythia-70M; ~500 vs ~50,000 on Gemma-2-2B), and their SHIFT technique surgically ablates task-irrelevant features to reach near-oracle worst-group accuracy (89.0% vs 91.9%) without disambiguating labels (source: Sparse Feature Circuits.pdf, Fig. 3, Table 2). See [[sparse-feature-circuits]].

- **Steering skill via SAE features.** The skill-adaptation work trains ReLU/L1 SAEs (width 2048, L1 alpha=1e-5, chosen by a Chess-Concept-AUC + loss metric) on Maia-2's skill-aware transformer blocks; specific square-level threat features (e.g. "capture possible on e3", CC-AUC 0.88–0.90) are causal — additively boosting one raises best-move accuracy from 0% to over 50% in transitional positions while random features stay near 0%, with a dose-response on intervention strength (source: UNDERSTANDING SKILL ADAPTATION IN TRANSFORMERS USING SPARSE AUTOENCODERS.pdf, §5.1, §5.2.2, Fig. 4). See [[skill-adaptation-saes]].

- **Transcoders + Lorsa on the pinned BT4.** Lin et al. (Fudan) give the first full sparse decomposition of Lc0/BT4 — replacing MLPs with TopK Transcoders and attention with Low-Rank Sparse Attention (Lorsa, preserving Smolgen and bilinear attention), with k=30 and expansion 16 (16,384 features/layer) — and build causally-verified reasoning pathways via feature steering; their features are far more interpretable than raw neurons (4.15/3.73 vs 2.05 human-consistency) and ablating a single own-pawn-detection feature flips P(Qxh7+) from 28.8% to ~91% (source: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf, Table 1, §2.2, Table 4). See [[fudan-tracing-thought]].

- **A skeptical counterpoint: SAEs can fail as edit targets.** Davis et al. directly compare linear probes vs SAE-based probes for editing a chess GPT's board state: linear probes on the raw residual stream reach 88% legal-move-probability-mass on the target board after intervention, while SAE-based probes reach only ~41% (barely above the unintervened baseline), despite high SAE reconstruction (cosine 0.97) — they conclude high reconstruction does not imply a usable, ontologically coherent decomposition (source: Exploring_the_Limits_of_Probes_for_Latent_Represen.pdf, Abstract, §4.8). This disagrees with the optimistic framing above: Karvonen et al. report the same probe-beats-SAE gap on board-state recovery (0.98 vs 0.85), so the two converge that SAEs lag [[linear-probes]] for board-state edits even where SAEs find interpretable features (source: Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models.pdf, §5). See [[limits-of-probes]].

## Relevance to Leela Chat

This concept is the core of Phase 4 (Sparse Features), where SAEs are loaded for several BT4 layers and an extractor produces a per-position firing list (the Leela Chat target of 30–80 firings matches CSAE's ~73 active features) that becomes findings the translator may cite and the validator checks against the firing list. The TopK + AuxK + tied-init recipe and the four quality metrics (downstream loss, probe recovery, N2G explainability, ablation sparsity) are the recommended baseline and exit-criteria template for training on BT4 (source: Scaling and evaluating sparse autoencoders.pdf, §4); the Fudan Transcoders/Lorsa configuration on the exact pinned network is the named go/no-go evaluation for the phase (source: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf, §2.2). Feature labeling (Phase 5) consumes these features, and feature-steering interventions extend into the differential/counterfactual handlers (Phase 6) and the reasoning-pathway extractor (Phase 7) — but the probe-vs-SAE evidence above is a standing caution to validate SAE features (coverage, intervention LMPM) before relying on them, and to keep linear probes as the fallback for concepts SAEs cover poorly.

## Related pages

- [[scaling-evaluating-saes]] — TopK SAE recipe, dead-latent mitigations, scaling laws.
- [[measuring-progress-dictionary-learning]] — coverage / board-reconstruction metrics and p-annealing.
- [[contrastive-sae-planning]] — contrastive SAE for Leela planning features (c/d partition).
- [[contrastive-sae-interpreting]] — sibling write-up; the side-to-move "blinking problem".
- [[sparse-feature-circuits]] — SAE features as nodes in causal circuits (SHIFT ablation).
- [[skill-adaptation-saes]] — SAE features that causally steer Maia-2's skill level.
- [[limits-of-probes]] — SAE-based edits underperform linear probes for board-state.
- [[fudan-tracing-thought]] — Transcoders + Lorsa full decomposition of the pinned BT4.
- [[linear-probes]] — supervised alternative that beats SAEs on board-state recovery/edits.
- [[feature-labeling]] — turning SAE features into named concepts (Phase 5).
- [[circuit-discovery]] — composing features into causal graphs.
- [[activation-patching]] — the causal-attribution machinery behind feature circuits and steering.
- [[reasoning-pathways]] — Phase 7 target built on sparse-feature decomposition.
- [[residual-stream]] — the activation site SAEs are trained on.
- [[board-state-probing]] — the ground-truth task used to evaluate chess SAEs.
