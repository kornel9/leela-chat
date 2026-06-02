# Understanding Skill Adaptation in Transformers Using Sparse Autoencoders: Chess as a Model System

**Summary**: SAEs trained on Maia-2's skill-aware residual stream identify threat-response features that, when intervened on, causally shift the model's effective skill level up or down. The work shows that skill-specific information in a transformer is concentrated in identifiable, manipulable features rather than diffusely distributed.

**Sources**: UNDERSTANDING SKILL ADAPTATION IN TRANSFORMERS USING SPARSE AUTOENCODERS.pdf

**Last updated**: 2026-06-02

---

## Citation
Anonymous authors (under double-blind review). "Understanding Skill Adaptation in Transformers Using Sparse Autoencoders: Chess as a Model System." arXiv 2024.

## Contribution
This paper applies Sparse Autoencoders to the internal residual stream representations of Maia-2, a skill-conditioned human move prediction model, to understand how skill information is encoded. SAEs are trained on Maia-2's two skill-aware transformer blocks, yielding 2048-dimensional feature dictionaries. The central finding is that specific SAE features correspond to threat-response concepts at the square level, and that additively intervening on those features causally shifts Maia-2's predicted move quality. The work demonstrates a mechanistic foothold for AI-powered skill coaching by showing that boosting a single feature in "transitional positions" raises best-move accuracy from 0% to over 50% at higher skill brackets. (source: UNDERSTANDING SKILL ADAPTATION IN TRANSFORMERS USING SPARSE AUTOENCODERS.pdf)

## Method
SAEs (ReLU, L1 sparsity penalty) are trained on mean-pooled residual streams from Maia-2's two upper transformer blocks using three months of Lichess rated games balanced across skill levels. (source: UNDERSTANDING SKILL ADAPTATION IN TRANSFORMERS USING SPARSE AUTOENCODERS.pdf, §4.1) Hyperparameter search covers SAE widths 1024–8192 and L1 coefficients 5e-6 to 1e-4, with selection guided by a combined Chess Concept AUC (CC-AUC) plus training-loss metric; the optimal configuration is dhid=2048, alpha=1e-5. A binary chess concept "square no longer attacked" is defined for each board square, and the highest-CC-AUC SAE feature per square is selected. Interventions additively inject that feature vector (scaled by strength s in 0.1–10) into Maia-2's last-layer residual stream; results are compared against random-feature interventions as a control. (source: UNDERSTANDING SKILL ADAPTATION IN TRANSFORMERS USING SPARSE AUTOENCODERS.pdf, §5.1–5.2)

## Key findings
- Maia-2 treats 27% of all board positions monotonically (skill-dependent transition from suboptimal to optimal), a stark contrast with 1% in non-skill-aware Maia-1 models. (source: UNDERSTANDING SKILL ADAPTATION IN TRANSFORMERS USING SPARSE AUTOENCODERS.pdf, §5.2.1)
- SAEs trained on Maia-2 residual streams achieve CC-AUC of 0.88–0.90 on square-specific threat concepts such as "capture possible on e3". (source: UNDERSTANDING SKILL ADAPTATION IN TRANSFORMERS USING SPARSE AUTOENCODERS.pdf, §5.1, Fig. 3)
- Targeted SAE feature intervention raises best-move accuracy from 0% to over 50% at skill bracket 8 in transitional positions, compared to near-0% for random-feature interventions. (source: UNDERSTANDING SKILL ADAPTATION IN TRANSFORMERS USING SPARSE AUTOENCODERS.pdf, §5.2.2, Fig. 4 left)
- Increasing intervention strength progressively lowers the transition-point Elo, with notable acceleration above strength=1, demonstrating a causal dose-response relationship. (source: UNDERSTANDING SKILL ADAPTATION IN TRANSFORMERS USING SPARSE AUTOENCODERS.pdf, §5.2.2, Fig. 4 right)
- The optimal SAE hyperparameters are dhid=2048 and L1 alpha=1e-5, selected by a composite CC-AUC + normalised training-loss metric over a grid search (1024–8192 width, 5e-6 to 1e-4 L1). (source: UNDERSTANDING SKILL ADAPTATION IN TRANSFORMERS USING SPARSE AUTOENCODERS.pdf, §5.1, Appendix Fig. 6)
- SAEs are trained only on the skill-aware transformer blocks (not the ResNet backbone), because the backbone is skill-agnostic and the transformer layers are where skill information is injected via query modulation. (source: UNDERSTANDING SKILL ADAPTATION IN TRANSFORMERS USING SPARSE AUTOENCODERS.pdf, §4.1)
- CC-AUC on candidate chess concepts provides a reliable early-stopping signal for SAE training, showing rapid initial increase followed by stabilisation. (source: UNDERSTANDING SKILL ADAPTATION IN TRANSFORMERS USING SPARSE AUTOENCODERS.pdf, §5.1, Fig. 2 left)
- SAE reconstruction preserves Maia-2's win-probability quantile distribution and move-prediction agreement across all skill levels, validating that the SAE bottleneck does not degrade downstream task fidelity. (source: UNDERSTANDING SKILL ADAPTATION IN TRANSFORMERS USING SPARSE AUTOENCODERS.pdf, §5.1, Fig. 2 right)

## Relevance to Leela Chat
**Phase 4 (Sparse Autoencoders) — extractor layer.** The CC-AUC metric and grid-search methodology (width 1024–8192, L1 5e-6 to 1e-4) for selecting SAE hyperparameters are directly adoptable for training SAEs on Lc0's BT4 residual stream. Using interpretable chess concepts as calibration signals rather than relying solely on reconstruction loss gives a concrete, phase-appropriate acceptance criterion for the SAE extractor. (source: UNDERSTANDING SKILL ADAPTATION IN TRANSFORMERS USING SPARSE AUTOENCODERS.pdf)

**Phase 3/4 interface — concept definition.** The "square no longer attacked" binary concept exemplifies how to construct training labels for concepts that depend on move consequences rather than static board-state properties. This is directly applicable when specifying Leela Chat's probe and SAE concept targets — particularly for tactical concepts (en prise, defender removal, fork setup) that are inherently consequence-dependent. (source: UNDERSTANDING SKILL ADAPTATION IN TRANSFORMERS USING SPARSE AUTOENCODERS.pdf, §4.1)

**Phase 6 (Differential/Counterfactual) — handler and translator layers.** The additive mediated intervention protocol — inject a scaled SAE feature vector into a single layer's residual stream and measure the downstream policy shift — is a concrete, tested procedure for Leela Chat's counterfactual handlers. The 0%-to-50% accuracy shift from a single feature addition confirms that SAE-based steering is potent enough to drive meaningful output changes, supporting the `why_is_my_move_wrong` handler's causal claims. (source: UNDERSTANDING SKILL ADAPTATION IN TRANSFORMERS USING SPARSE AUTOENCODERS.pdf, §5.2.2)

**Architectural caveat.** Maia-2 uses a ResNet backbone with skill-aware transformer blocks (query injection Q* = Q + (ea ⊕ eo)W*), which differs substantially from Lc0's BT4 (pure transformer with Smolgen attention). The SAE training and evaluation methodology transfers cleanly; direct architectural transfer of the skill-conditioning mechanism does not. Lc0's policy head is not a linear projection from the residual stream (Smolgen), so the same CC-AUC ceiling observed here may apply. (source: UNDERSTANDING SKILL ADAPTATION IN TRANSFORMERS USING SPARSE AUTOENCODERS.pdf)

## Related pages
- [[sparse-autoencoders]]
- [[maia-human-alignment]]
- [[feature-labeling]]
- [[activation-patching]]
- [[board-state-probing]]
- [[residual-stream]]
- [[measuring-progress-dictionary-learning]]
- [[maia-aligning-superhuman-ai]]
- [[acquisition-chess-knowledge-alphazero]]
- [[contrastive-sae-planning]]
- [[evidence-learned-look-ahead]]
