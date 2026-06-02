# Tracing the Thought of a Grandmaster-level Chess-Playing Transformer

**Summary**: First full-transformer sparse decomposition of Lc0/BT4 via Transcoders + Lorsa, building causally-verified reasoning pathways for chess moves. Constructs circuit graphs with sparse features as nodes and measured causal feature-to-feature effects as edges, then uses them to expose interpretable tactical considerations in a grandmaster-level transformer.

**Sources**: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf

**Last updated**: 2026-06-02

---

## Citation
Rui Lin, Zhenyu Jin, Guancheng Zhou, Xuyang Ge, Wentao Shu, Jiaxing Wu, Junxuan Wang, Zhengfu He, Junping Zhang, Xipeng Qiu. "Tracing the Thought of a Grandmaster-level Chess-Playing Transformer." arXiv preprint (Shanghai Innovation Institute / Fudan University); preprint dated April 14, 2026 2026.  (arXiv:2604.10158)

## Contribution
This paper introduces a sparse decomposition framework to interpret the internal computation of Leela Chess Zero (LC0), the BT4 transformer chess network. It replaces MLP modules with Transcoders and attention modules with Low-Rank Sparse Attention (Lorsa), yielding unified interpretable, feature-level sparse replacement layers for the entire transformer — claimed as the first work to decompose both MLP and attention modules of a transformer for interpretability. By combining these replacement layers with feature-steering causal interventions, the authors construct "reasoning pathways": full-model circuit graphs whose nodes are sparse features and whose edges are measured causal feature-to-feature effects. A detailed case study (defensive fixation) shows these pathways expose interpretable, empirically verifiable tactical considerations and can diagnose an over-evaluation of defensive dependencies that blinds the model to a forced mate. (source: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf)

## Method
Transcoders approximate each MLP with a Top-K sparse encoder (z = Top-K(W_E x + b_E)) and a linear decoder reconstruction (y = W_D z + b_D). Lorsa disentangles attention into rank-1 output-value heads with scalar per-token values, preserving the original bilinear attention and Smolgen module and decomposing activations into token-wise z-patterns. Both are trained on activations from the lichess standard rated dataset (positions as 64-token sequences over the 64 squares), sampling 800M tokens for training and 100M for analysis, with k=30 and an expansion factor of 16 (16,384 features per layer), an auxiliary-k loss to reduce dead features, and Lorsa W_O initialized within the original attention heads' output subspace. Features are interpreted via highest-activating squares and validated with rule-based statistical tests yielding per-feature precision/recall. Reasoning pathways are built (Algorithm 1) by feature steering — adding a feature's decoder direction to the residual stream scaled by alpha (alpha = beta = -1, equivalent to zero-ablation) — measuring feature-to-output effect (change in normalized P(move)) and feature-to-feature effect (relative change in downstream activation), then pruning nodes (top-200 by influence) and edges into a graph with semantically-grouped supernodes. (source: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf)

## Key findings

- First work to decompose BOTH the MLP and attention modules of a transformer with unsupervised sparse dictionary learning (Transcoders + Lorsa), enabling full-model circuit-level analysis. (source: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf, Abstract; §1 Introduction; Contributions)
- Reasoning pathways — full-model circuit graphs with sparse features as nodes and measured causal feature-to-feature effects as edges — are constructed for any input position and target move via feature steering (Algorithm 1). (source: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf, §3 Constructing Reasoning Pathways; Algorithm 1)
- Feature steering adds a feature's decoder direction to the residual stream scaled by alpha; the authors use alpha=beta=-1 (equivalent to zero-ablation), justified empirically as lying in the linear regime (mean |Pearson r| ~0.96). (source: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf, §3 (Eq. 5-7); Appendix F.1; Fig. 9-10)
- Bishop-control information for the h7 checkmate is transferred via three Lorsa features (layers 1, 7, 8) exhibiting cross-layer superposition; ablating any single one only weakly affects output, but ablating d3->h7 attention drops P(Qxh7+) from 28.75% to 23.35% and alpha=-2 steering collapses it to 0.7%. (source: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf, §4 Finding 1; Fig. 3B; Appendix H.1, Table 7)
- LC0 incorporates the opponent's rook defensive coverage (feature Lorsa.0.7083); copying its activation from g7 onto h7 makes the model perceive h7 as defended and reduces P(Qxh7+) from 28.75% to 9.32%. (source: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf, §4 Finding 2; Fig. 3C; Table 3)
- Position ambiguity is traced to over-evaluation of defensive dependencies (the g2 pawn): ablating the own-pawn-detection feature Tc.0.12663@g2 (100% precision) flips P(Qxh7+) from 28.8% to ~91%, a state the authors term "defensive fixation." (source: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf, §4 Finding 3; Table 4; Appendix I (Fig. 14))
- Transcoder and Lorsa features are more interpretable than raw MLP neurons (human activation-consistency 4.15 / 3.73 vs 2.05 on a 1-5 scale), and rule-based validation gives high precision (often near 100%) though some tactical features show low recall from feature splitting. (source: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf, Table 1; Table 2; §2.2; §7.1)
- Three metrics — path overlap, path cohesion, path coupling — show top-2-move reasoning pathways are highly parallel and use largely disjoint feature sets (overlap <15%, vs 0.17% baseline); cohesion (within-path) consistently exceeds coupling (cross-path). (source: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf, §5.1; Table 5; Fig. 4)
- Decision-critical information progressively converges onto a move's source and target squares in deeper layers (Move-Square Contribution Ratio nears 1 by layer 14), aligning with LC0's attention policy head where each move logit derives from its source/target square representations. (source: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf, §5.2; Eq. 11-12; Fig. 5)
- The studied network is BT4-1024x15x32h-swa-6147500: 15 transformer layers, hidden size 1024, 32 attention heads, input length 64 (the 8x8 board), grandmaster-level using only the policy network without search. (source: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf, Appendix A.1-A.2; footnote 2)
- Reasoning pathways recover and extend the learned look-ahead phenomenon (Jenner et al. 2024), giving a "lookahead-to-action" mechanistic account in mate-in-two sacrifice cases (e.g. Ng6+ via Lorsa.14.1078). (source: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf, §6 Related Work; §7 Discussion; Appendix I (Fig. 15-18))

## Relevance to Leela Chat

This is the project's North Star and the literal deliverable target for Phase 7: reproduce and productize this paper. It is built on the exact same network the project pins (BT4-1024x15x32h-swa-6147500), so its features, pathways, and metrics transfer directly rather than needing re-derivation. (source: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf)

**Phase 4 (Sparse Features):** The Transcoders (MLP replacement) and Lorsa (attention replacement) are the SAE/dictionary-learning approach to adopt. The exact configuration — k=30, expansion factor 16, 16,384 features per layer, lichess-rated training data, auxiliary-k loss — and the public code at github.com/JacklE0niden/Leela-SAEs are the go/no-go evaluation target named in CLAUDE.md's Phase 4 protocol. Faithfulness figures from Appendix C (L2 norm error ratio <0.3, explained variance >0.6 for most layers; Lorsas weaker in middle layers) set the floor for what "working" means before adopting the repo. (source: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf)

**Phase 5 (Feature Labeling):** The rule-based feature validation (per-feature precision/recall, Table 2) and the interpretation language from Appendix D map directly onto Phase 5 feature labeling. The project's hard requirement that every claim trace to an extracted number aligns with the paper's stance: every mechanistic claim is backed by a measurable steering intervention and rule-based precision/recall, not chess intuition. Feature IDs of the form Tc.layer.id@square and Lorsa.layer.id@square are addressable in the pinned BT4 and are natural first-class findings-JSON fields for the extractor layer. (source: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf)

**Phase 6 (Differential/Counterfactual):** Feature-steering causal interventions (add decoder direction scaled at alpha=-1, measure feature-to-output and feature-to-feature effects) are the exact differential/counterfactual primitive. The paper's Hydra effect / cross-layer superposition warning — ablating any single bishop-control feature only weakly affects output — is the failure mode CLAUDE.md's pitfalls flag; the project must replicate the multi-feature ablation strategy before drawing causal conclusions from single-feature interventions. (source: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf)

**Phase 7 (Reasoning Pathways):** Algorithm 1's pathway construction (significant-feature selection, edge construction via causal feature pairs, supernode grouping) plus the case studies (defensive fixation, lookahead-to-action) are the template for the pathway extractor. The LLM translator then narrates these pathways as sequential steps — validated against the project's Rule 1, since each step cites a specific Transcoder/Lorsa feature activation and measured P(move) delta rather than chess intuition. The parallel-reasoning metrics (path overlap, cohesion, coupling) provide a quantitative handle for when pathway narration is meaningful vs. when reasoning is too distributed to narrate cleanly. (source: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf)

**Limitations to track before Phase 7:** confined to single architecture (BT4 only); auto-interpretation of features via LLMs is unreliable in isolation (features are easier to interpret within a pathway than alone); conservative binary rule validation magnifies noise and feature splitting/absorption lowers recall; some dense features remain uninterpretable. Smolgen and bilinear attention are explicitly preserved in the Lorsa decomposition — consistent with CLAUDE.md's "Smolgen in BT4 is not vanilla attention" pitfall. (source: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf)

## Related pages
- [[reasoning-pathways]]
- [[sparse-autoencoders]]
- [[circuit-discovery]]
- [[activation-patching]]
- [[transformer-chess-architecture]]
- [[evidence-learned-look-ahead]]
- [[acquisition-chess-knowledge-alphazero]]
- [[contrastive-sae-interpreting]]
- [[sparse-feature-circuits]]
- [[mastering-chess-transformer-monroe]]
- [[grandmaster-chess-without-search]]
- [[hydra-effect]]
