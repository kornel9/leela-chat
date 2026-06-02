# Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models

**Summary**: Introduces supervised SAE evaluation metrics — coverage and board reconstruction — using chess and Othello models with formally specifiable ground-truth features, plus p-annealing, a cheaper training technique that matches Gated SAE quality.

**Sources**: Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models.pdf

**Last updated**: 2026-06-02

---

## Citation
Karvonen et al.. "Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models." NeurIPS 2024.  (arXiv:2408.00113)

## Contribution
This paper addresses the core evaluation problem for sparse autoencoders: without ground-truth features, researchers can only rely on proxy metrics like loss-recovered and L0, which may not reflect true interpretability. By leveraging chess and Othello language models — which have formally specifiable board-state features — the authors define two new supervised metrics: coverage (fraction of a board-state property catalog recovered by the SAE) and board reconstruction (ability to recover game state from active SAE features). They also open-source over 500 SAEs for chess and Othello, providing a concrete testbed for the field. The work confirms that standard proxy metrics miss real quality improvements, such as those from larger expansion factors, that the new metrics detect.

## Method
SAEs are trained on residual stream activations after layer 6 of an 8-layer GPT (512-dimensional hidden state) trained on chess PGN transcripts (16M Lichess games) and Othello transcripts (20M synthetic games). Four variants are compared: standard (L1 penalty), standard+p-annealing, Gated, and Gated+p-annealing. Board state properties (BSPs) are defined as binary functions over game boards: Gboard_state covers all 8×8×12 piece-location combinations; Gstrategy covers 15 strategic properties (check, pin, fork, castling rights, legal moves, threatened squares, etc.) drawn from McGrath et al. AlphaZero concepts. Coverage computes the max F1 score achieved by any SAE feature binarized at threshold t per BSP. Board reconstruction uses a high-precision (≥0.95) rule to predict the full board from active features and measures F1 against ground truth. P-annealing transitions the sparsity penalty from convex L1 (p=1) linearly to p=0.2 during training, using coefficient annealing to keep sparsity strength locally constant across p updates. Training used 300M tokens, expansion factors 8 and 16, on a single A100 GPU. (source: Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models.pdf)

## Key findings

- Chess and Othello LM settings provide formally specifiable ground-truth features, enabling supervised evaluation metrics for SAE quality that are more objective than proxy metrics. (source: Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models.pdf, Abstract)
- SAE features can accurately reconstruct game boards; SAEs on trained models achieve chess board reconstruction F1=0.85 vs. 0.01 for random-weight models. (source: Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models.pdf, Table 1)
- Linear probes outperform SAEs on board state recovery (0.98 vs 0.85 reconstruction F1), indicating SAEs do not capture all linearly-encoded board information. (source: Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models.pdf, §5)
- Standard SAEs trained with p-annealing achieve coverage on par with Gated SAEs while requiring 33% less compute (Gated SAEs require 50% more compute per forward pass). (source: Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models.pdf, §5)
- Larger expansion factors (8192 vs 4096) improve SAE quality — detectable by the new coverage and board reconstruction metrics but invisible to prior proxy metrics. (source: Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models.pdf, §5)
- P-annealing starts at p=1 (L1, convex) and decreases linearly to pend=0.2 during training; coefficient annealing keeps sparsity penalty strength locally constant across p updates. (source: Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models.pdf, §4.3)
- Strategic BSP coverage varies greatly by concept complexity: check=1.00, queen=0.96, castling=0.82, legal_moves=0.63, pin=0.33, fork=0.38 for the best SAEs. (source: Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models.pdf, Table 4)
- SAE features typically track properties relative to the player to move (mine/yours) rather than absolute color (white/black), consistent with Nanda et al. Othello findings. (source: Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models.pdf, §5)
- Coverage and board reconstruction metrics are optimal in the elbow region of the L0 vs. loss-recovered Pareto frontier, empirically validating the common wisdom about that region. (source: Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models.pdf, §5)

## Relevance to Leela Chat

This paper is directly relevant to Phase 4 (sparse autoencoders). It provides open-sourced chess SAEs (huggingface.co/adamkarvonen/chess_saes) and the evaluation framework to assess whether those SAEs have captured the board-state and strategic features Leela Chat needs to ground responses in the extractor layer. (source: Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models.pdf)

The coverage and board reconstruction metrics give a principled method for selecting which SAE layer and expansion factor to use against Lc0's BT4 residual stream. The finding that SAEs miss pin/fork coverage (reconstruction F1 of 0.20–0.38) means any extractor relying on SAE features for those concepts must be validated against these benchmarks before shipping, or [[linear-probes]] (Phase 3) should serve as the fallback for concepts with poor SAE coverage. (source: Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models.pdf)

The strategy BSP list (Table 3, drawn from [[acquisition-chess-knowledge-alphazero]]) is a direct starting checklist for Phase 3 probe concept selection — it enumerates exactly the kind of properties (check, pin, fork, castling, threatened squares, legal moves) that handlers such as `why_this_move` and `did_you_consider` need to cite. (source: Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models.pdf)

P-annealing is the recommended training method if custom SAEs must be trained for BT4; however, per the Phase 4 go/no-go rule, the existing open-source chess SAEs must be evaluated first before any training is attempted. Note that the evaluated model (8-layer GPT, 512-dim) is much smaller than BT4 (1024-wide, 15 blocks), so SAE weights do not transfer directly — only the BSP taxonomy and metric design are architecture-agnostic and immediately adoptable. (source: Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models.pdf)

## Related pages
- [[sparse-autoencoders]]
- [[board-state-probing]]
- [[world-models]]
- [[residual-stream]]
- [[linear-probes]]
- [[feature-labeling]]
- [[emergent-world-models-chess]]
- [[scaling-evaluating-saes]]
- [[contrastive-sae-planning]]
- [[acquisition-chess-knowledge-alphazero]]
- [[sparse-feature-circuits]]
