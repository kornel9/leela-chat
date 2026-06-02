# Acquisition of Chess Knowledge in AlphaZero

**Summary**: Linear probing of AlphaZero's ResNet shows human chess concepts (material, king safety, threats) emerge in a specific layer-and-training-step pattern during self-play, with a critical acquisition window around 25k–64k steps. This paper establishes which ~200 chess concepts are linearly decodable from intermediate activations, and where in the network they peak — the methodological template for Phase 3 concept probes in Leela Chat.

**Sources**: Acquisition of Chess Knowledge in AlphaZero.pdf

**Last updated**: 2026-06-02

---

## Citation
McGrath et al. (Thomas McGrath, Andrei Kapishnikov, Nenad Tomašev, Adam Pearce, Demis Hassabis, Been Kim, Ulrich Paquet, Vladimir Kramnik). "Acquisition of Chess Knowledge in AlphaZero." arXiv 2022.  arXiv:2111.09259

## Contribution
McGrath et al. probe AlphaZero's 20-block ResNet with sparse linear regression across every layer and training checkpoint to map when and where 200+ human chess concepts become linearly decodable. They show that many human-defined concepts — from material counts to king safety and mate threats — are accurately recoverable from intermediate activations, with a rapid knowledge-acquisition phase around 25k–64k training steps coinciding with the discovery of standard opening theory. Qualitative analysis by GM Kramnik confirms a training-step-ordered acquisition of tactical before positional skill. The paper establishes that superhuman self-play systems do develop human-interpretable internal representations, setting the foundation for deeper mechanistic investigation.

## Method
Sparse L1-regularised linear probes (linear regression for continuous concepts, logistic regression for binary ones) are trained from AlphaZero's 8×8×256-dimensional layer activations against 200+ human chess concepts: Stockfish 8's public evaluation sub-scores plus 116 custom boolean/integer features. Probes are trained at every network block (1–20) and every training checkpoint, producing "what-when-where" accuracy plots across layer depth and training time. Complementary unsupervised analysis employs non-negative matrix factorisation (NMF, K=36 factors per block) and direct activation-input covariance over 1.4 million grandmaster positions. A separate value-regression experiment uses a generalised linear model predicting AlphaZero's value head from Stockfish sub-concept scores to track how concept weights evolve over training.

## Key findings
- Human chess concepts are linearly decodable from AlphaZero's internal activations even though AlphaZero was trained entirely via self-play without exposure to human games. (source: Acquisition of Chess Knowledge in AlphaZero.pdf, Abstract)
- Stockfish 8 total score (total_t_ph) can be regressed with r² > 0.75 from layer-10 activations at 64k+ training steps. (source: Acquisition of Chess Knowledge in AlphaZero.pdf, §4.3.2, Fig. 2a)
- A critical knowledge-acquisition window occurs between roughly 25k and 64k training steps, during which material value, piece mobility, and standard opening moves are all rapidly acquired. (source: Acquisition of Chess Knowledge in AlphaZero.pdf, §6, Fig. 7)
- Material is the first concept to predict AlphaZero's value function; more complex concepts like king safety, threats, and mobility emerge only at ~32k+ steps. (source: Acquisition of Chess Knowledge in AlphaZero.pdf, §4.6, Fig. 4c)
- Probe accuracy for threats, material imbalance, and material peaks at mid-network layers and declines in later layers, indicating intermediate computations are either discarded or non-linearly re-encoded. (source: Acquisition of Chess Knowledge in AlphaZero.pdf, §4.3.2, Fig. 2d/g/h)
- AlphaZero's training trajectory narrows opening entropy (from 4.32 bits to ~2.8 bits), the opposite of five centuries of human play where first-move entropy increased from 0.33 bits to 1.87 bits. (source: Acquisition of Chess Knowledge in AlphaZero.pdf, §5.2, Fig. 5)
- NMF factors in early layers show move computations developing progressively across blocks: diagonal moves are only partially developed by block 1 and fully computed by block 3. (source: Acquisition of Chess Knowledge in AlphaZero.pdf, §7.1.2, Fig. 8a/b)
- has_mate_threat accuracy rises throughout the network, providing evidence that AlphaZero implements at least single-move look-ahead within the network weights rather than deferring entirely to MCTS. (source: Acquisition of Chess Knowledge in AlphaZero.pdf, §4.3.2, Fig. 2f)
- GM Kramnik confirmed tactical skill acquisition precedes positional skill acquisition; 64k vs 32k AlphaZero shows a larger Elo gap in tactical openings (365 points) than in positional openings (343 points). (source: Acquisition of Chess Knowledge in AlphaZero.pdf, §6.3)

## Relevance to Leela Chat
This paper is the **methodological template for Phase 3 (linear concept probes)**. It establishes which ~200 chess concepts are worth probing (material, imbalance, king_safety, mobility, threats, passed_pawns, space, forks, pins, contested files, etc.) and provides the sparse linear probe protocol (L1 regularisation, separate probe per layer and checkpoint). The concept list from Tables 2–4 should be adopted directly as the Phase 3 target set before any probe training begins, per CLAUDE.md §10.

At the extractor layer, the finding that concepts localise to specific network depths — and that some peak at mid-network layers and decline in later blocks — is a concrete design constraint: BT4 extractors must expose layer-selectable activation hooks, not just the final residual stream at block 20. Probing only block 15 of BT4 will miss concepts that are linearly available at earlier blocks.

At the handler layer, the concept taxonomy (material, king safety, threats, mobility) maps directly to the `what_is_leela_seeing` and `why_this_move` handlers' findings JSON schema, providing principled field names and a validated coverage target.

At the translator layer, the paper demonstrates that continuous concept scores (not just binary labels) are recoverable, which means the LLM can receive graded numerical inputs (e.g. imbalance score, king safety index) rather than only presence/absence flags — enabling more nuanced prose while staying grounded in extractor output.

**Note**: AlphaZero uses a 20-block ResNet (8×8×256), not BT4's transformer architecture. Concept and methodology findings transfer directly, but layer-depth conclusions (e.g. "peaks at block 10") do not map to BT4 block numbers without re-running the what-when-where analysis on BT4. (source: Acquisition of Chess Knowledge in AlphaZero.pdf)

## Related pages
- [[alphazero-science]]
- [[evidence-learned-look-ahead]]
- [[limits-of-probes]]
- [[bridging-human-ai-knowledge-gap]]
- [[alphazero-hex-evaluation]]
- [[linear-probes]]
- [[board-state-probing]]
- [[world-models]]
- [[learned-look-ahead]]
- [[residual-stream]]
- [[mcts]]
- [[superhuman-ai]]
