# Grandmaster-Level Chess Without Search

**Summary**: A 270M-parameter transformer trained via supervised learning on Stockfish-annotated action-values reaches Lichess blitz Elo 2895 against humans without any explicit search. The paper demonstrates that network-internal value signals, not search, are sufficient for grandmaster-level play.

**Sources**: Grandmaster-Level Chess Without Search.pdf

**Last updated**: 2026-06-02

---

## Citation
Ruoss et al. (DeepMind). "Grandmaster-Level Chess Without Search." arXiv 2024.  arXiv:2402.04494

## Contribution
The paper investigates whether a feed-forward transformer trained purely via supervised learning can match grandmaster-level chess play without explicit search or domain-specific heuristics. 10 million Lichess games are annotated with Stockfish 16 action-values (win percentages for every legal move), producing approximately 15 billion training points. A 270M-parameter decoder-only transformer is trained to predict these action-values from FEN-tokenized board states, reaching Lichess blitz Elo 2895 against humans and solving 69.4% of puzzle-test-set actions correctly. This establishes that rich value signals latent in a network's weights — not search — are the core driver of strong play. (source: Grandmaster-Level Chess Without Search.pdf)

## Method
Board states are encoded as fixed-length FEN strings (77 ASCII characters, each a token; 79-token context with action appended for action-value prediction). A decoder-only transformer with LLaMA-style improvements (post-norm, SwiGLU) outputs a discrete probability distribution over K=128 value bins representing Stockfish win percentages; cross-entropy loss is used throughout. Three predictor variants are compared: action-value (AV), state-value (SV), and behavioral cloning (BC). Policies are formed by evaluating the predictor over all legal actions and taking the argmax. No MCTS, no search, and no domain-specific chess heuristics are used at inference. Evaluation uses internal tournament Elo, Lichess blitz Elo against humans and bots, puzzle accuracy on 10k Lichess puzzles, action accuracy, and Kendall's τ rank correlation against Stockfish. (source: Grandmaster-Level Chess Without Search.pdf)

## Key findings
- A 270M-parameter transformer trained with supervised learning reaches Lichess blitz Elo 2895 against humans without any explicit search. (source: Grandmaster-Level Chess Without Search.pdf, Abstract)
- The model outperforms AlphaZero's policy network (Elo 1620), value network (Elo 1853), and GPT-3.5-turbo-instruct (Elo 1755) without MCTS. (source: Grandmaster-Level Chess Without Search.pdf, Table 1)
- Strong chess performance only arises at sufficient scale: both model size and dataset size must be large enough, as shown by scaling ablations over 10k, 100k, and 1M game training sets. (source: Grandmaster-Level Chess Without Search.pdf, §3.3, Fig. 3)
- Action-value prediction outperforms state-value prediction and behavioral cloning when the number of training games is matched; when data points are equalized the advantage disappears, suggesting the effect is primarily a data-quantity artifact. (source: Grandmaster-Level Chess Without Search.pdf, §3.4, Table A2, Fig. A6)
- The model is blind to threefold repetition because it observes only the current FEN with minimal history; a workaround sets the win percentage of the repetition-triggering move to 50%. (source: Grandmaster-Level Chess Without Search.pdf, §5)
- In winning positions where multiple moves map to the maximum value bin (100%), the policy becomes indecisive and may fail to finish the game, requiring a Stockfish consistency check when all top-5 moves exceed 99%. (source: Grandmaster-Level Chess Without Search.pdf, §5, Fig. 4)
- Board states are tokenized as 77-character fixed-length FEN strings; 1968 unique legal actions cover all games and are indexed as single tokens. (source: Grandmaster-Level Chess Without Search.pdf, §2.2 (Tokenization), Appendix A.1)
- Network depth saturates around 8 transformer layers when parameter count is held constant, indicating that depth matters but has diminishing returns. (source: Grandmaster-Level Chess Without Search.pdf, §3.4 (Network depth), Table 2)
- National-Master-level analysis describes the agent's style as aggressive and enterprising — material sacrifices for long-term strategic gain — and distinct from Stockfish. (source: Grandmaster-Level Chess Without Search.pdf, Appendix B.5)

## Relevance to Leela Chat
This paper is primarily a scaling and architecture baseline for Phase 1 and Phase 2. It demonstrates that a FEN-tokenized transformer can produce strong action-value estimates without search, validating that network-internal signals (win percentages, action rankings) extracted from Lc0 are meaningful and worth surfacing to users. (source: Grandmaster-Level Chess Without Search.pdf)

For Phase 1 extractors, the Stockfish-win-percentage-to-bin discretization scheme (K=128 bins, cross-entropy) mirrors how Leela's value head works and is directly relevant to how Leela Chat normalizes and presents evaluation numbers in the findings JSON passed to the translator layer. (source: Grandmaster-Level Chess Without Search.pdf)

The threefold-repetition and indecisiveness pitfalls described in §5 are concrete failure modes to test in Leela Chat's validator: the FEN must encode castling and en passant correctly, and high-confidence positions (all top-5 moves exceeding 99%) require special handling to avoid vacuous or misleading responses. These map directly to the validator layer's grounding checks. (source: Grandmaster-Level Chess Without Search.pdf, §5)

The playing-style analysis in Appendix B.5 — NM-level qualitative characterization of the agent as aggressive and enterprising — is a concrete precedent for the kind of commentary Leela Chat targets in the translator layer: qualitative, grounded in engine-generated signals, and distinct from raw engine output. This informs prompt design for the `why_this_move` and `what_is_leela_seeing` handlers. (source: Grandmaster-Level Chess Without Search.pdf, Appendix B.5)

## Related pages
- [[transformer-chess-architecture]]
- [[superhuman-ai]]
- [[mcts]]
- [[world-models]]
- [[board-state-probing]]
- [[learned-look-ahead]]
- [[alphazero-science]]
- [[acquisition-chess-knowledge-alphazero]]
- [[mastering-chess-transformer-monroe]]
- [[chessgpt]]
- [[chess-testbed-lm-state-tracking]]
