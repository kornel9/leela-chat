# Improving Chess Commentaries by Combining Language Models with Symbolic Reasoning Engines

**Summary**: Combines Leela Chess Zero as a symbolic oracle with a controllable BART language model via extracted "control tags" to generate grounded, human-preferred chess commentary.

**Sources**: mproving Chess Commentaries by Combining Language Models with Symbolic Reasoning Engines.pdf

**Last updated**: 2026-06-02

---

## Citation
Lee et al. "Improving Chess Commentaries by Combining Language Models with Symbolic Reasoning Engines." arXiv 2022.  (arXiv:2212.08195)

## Contribution
The paper presents a chess commentary generation system that decouples symbolic reasoning from language generation. A BART-based language model is trained conditioned on "control tags" (move quality, suggested moves, commentary type, pronouns/nouns, length) extracted from commentary training data. At inference, Leela Chess Zero supplies those tags — win probability before/after a move, best-move suggestion — so the language model never reasons about chess itself. The system is evaluated on 373,919 game-commentary triplets from gameknot.com, with human judges preferring the fully-conditioned model over prior baselines at rates of at least 72% across all commentary types.

## Method
Game state is represented three ways: PGN move history, enumerated board-state piece-square tokens, and attack-state (attacker–attacked pairs). Five tag types are extracted from training commentary using fine-tuned BART classifiers/generators: commentary type (6-class), move quality (5-class: excellent/good/inaccuracy/mistake/blunder), suggested moves (generative), pronouns/proper nouns (SpaCy NER), and length (short/medium/long heuristic). The commentary generation model is a 4M-parameter BART model trained to produce P(commentary | game, move, tags). At inference, Leela provides move-quality tags via win-probability differences (PM* − PM) and suggested-move tags directly. All models use the Adam optimizer trained on 8 V100 GPUs. (source: mproving Chess Commentaries by Combining Language Models with Symbolic Reasoning Engines.pdf)

## Key findings
- Symbolic reasoning engines (Leela) can be integrated with controllable language models via extracted control tags, relieving the LLM of any chess reasoning tasks while allowing super-human insights even when training data came from weaker players. (source: mproving Chess Commentaries by Combining Language Models with Symbolic Reasoning Engines.pdf, §1)
- Five control tag types are extracted from commentary training data — commentary type, move quality, suggested moves, pronouns/nouns, and length — enabling fine-grained controllability over the generation model. (source: mproving Chess Commentaries by Combining Language Models with Symbolic Reasoning Engines.pdf, §4.2.1)
- Move quality is derived from Leela by comparing PM* (win probability after best move) to PM (win probability after played move); the difference range maps to excellent/good/inaccuracy/mistake/blunder classifications. (source: mproving Chess Commentaries by Combining Language Models with Symbolic Reasoning Engines.pdf, §4.3)
- Human judges preferred the fully-conditioned model (Leela-tagged) over Jhamtani et al. (2018) at rates of at least 72% across all three commentary types evaluated. (source: mproving Chess Commentaries by Combining Language Models with Symbolic Reasoning Engines.pdf, Table 2)
- Perplexity drops monotonically as game-state representations (PGN, board, attacks) and tags are cumulatively added, demonstrating that each signal provides meaningful grounding for the language model. (source: mproving Chess Commentaries by Combining Language Models with Symbolic Reasoning Engines.pdf, Figure 2)
- Belief-state accuracy (whether the model's argmax over 64 squares matches the true square) is highest for a simple board-state-only representation (80.6%) and worsens slightly in the fully-conditioned model (72.6%), suggesting some input tokens are misused. (source: mproving Chess Commentaries by Combining Language Models with Symbolic Reasoning Engines.pdf, Table 5 / Appendix F)
- The model's belief-state concentrates probability mass on recently moved pieces, consistent with how commentary training data discusses recent moves disproportionately. (source: mproving Chess Commentaries by Combining Language Models with Symbolic Reasoning Engines.pdf, §6 / Appendix F)
- Crowdworkers prefer the system because of accuracy (36%) and strategy (14%) at much higher rates than when they prefer a baseline (15% and 7% respectively), confirming that Leela-derived tags drive the improvement. (source: mproving Chess Commentaries by Combining Language Models with Symbolic Reasoning Engines.pdf, §D.1)

## Relevance to Leela Chat
This is the closest prior work to Leela Chat's core architecture: both systems use a chess engine as a symbolic oracle whose outputs are fed to a language model that is forbidden from reasoning about chess itself. The control-tag pattern directly maps to Leela Chat's findings-JSON contract — the engine produces structured facts, the LLM translates them to prose. This mapping spans Phase 1 and beyond: the handler layer assembles the findings JSON (analogous to tag extraction), and the translation layer conditions the LLM on those facts exclusively (analogous to P(commentary | game, move, tags)).

The six commentary-type taxonomy (move description, quality, comparison, rationale, contextual, general) is a ready-made handler classification scheme applicable to the `what_is_leela_seeing`, `why_this_move`, and `did_you_consider` handlers in Phase 1, and provides a vocabulary for extending the classifier to new question types.

The belief-state probing technique — template-prompted conditional likelihood over all 64 squares — is a lightweight board-understanding diagnostic that Leela Chat's eval harness could adopt without requiring full interpretability infrastructure (Phase 3+). It tests whether the translator layer is tracking the board state implied by the findings JSON.

The paper's documented failure mode — the model contradicts game state when reasoning extends beyond the provided tags — is precisely the hallucination risk that Leela Chat's Rule 1 (LLM never invents chess facts) and the output validator are designed to prevent. The preference improvement driven by accuracy (36%) over stylistic factors is direct evidence that grounding quality is the dominant driver of user preference, supporting Leela Chat's design priority of correctness over fluency. (source: mproving Chess Commentaries by Combining Language Models with Symbolic Reasoning Engines.pdf)

## Related pages
- [[chess-commentary-generation]]
- [[chess-commentary-dataset]]
- [[learning-to-generate-move-commentary-jhamtani-2018]]
- [[mcts]]
- [[world-models]]
- [[board-state-probing]]
- [[superhuman-ai]]
- [[maia-aligning-superhuman-ai]]
- [[acquisition-chess-knowledge-alphazero]]
- [[chess-testbed-lm-state-tracking]]
- [[chessgpt]]
- [[alphazero-science]]
