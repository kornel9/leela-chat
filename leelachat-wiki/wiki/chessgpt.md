# ChessGPT: Bridging Policy Learning and Language Modeling

**Summary**: Large-scale chess dataset plus GPT/CLIP models that jointly learn policy and language, with a three-axis evaluation benchmark covering state tracking, value judgement, and policy ability. The nearest existing product to a chess LLM assistant, and the style reference for Leela Chat's Phase 8.

**Sources**: ChessGPT Bridging Policy Learning and Language Modeling.pdf

**Last updated**: 2026-06-02

---

## Citation
Feng et al. "ChessGPT: Bridging Policy Learning and Language Modeling." NeurIPS 2023 (Datasets and Benchmarks Track) 2023.  (arXiv:2306.09200)

## Contribution
ChessGPT proposes a pipeline for studying the interplay between policy learning and natural language by building the largest known chess game-and-language corpus (~42.8 GB, 28.1 M documents) covering game replays, language texts, mixed annotated games, and instruction/conversation data. From this corpus the authors train ChessCLIP (contrastive alignment of board states and moves with natural language) and ChessGPT (a fine-tuned RedPajama-3B causal language model). A comprehensive three-axis evaluation framework — chess modeling ability, value judgement, and policy ability — benchmarks both models against LLaMA-7B and RedPajama baselines and establishes a reusable template for evaluating chess language models. (source: ChessGPT Bridging Policy Learning and Language Modeling.pdf)

## Method
ChessCLIP applies CLIP-style contrastive learning between board/action encodings (R8×8×112 feature map + R1858 move vector, identical to Lc0/AlphaZero encoding) and a text encoder fine-tuned from OpenAI CLIP; training uses 245K annotated games with 1.3 M board-language pairs plus ~83K YouTube transcripts. ChessGPT fine-tunes RedPajama-3B (GPT-NeoX architecture) on the full corpus via causal language modelling to produce ChessGPT-Base, then applies supervised fine-tuning on instruction/conversation data to produce ChessGPT-Chat. Evaluation tasks span BigBench state tracking (UCI legal-move prediction), UCI/PGN-to-FEN conversion (Levenshtein similarity), state value multiple-choice (Stockfish depth-18 labels), annotation multiple-choice, opening recognition, checkmate-in-one, and Elo-conditioned move scoring. (source: ChessGPT Bridging Policy Learning and Language Modeling.pdf)

## Key findings
- The dataset totals 42.8 GB across 28.1 M documents, spanning game, language, mixed, and conversation subsets. (source: ChessGPT Bridging Policy Learning and Language Modeling.pdf, Table 9)
- ChessGPT-Base reaches 92–99% on BigBench chess state tracking across short/medium/long real and synthetic games, versus 23–53% for LLaMA-7B and RedPajama baselines. (source: ChessGPT Bridging Policy Learning and Language Modeling.pdf, Table 1)
- UCI-to-FEN and PGN-to-FEN Levenshtein similarity exceeds 85% for ChessGPT models versus below 10% for all baselines. (source: ChessGPT Bridging Policy Learning and Language Modeling.pdf, Table 2)
- Board/action encoding uses the same R8×8×112 feature map and R1858 move vector as Leela Chess Zero and AlphaZero. (source: ChessGPT Bridging Policy Learning and Language Modeling.pdf, §4.1)
- ChessGPT-Base achieves 71.4% exact-string match on checkmate-in-one with a hint suffix, versus 1.6% for LLaMA-7B. (source: ChessGPT Bridging Policy Learning and Language Modeling.pdf, Table 6)
- Elo-conditioned move scoring shows only minor variation across Elo bins (700–1000: 59.4%, 1700–2000: 59.6%, 2700–3000: 59.8%), indicating the model struggles to adapt play to skill level. (source: ChessGPT Bridging Policy Learning and Language Modeling.pdf, Table 8)
- ChessCLIP, trained only on annotated board-language pairs without raw game data, achieves 63.6% on chess annotation multiple-choice and 80.5% on PGN-to-opening tasks. (source: ChessGPT Bridging Policy Learning and Language Modeling.pdf, Table 4, Table 5)
- The annotated games dataset contains 245K games with 1.3 M board-language pairs collected from seven sources. (source: ChessGPT Bridging Policy Learning and Language Modeling.pdf, §3.3)
- ChessGPT-Chat exhibits slightly lower state tracking performance than ChessGPT-Base, suggesting a trade-off between instruction-following capability and game-state accuracy. (source: ChessGPT Bridging Policy Learning and Language Modeling.pdf, §5.1)

## Relevance to Leela Chat
ChessGPT is the nearest existing product to a chess LLM assistant and is explicitly cited in CLAUDE.md as the Phase 8 style reference. Its three-axis evaluation benchmark (modeling ability / value judgement / policy ability) is a structured template for Leela Chat's own eval harness — the axis decomposition maps cleanly onto Leela Chat's handler taxonomy (state handlers, evaluation handlers, move-selection handlers). (source: ChessGPT Bridging Policy Learning and Language Modeling.pdf)

The board encoding used (R8×8×112, R1858) is identical to Lc0/AlphaZero, confirming that Leela Chat's extractor layer can reuse the same positional representation for any CLIP-style alignment work relevant to Phase 5+ feature labeling. The annotated PGN corpus (245K games, 1.3 M board-language pairs) is a direct candidate dataset for evaluating Leela Chat's translation layer, though CLAUDE.md Rule 1 prohibits using it to let the LLM fill gaps not grounded in Lc0 internals — the dataset can inform eval design, not prompt knowledge. (source: ChessGPT Bridging Policy Learning and Language Modeling.pdf)

The finding that Elo-conditioned move scoring varies negligibly across skill bins is a cautionary data point for any personalisation feature in Leela Chat: a fine-tuned LM alone cannot reliably adapt to skill level; skill-conditioned behaviour must be grounded in Lc0 signals (policy distribution, visit share, network confidence) extracted by the handler layer, not implicitly learned by the translator. (source: ChessGPT Bridging Policy Learning and Language Modeling.pdf, Table 8)

## Related pages
- [[chess-commentary-generation]]
- [[transformer-chess-architecture]]
- [[world-models]]
- [[board-state-probing]]
- [[maia-human-alignment]]
- [[mcts]]
- [[chess-testbed-lm-state-tracking]]
- [[maia-aligning-superhuman-ai]]
- [[improving-chess-commentaries-symbolic]]
- [[acquisition-chess-knowledge-alphazero]]
- [[grandmaster-chess-without-search]]
