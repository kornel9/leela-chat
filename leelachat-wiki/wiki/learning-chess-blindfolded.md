# Learning Chess Blindfolded

**Summary**: A GPT-2-based language model trained on UCI-notation chess game sequences to evaluate world state tracking in language models. Introduces Randomized Auxiliary Prediction (RAP) — injecting board-state tokens (piece type at source/destination squares) during training as an auxiliary signal to improve move legality.

**Sources**: learning-chess-blindfolded-master/README.md

**Vendored at**: vendored snapshot; ref not recorded

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Research project by Shubham Toshniwal, Sam Wiseman, Karen Livescu, and Kevin Gimpel (TTI-Chicago / CMU / U Chicago). Upstream: https://github.com/shtoshni92/learning-chess-blindfolded. Accompanies arXiv 2102.13249 ('Learning Chess Blindfolded: Evaluating Language Models on State Tracking', 2021). Pretrained GPT-2 model released on HuggingFace as shtoshni/gpt2-chess-uci. (source: learning-chess-blindfolded-master/README.md)

## What it provides
- `src/data_processing/`: PGN-to-UCI pipeline (parse_pgn.py, filter_data.py, create_vocab_models.py), board-state extraction to numpy (get_piece_type_rap.py, get_fen_repr.py), and cloze-task generation (generate_cloze_eval_tasks.py)
- `src/main.py` + `src/lm_model.py`: training harness for GPT-2, RNN, Reformer, and Performer models on UCI move sequences with optional RAP (randomized auxiliary prediction) or oracle board-state conditioning
- `src/analysis/`: random-legal-move baseline script and error-analysis script classifying illegal moves into unreachable / syntactic / pseudo-legal / path-obstruction categories
- Cloze evaluation tasks: Ending-Square and Starting-Square prediction benchmarks derived from the MillionBase 2.5 PGN corpus
- Pretrained GPT-2-chess-uci checkpoint on HuggingFace (shtoshni/gpt2-chess-uci) usable via transformers

## Key files & entry points
- `src/main.py` — Entry point for training; controls model type, RAP flag, dataset size, context window
- `src/lm_model.py` — Model definition wrapping GPT-2/RNN/Reformer with optional RAP auxiliary heads
- `src/data_processing/parse_pgn.py` — PGN-to-UCI conversion; most reusable standalone utility
- `src/data_processing/get_piece_type_rap.py` — Extracts piece-type board-state as numpy arrays for RAP training
- `src/data_processing/get_fen_repr.py` — Extracts full FEN board-state representation as numpy arrays
- `src/analysis/error_analysis_end.py` — Classifies model move-prediction errors by legality category — useful as a probe-quality diagnostic template
- `requirements.txt` — Pins old stack: torch==1.7.1, transformers==4.2.2, pytorch-lightning==0.9.0, chess==1.3.0

## License & language
unknown (no LICENSE file in repo — verify before vendoring; no LICENSE file in repo — verify before vendoring). Language: Python 3 (tested with torch 1.7.1 / transformers 4.2.2 stack; not validated against Python 3.11).

## Relevance to Leela Chat
- **Phase(s):** 1, 3
- **Adopt:** The error-classification taxonomy (unreachable / syntactic / pseudo-legal / path-obstruction) and the cloze-task benchmark design are directly reusable as eval-set templates. The PGN-to-UCI pipeline (parse_pgn.py) is a clean reference. RAP's piece-type injection is a precedent for board-state conditioning that informs Phase 3 probe training design.
- **Do not adopt / caveats:** Do NOT import or reuse the model weights or training harness — Leela Chat is grounded in Lc0/BT4 network internals, not a separately trained language model. The old dependency stack (torch 1.7.1, transformers 4.2.2, pytorch-lightning 0.9.0) is incompatible with the project's pinned versions and must not be pulled in. This repo has no dedicated PRIOR_WORK.md entry — derived from README only.

## Related pages
- [[chess-testbed-lm-state-tracking]]
- [[world-models]]
- [[board-state-probing]]
- [[linear-probes]]
- [[emergent-world-models-chess]]
