# OpenSI-CoSMIC (Cognitive System of Machine Intelligent Computing)

**Summary**: Multi-service LLM router (OpenSI / University of Canberra) with five base services — chess, vector DB, RAG, Python code generation, and generic QA — dispatched by an LLM-based QueryAnalyser. Chess is Stockfish-only with chain-of-thought generation, making it a competitive-intelligence reference rather than an adoptable dependency.

**Sources**: CoSMIC-production/README.md; docs/PRIOR_WORK.md L879-888

**Vendored at**: vendored snapshot; ref not recorded

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Production system by OpenSI (Open Source Institute, University of Canberra); upstream at github.com/TheOpenSI/CoSMIC (production branch); accompanies the paper 'Unleashing Artificial Cognition: Integrating Multiple AI Systems', ACIS 2024, arXiv 2408.04910. (source: CoSMIC-production/README.md)

## What it provides
- LLM-based query routing (src/query_analyser/query_analyser.py) — classifies free-text queries to one of five services; structurally isomorphic to Leela Chat's CLASSIFIER→HANDLER design
- Per-LLM-family prompt parameterisation (src/maps.py, src/query_analyser/user_prompt.py) — prompt templates specialised by model family (Llama, GPT, Gemma, Mistral)
- Chess puzzle next-move prediction and analysis (src/modules/chess_qa_puzzle.py, 363 lines)
- FEN generation from move sequences (src/modules/chess_genfen.py)
- Chain-of-thought next-move generation (src/modules/chess_gencot.py) — the LLM-as-inventor pattern
- Vector database and RAG service (src/services/vector_database.py, src/services/rag.py)
- Docker-first deployment with Llama 3.1 via Ollama as the local LLM backend

## Key files & entry points
- `src/query_analyser/query_analyser.py` — LLM-based routing logic — 30-minute read for Phase 1 classifier design; structurally mirrors CLASSIFIER→HANDLER
- `src/maps.py` — Per-LLM-family prompt parameterisation
- `src/query_analyser/user_prompt.py` — User prompt templates by model family
- `src/modules/chess_qa_puzzle.py` — Chess puzzle QA (363 lines) — Stockfish-backed, competitive-intelligence reference
- `src/modules/chess_gencot.py` — Chain-of-thought move generation — canonical example of the LLM-as-inventor anti-pattern (CLAUDE.md §2 Rule 1)

## License & language
MIT (LICENSE present; Third-party model weights (Llama 3.1, Mistral, Gemma, GPT) carry their own licences as noted in README). Language: Python 3.8+.

## Relevance to Leela Chat
- **Phase(s):** 1
- **Adopt:** Read src/query_analyser/query_analyser.py + src/maps.py during Phase 1 classifier design for structural comparison to CLASSIFIER→HANDLER; useful competitive-intelligence read on an LLM-wrapped chess product. (source: docs/PRIOR_WORK.md L879-888)
- **Do not adopt / caveats:** No Leela, no network internals, no probes, no SAEs. Chess code is Stockfish-only and uses the LLM-as-inventor pattern (chain-of-thought generation) explicitly forbidden by CLAUDE.md §2 Rule 1. Not adoptable as a dependency. (source: docs/PRIOR_WORK.md L879-888)

Adopt/skip judgements summarise (source: docs/PRIOR_WORK.md L879-888) — PRIOR_WORK is canonical.

## Related pages
- [[stockfish]]
- [[python-chess]]
- [[chess-tutor]]
- [[chessgpt]]
