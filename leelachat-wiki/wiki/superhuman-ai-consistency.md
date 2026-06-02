# Superhuman AI Consistency (Fluri, Paleka, Tramèr — ETHZ)

**Summary**: Framework and code for evaluating superhuman models (chess AIs, LLMs) via logical consistency checks rather than ground-truth correctness. The chess module tests Leela Chess Zero for violations of forced-move invariance, board-transformation symmetry, position-mirroring, and recommended-move stability, and provides an MCTS tree parser and adversarial position generators directly applicable to Leela Chat.

**Sources**: superhuman-ai-consistency-main/README.md; docs/PRIOR_WORK.md L242-259

**Vendored at**: vendored snapshot; ref not recorded

**Verified**: README-derived, not run

**Last updated**: 2026-06-02

---

## What it is
Research project by Lukas Fluri, Daniel Paleka, and Florian Tramèr (ETH Zurich / ETH Privacy and Security group). Upstream: https://github.com/ethz-privsec/superhuman-ai-consistency. Accompanies arXiv 2306.09983, 'Evaluating Superhuman Models with Consistency Checks'. Vendored snapshot; ref not recorded. (source: superhuman-ai-consistency-main/README.md)

## What it provides
- `chess-ai-testing/rl_testing/mcts/tree_parser.py` (588 lines): parses Lc0 --verbose-move-stats token stream into TreeInfo/NodeInfo objects backed by networkx graphs; per-node attributes include N (visits), P (policy), WL (W-L), D (draw), M (moves-left), Q, U, S, V, T (is_terminal) — the primary Phase 2 MCTS parser target
- `chess-ai-testing/rl_testing/engine_generators/relaxed_uci_protocol.py`: ExtendedAnalysisResult + RelaxedUciProtocol subclassing python-chess UciProtocol to consume Lc0 tree-info tokens natively — the right boundary for leela_chat/engine/
- `chess-ai-testing/experiments/transformation_invariance_testing.py`: board-transformation invariance tests (rot90, rot180, rot270, flip_diag, flip_anti_diag, flip_hor, flip_vert, mirror) — mechanical ground-truth eval positions for Phase 0/Phase 8
- `chess-ai-testing/experiments/recommended_move_invariance_testing.py`: positions where the recommended move should be stable under controlled perturbations — adversarial eval-set fodder
- `chess-ai-testing/rl_testing/evolutionary_algorithms/`: evolutionary search framework (algorithms, crossovers, fitnesses, individuals, mutations, populations, selections) for finding positions where Leela disagrees with itself most strongly — goldmine for Phase 6 why_is_my_move_wrong eval positions
- `chess-ai-testing/data/create_forced_moves_database.py`: generator for forced-move positions (single legal move; positions must evaluate identically before and after)
- `chess-ai-testing/data/create_no_pawns_position_database.py` + `create_random_board_database.py`: additional position-database generators for transformation-invariance tests

## Key files & entry points
- `chess-ai-testing/rl_testing/mcts/tree_parser.py` — Core MCTS parser — 588 lines; parses --verbose-move-stats into networkx-backed TreeInfo/NodeInfo. The single highest-value file in this repo for Phase 2.
- `chess-ai-testing/rl_testing/engine_generators/relaxed_uci_protocol.py` — RelaxedUciProtocol + ExtendedAnalysisResult: subclasses python-chess UciProtocol to handle Lc0 tree-info token stream. Template for leela_chat/engine/ UCI bridge.
- `chess-ai-testing/rl_testing/evolutionary_algorithms/` — Full evolutionary algorithm framework for adversarial position discovery (Phase 6 eval positions).
- `chess-ai-testing/experiments/transformation_invariance_testing.py` — Board-rotation and reflection invariance tests — mechanical ground truth, directly usable as eval-set generators.
- `chess-ai-testing/data/create_forced_moves_database.py` — Forced-move position generator; positions have provable evaluation invariance.

## License & language
MIT (LICENSE present; LICENSE file present at repo root and in chess-ai-testing/. MIT confirmed in PRIOR_WORK.md.). Language: Python 3.8 (PRIOR_WORK notes 3.11 compat work needed; deps include python-chess, networkx, numpy, scipy, asyncssh, wandb).

## Relevance to Leela Chat
- **Phase(s):** 0, 2, 6, 8
- **Adopt:** Port tree_parser.py and relaxed_uci_protocol.py into leela_chat/engine/ for Phase 2 MCTS tree parsing — these are the primary reason this repo was vendored. Use transformation- and forced-move generators as Phase 0/8 adversarial eval positions with mechanical ground truth. Use the evolutionary algorithm for Phase 6 why_is_my_move_wrong eval positions. (source: docs/PRIOR_WORK.md L242-259)
- **Do not adopt / caveats:** Do not use the specific Lc0 positions or evals from their experiments — they used Lc0 release/0.29 and T-net 807785, neither of which is BT4; raw numbers do not transfer. Pinned old deps (chess 1.9.4, numpy 1.23) mean vendor-as-import-path is impractical; port the relevant modules. (source: docs/PRIOR_WORK.md L242-259)

Adopt/skip judgements summarise (source: docs/PRIOR_WORK.md L242-259) — PRIOR_WORK is canonical.

## Related pages
- [[mcts]]
- [[superhuman-ai]]
- [[activation-patching]]
- [[lc0]]
- [[python-chess]]
