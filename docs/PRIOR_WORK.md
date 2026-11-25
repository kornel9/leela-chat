# Prior Work — Conversational Interface to Leela Chess Zero

> **Purpose.** Complete inventory of every paper, GitHub repository, dataset, library, model, and product you can stand on. Read alongside `BIBLE.md` (the "why") and `CLAUDE.md` (the "how").
>
> **Verification standard.** Items marked ✓ were confirmed via direct repo or paper inspection. Unmarked entries are referenced from secondary sources; treat as unverified and check before integrating.
>
> **Maintenance.** When you adopt a dependency, copy its entry into `docs/DECISIONS.md` with date, version/hash, and a one-line rationale. When you reject one, write a one-line rejection. Don't delete entries — strike them through. The history matters for understanding why earlier decisions made sense.

---

## Table of Contents

- [TL;DR — the assembly](#tldr--the-assembly)
- [Tier 1 — Critical stack (must-have)](#tier-1--critical-stack-must-have)
- [Tier 2 — Major SAE and interpretability dependencies](#tier-2--major-sae-and-interpretability-dependencies)
- [Tier 3 — Mechanistic interpretability toolkits](#tier-3--mechanistic-interpretability-toolkits)
- [Tier 4 — Pre-trained models and weights](#tier-4--pre-trained-models-and-weights)
- [Tier 5 — Datasets](#tier-5--datasets)
- [Research — Chess interpretability (primary)](#research--chess-interpretability-primary)
- [Research — Chess networks and architecture](#research--chess-networks-and-architecture)
- [Research — General mechanistic interpretability (foundational)](#research--general-mechanistic-interpretability-foundational)
- [Research — Sparse autoencoders (methodology)](#research--sparse-autoencoders-methodology)
- [Research — Causal interventions, patching, counterfactuals](#research--causal-interventions-patching-counterfactuals)
- [Research — Concept probing methodology](#research--concept-probing-methodology)
- [Research — Chess commentary, NLP, LLM-as-translator](#research--chess-commentary-nlp-llm-as-translator)
- [Existing chess analysis products (competitive intelligence)](#existing-chess-analysis-products-competitive-intelligence)
- [Community resources and forums](#community-resources-and-forums)
- [What's NOT done — your work](#whats-not-done--your-work)
- [Phase-by-phase reading and adoption order](#phase-by-phase-reading-and-adoption-order)
- [License survey](#license-survey)
- [Open threads — re-search periodically](#open-threads--re-search-periodically)
- [Quick reference — where to find things](#quick-reference--where-to-find-things)

---

## TL;DR — the assembly

You will:

1. Use **lczerolens** to load Lc0 networks into PyTorch and hook them.
2. Use **python-chess** for boards, FEN, PGN, legal-move validation, and **Stockfish + Lc0 via UCI** for evaluations and MCTS access.
3. Use the **Fudan Leela-SAEs repo** for pre-trained Transcoders and Lorsas — your concept dictionary.
4. Use **OpenMOSS Language-Model-SAEs** if you need to train new SAEs on layers Fudan didn't cover.
5. Use **Jenner et al.'s look-ahead repo** for activation patching infrastructure.
6. Use **Lichess puzzle database + ChessGPT data + Lee et al. chess.com Q&A** for probe-training labels and commentary fine-tuning.
7. Wrap all of this with a question-router + frontier LLM (Claude / GPT-class) for the translation layer.

The only genuinely original engineering is the question-router, differential analysis, evaluation harness, output validator, MCTS tree extraction with eval trajectories, and conversational UX. The rest is glue between existing tools.

If you adopt nothing else, these seven items cut years off the project:

| Item | Type | Why critical |
|---|---|---|
| **lczerolens** | Library | Loads Lc0 as PyTorch with hooks already wired. Saves ~1 month. |
| **Lc0 (with `--verbose-move-stats`)** | Engine binary | The engine itself; verbose mode is what makes MCTS tree extraction tractable. |
| **python-chess** | Library | Board logic, FEN/PGN, UCI wrapper. Used in every layer. |
| **Fudan Leela-SAEs repo** | Pre-trained SAEs + code | If it reproduces, skips ~3 months of SAE training and labeling. |
| **Lichess open database** | Dataset | 7.4B+ games. Free training data for probes, SAEs, max-activation caches. |
| **Frontier LLM API** (Anthropic Claude / OpenAI) | Service | The translation layer. No fine-tuning needed in v1–v6. |
| **Fudan paper (arXiv 2604.10158)** | Research paper | Methodological blueprint for Phases 4, 6, and 7. |

---

## Tier 1 — Critical stack (must-have)

### lczerolens (Xmaster6y / Yoann Poupart) ✓

- **Repo:** `github.com/Xmaster6y/lczerolens`
- **PyPI:** `pip install "lczerolens[hf,viz,backends]"`
- **Docs:** lczerolens.readthedocs.io
- **License:** MIT (permissive, commercial use allowed)
- **Last release:** v0.4.0, late April 2026 (actively maintained)

**What it gives you:**
- One-line Lc0 → PyTorch loading: `LczeroModel.from_hf("lczerolens/maia-1100")`
- Direct ONNX path: convert any `.pb.gz` Lc0 net to PyTorch
- Board encoding (the 112-channel Lc0 input format — history planes correct)
- Move encoding/decoding for the 1858-policy-vector format
- Built-in concept probing notebook — your Phase 3 starter
- Saliency, LRP heatmaps via `LrpLens`, framework-agnostic interp (captum, zennit, nnsight, tdhook)
- Puzzle evaluation utilities
- HuggingFace Hub integration for pre-converted Lc0 models including Maia variants
- A Gradio demo space you can fork

**Where it fits in your project:**
- Phase 0: install, run the walkthrough notebook end-to-end. Your "is my environment working" smoke test.
- Phase 1: `LeelaInspector` wraps lczerolens for forward-pass extraction.
- Phase 2: hooks for residual stream + attention + MLP activation capture.
- Phase 3: the `Probe Concepts` notebook is essentially Phase 3 starter code.
- Phase 4: SAE input collection at any layer.
- Phase 6: activation patching for counterfactual analysis.
- Phase 7: reasoning pathway extraction primitives.

**Author note:** Yoann Poupart published "Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents" (arXiv 2406.04028) — research directly adjacent to your project. The library reflects deep knowledge of what's useful.

**Caveats:** Active development; pin a specific version in `pyproject.toml`. API may shift between minor versions.

### python-chess (niklasf) ✓

- **Repo:** `github.com/niklasf/python-chess`
- **PyPI:** `pip install python-chess`
- **Docs:** python-chess.readthedocs.io
- **License:** GPL-3 (copyleft — affects commercialization if you ever bundle source; inference-only deployment fine)

**What it gives you:**
- Board representation, FEN parsing, PGN reading/writing
- Move generation, legality checks (the foundation of your hallucination validator)
- SAN ↔ UCI ↔ coordinate notation conversion
- UCI protocol wrapper: `chess.engine.SimpleEngine.popen_uci("lc0")` and `popen_uci("stockfish")`
- `engine.analyse(board, limit, multipv=5)` returns scores, PVs, per-move info
- Polyglot opening book reading, Syzygy tablebase probing
- Async API available
- Mature: used by lichess-bot, picochess, and most chess Python tooling

**Where it fits:** every phase. This is the chess substrate.
- Phase 1: `engine.analyse(board, Limit(nodes=N), multipv=10)` against Lc0 gives policy + Q + PV per move in one call.
- Validator: `board.is_legal(move)` blocks LLM hallucination of impossible moves.

### Lc0 (Leela Chess Zero engine) ✓

- **Repo:** `github.com/LeelaChessZero/lc0`
- **License:** GPL-3
- **Latest:** v0.32.1 (Nov 2025)
- **Network downloads:** `lczero.org/play/networks/bestnets/` and `storage.lczero.org/files/networks-contrib/`

**Networks worth pinning:**
- **BT4-1024x15x32h-swa-6147500** — the network the Fudan paper analyzes. 15 layers, 1024-dim, 32 heads. Use this if you want their SAEs to load against it. **Recommended.**
- **T82-768x15x24h-swa-7464000** — used by Jenner et al. and the Iterative Inference paper. 768-dim. Use this if you want their interp tooling.
- Pick one. Pin it. Record SHA-256 in `docs/DECISIONS.md`. Switching invalidates every probe and SAE you train.

**Useful Lc0 flags:**
- `--verbose-move-stats` — per-move visit counts and Q-values. Your MCTS tree extraction primitive.
- `--show-wdl` — return win/draw/loss probabilities.
- `--multipv=N` in UCI mode — richer analysis output.
- `lc0 --help` for the full option list.

**Build:** clone, checkout the latest release tag, `./build.sh` with CUDA backend. macOS is finicky. WSL2 works but slower than native Linux. Don't compile with `gtest` unless you want an extra hour of build time.

### Stockfish ✓

- **Repo:** `github.com/official-stockfish/Stockfish`
- **License:** GPL-3
- **Install:** `sudo apt install stockfish` (Ubuntu) or build from source for latest.

**Why included in Tier 1:** the `compare_to_stockfish` handler (Phase 2) and as a baseline in eval. Disagreements between Leela and Stockfish are *findings* — surface them as "Leela sees X that Stockfish misses." Different evaluation philosophy (classical alpha-beta + NNUE vs. policy/value + MCTS) makes the comparison informative.

---

## Tier 2 — Major SAE and interpretability dependencies

### Leela-SAEs (Fudan / JacklE0niden)

- **Repo:** `github.com/JacklE0niden/Leela-SAEs` (verify — may rename; this is the most-cited path)
- **Paper:** Lin et al., "Tracing the Thought of a Grandmaster-level Chess-Playing Transformer," arXiv 2604.10158, April 2026
- **License:** check repo before integrating

**What it gives you (per the paper):**
- Trained Transcoders (sparse MLP replacements) for every layer of BT4
- Trained Lorsas (Low-Rank Sparse Attention replacements) for every layer of BT4
- Reasoning-pathway construction code (their Algorithm 1)
- A labeled, validated feature dictionary covering Detection, Source/Target, Value, Capture, Tactical, Spatial, Movement features
- Code to perform feature-steering interventions and compute feature-to-output / feature-to-feature effects
- "Defensive fixation on g2" case study reproduction

**Where it fits:**
- Phase 4 (replaces or augments your linear probes — their features are richer and validated)
- Phase 5 (skips the "train SAEs" step entirely if their net matches yours)
- Phase 7: the reasoning pathway output is directly the data your LLM translator wants — graph of features causally connected to the chosen move

**Critical evaluation step:** Week 1 of Phase 4, run their reproduction script on a sample position from their paper. Verify feature activations match. **Decide go/no-go before committing further.** If their code doesn't reproduce in 2 weeks, fall back to OpenMOSS or sparsify.

**Risk:** Repo may be sparse on documentation, recently published, and tied to a specific BT4 checkpoint. Plan for several days of "make their code run on my machine" before integrating.

### OpenMOSS Language-Model-SAEs

- **Repo:** `github.com/OpenMOSS/Language-Model-SAEs`
- **License:** check repo (Fudan group, generally permissive)

**What it gives you:**
- The underlying SAE/Transcoder/Lorsa training framework used by the Fudan paper
- Complete Replacement Models (CRMs) — transcoders + Lorsas combined
- Crosscoder support for tracking features across layers
- A unified web visualization interface for SAE features
- Active development, ICLR papers behind it

**Where it fits:**
- Phase 4 fallback: if Leela-SAEs doesn't work cleanly or you want to train on a different Lc0 net.
- Visualization: the web interface is useful for your own debugging.

**Compute:** 1–3 days on RTX 4090 per layer; faster on A100.

### EleutherAI sparsify

- **Repo:** `github.com/EleutherAI/sparsify`
- **License:** MIT ✓

**What it gives you:**
- Alternative SAE / transcoder training framework
- `Sae.load_from_hub()` for loading pretrained SAEs from HuggingFace Hub
- More mature than OpenMOSS in some respects, less chess-specific
- Apache 2.0 / MIT-style licensing (verify exact)

**Where it fits:** Tier 2 alternative to OpenMOSS for Phase 4. Don't use both — pick one. OpenMOSS is more chess-aligned; sparsify is more LLM-aligned.

### leela-interp (Jenner et al.) ⭐⭐

- **Paper:** arXiv 2406.00877 (NeurIPS 2024)
- **Repo:** `github.com/leela-interp/leela-interp` (also project page at `https://leela-interp.github.io/`)
- **License:** GPL-3 — same as our existing GPL-3 deps (Lc0, python-chess); subprocess/API isolation already part of our deployment model.
- **Python:** 3.10+ compatible; our 3.11 pin works.
- **Stack:** nnsight 0.2.\*, zarr, einops, onnx2torch, scikit-learn, custom python-chess fork (`ejnnr/python-chess`). The custom python-chess fork is worth noting — used for the 1858 move encoding.
- **What it gives you (concrete modules):**
  - `core/lc0.py` — Lc0 model wrapper
  - `core/leela_board.py` — 112-channel input encoding (the format your CLAUDE.md §8 pitfall list explicitly warns about reimplementing)
  - `core/nnsight.py` — nnsight integration for Lc0 (resolves the Tier 3 "investigate whether nnsight supports Leela" question; see Correction 3)
  - `core/uci_to_idx.py` — **the canonical 1858-element move vector mapping. CLAUDE.md §8 lists "Changing the 1858-element move vector mapping" as an explicit pitfall.** This file is the existing, working implementation. Vendor candidate.
  - `core/forward_pass_implementation.py` — Lc0 forward pass tied to nnsight hooks
  - `tools/activations.py` — activation capture
  - `tools/attention.py` — attention pattern utilities
  - `tools/patching.py` — activation patching (Phase 6 starter code)
  - `tools/probing.py` — linear probing on Leela (Phase 3 starter code)
  - `tools/piece_movement_heads.py` — named-attention-head analysis methodology (template for BT4-specific head taxonomy work in Phase 3 / Phase 6)
  - `scripts/probing.py`, `run_global_patching.py`, `run_single_head.py`, `make_puzzles.py`, `make_corruptions.py` — reproducer scripts
- **Puzzle dataset:** `interesting_puzzles.pkl` available at the FigShare link in the README. ~22k puzzles solvable by Leela but hard for weaker engines (same dataset Cruz's follow-up uses).
- **Model variant caveat:** the default `lc0.onnx` provided is the **history-stripped finetune** Cruz also uses, not stock BT4. Phase 6 patching against BT4 needs either a history-stripping finetune or careful experimental design that controls for history-plane noise. Already on the open-questions list.

### SAELens

- **Repo:** `github.com/jbloomAus/SAELens`
- **License:** MIT ✓

**What it gives you:** training and analyzing sparse autoencoders, strong tooling for LLM SAE work, generalizable to other transformers. Mature tooling for max-activation extraction.

**Where it fits:** alternative SAE training framework. Useful for max-activation cache infrastructure even if you train SAEs elsewhere.

### Anthropic SAE training code

- Search Anthropic's GitHub orgs (`anthropics/`) for the "Towards Monosemanticity" companion code.
- **What it provides:** Reference implementations of dictionary learning, feature visualization, evaluation metrics.
- **Where it fits:** Reference for what "good" SAE evaluation looks like.

### superhuman-ai-consistency / rl_testing (Fluri, Paleka, Tramèr — ETHZ) ⭐⭐

- **Paper:** *Evaluating Superhuman Models with Consistency Checks*, arXiv 2306.09983
- **Repo:** `github.com/ethz-privsec/superhuman-ai-consistency` (also: `github.com/luk-s/rl-testing-experiments` for the chess-only subset)
- **License:** MIT ✓
- **Stack:** Python 3.8 (compat work needed for our 3.11 pin), python-chess, asyncssh for remote engine, networkx, numpy, scipy, wandb
- **Subject engine:** Leela Chess Zero. Their experiments use Lc0 `release/0.29` and T-net `807785`; not BT4. Methodology and parser code transfer; specific positions don't.
- **What it provides — the two big-ticket items:**
  1. **`rl_testing/mcts/tree_parser.py`** (588 lines). Parses Lc0's `--verbose-move-stats` token stream into `TreeInfo` / `NodeInfo` Python objects backed by networkx graphs. Per-node attribute map: `N` (visits), `P` (policy), `WL` (W−L), `D` (draw), `M` (moves-left), `Q`, `U`, `S`, `V`, `T` (is_terminal). Token grammar: `TREE INFO` / `START TREE` / `END TREE` / `START NODE:` / `END NODE` / `POSITION:` / per-node and per-edge attribute lines. Exactly the data structure your Phase 2 chunk plan calls for.
  2. **`rl_testing/engine_generators/relaxed_uci_protocol.py`** — an `ExtendedAnalysisResult` and `RelaxedUciProtocol` that subclass python-chess's `UciProtocol` to consume Lc0's tree-info tokens directly, rather than scraping `info string` lines. This is the right boundary for our `leela_chat/engine/` module.
- **What it also provides — Phase 0 / Phase 8 adversarial eval-set generators:**
  - `experiments/recommended_move_invariance_testing.py` — positions where the recommended move should be stable under controlled perturbations.
  - `experiments/transformation_invariance_testing.py` — supports `rot90`, `rot180`, `rot270`, `flip_diag`, `flip_anti_diag`, `flip_hor`, `flip_vert`, `mirror`. Any chess AI must evaluate transformed boards identically; positions where Leela doesn't are objective eval-set fodder with mechanical ground truth.
  - `experiments/evolutionary_algorithms/evolutionary_algorithm_distributed_oracle_queries_async.py` — evolutionary search for positions where Leela disagrees with itself most strongly. A goldmine for Phase 6 `why_is_my_move_wrong` eval positions.
  - `data/create_forced_moves_database.py` — generator for positions with a single legal move (those positions must evaluate identically before and after the forced move).
- **Where it fits:** Phase 0 / Phase 8 (adversarial eval positions with mechanical ground truth), Phase 2 (MCTS parser — the time savings of this entire prior-work review), Phase 6 (evolutionary algorithm for `why_is_my_move_wrong` eval positions).
- **Caveats:** Python 3.8, Lc0 0.29, T-net 807785 — none are blockers. Pinned old deps (chess 1.9.4 + python-chess 1.999, numpy 1.23) push toward "port" rather than "vendor."
- **ADR:** ADR-0006 captures vendor-vs-port choice.

### lczero-planning (Poupart — ENS de Lyon) ⭐

- **Paper:** arXiv 2406.04028 — *Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents*
- **Repo:** `github.com/yp-edu/lczero-planning` (cite Poupart, 2024); project page `https://yp-edu.github.io/publications/lczero-planning`
- **License:** MIT ✓
- **Stack:** Python ≥3.9, poetry-managed. Uses `lczerolens` (Poupart's own library, pinned to the `patching` git rev), nnsight (Poupart's fork at `Xmaster6y/nnsight`, `modules` rev), pydantic 2.x ✓, loguru ✓, wandb. **Stack-aligned with this project's choices in CLAUDE.md §4 and §6.**
- **README warning:** *"Not a stable codebase."* Take seriously; it's a research repo, not a library.
- **What it gives you — the only Lc0-specific SAE training pipeline in the prior-work inventory:**
  - **`scripts/sae_training/train_regular.py`** + **`train_contrastive.py`** + **`train_buffer.py`** — regular and contrastive SAE training for Lc0.
  - **`scripts/sae_utils/sae.py`** + **`buffer_training.py`** — SAE model + activation-buffer training loop.
  - **`scripts/datasets/make_activations_datasets.py`** + **`make_features_dataset.py`** + **`make_trajectories_dataset.py`** + **`convert_tcec_dataset.py`** — full dataset pipeline including TCEC games.
  - **`scripts/sae_analysis/`** — analysis utilities.
  - **`scripts/play/raw_matchmaking.py`** — engine-vs-engine matchmaking with activation collection.
  - HuggingFace Spaces demo at `huggingface.co/spaces/lczero-planning/demo`.
- **Where it fits:**
  - **Phase 4 fallback path (major):** CLAUDE.md §10 Phase 4 currently says: *"if Fudan's code doesn't reproduce in 2 weeks, fall back to OpenMOSS or sparsify."* That fallback list should be reordered with `lczero-planning` as **Option A** because it's already Leela-shaped. OpenMOSS and sparsify both require porting work. ADR-0008 captures this.
  - **Phase 4 trajectory datasets:** if you ever want activation-collection over Lc0 game trajectories rather than single positions, `make_trajectories_dataset.py` is the closest existing reference.
  - **Phase 6:** the contrastive SAE framing (pairs of game trajectories) maps onto Phase 6 differential analysis — already noted in v2; lczero-planning is the concrete code.
- **Caveats:** poetry not uv; requires pinning translation to your `pyproject.toml`. The lczerolens "patching" branch and Poupart-fork nnsight "modules" branch are specific, possibly out-of-sync with mainline now. Verify branch availability before adopting.

### Tang et al. — Skill-Adaptation SAEs on Maia-2 ⭐

- **Title:** *Understanding Skill Adaptation in Transformers Using Sparse Autoencoders: Chess as a Model System*
- **Status:** Under review at ICLR 2024 (anonymous submission)
- **Code:** `https://anonymous.4open.science/r/chess-sae-3C06/` — check license on de-anonymization
- **Subject model:** Maia-2 (the unified skill-aware successor to Maia-1100/1300/etc. — ResNet backbone + skill-aware transformer with skill embeddings injected into attention queries).
- **What it does:** Trains SAEs on Maia-2's skill-modulated residual stream, identifies latent features that fire on interpretable chess concepts, performs mediated interventions on those SAE features to elicit higher- or lower-skill play in the same position.
- **Why it matters:** Closest existing experimental template for the coaching aspiration. Their "push a feature, model plays better/worse on the same board" is structurally the Phase 6 counterfactual experiment for `why_is_my_move_wrong` and "what would a 2400 player see here?" handlers.
- **Where it fits:**
  - **Phase 4 fallback option B:** if Fudan no-goes AND lczero-planning is also blocked, this paper's pipeline is the next-closest methodology (Maia-2, not BT4 — architecture differs). Use the *methodology*, not the weights.
  - **Phase 6:** mediated SAE intervention protocol is the template.
- **Caveats:** Maia-2 ≠ BT4 (ResNet+transformer hybrid vs. pure transformer with smolgen). Methodology transfers; weights and feature dictionary do not.

---

## Tier 3 — Mechanistic interpretability toolkits

General-purpose libraries you can adapt to Lc0.

### TransformerLens

- **Repo:** `github.com/TransformerLensOrg/TransformerLens`
- **License:** MIT ✓
- **What it does:** research-grade hook management for transformer models. Activation caching, intervention patching, attention pattern utilities. The de facto mech-interp library.
- **Where it fits:** investigate whether it supports Leela's architecture (smolgen attention is unusual). If yes, use it. If no, port its patterns to lczerolens-based code. Bible §3.3 calls this out: "may or may not support Leela directly — investigate."

### nnsight

- **Repo:** `github.com/ndif-team/nnsight`
- **License:** MIT ✓
- **What it does:** modern PyTorch model introspection and manipulation. Backed by NDIF (National Deep Inference Fabric). Clean intervention syntax: `with model.trace(input) as tracer:`.
- **Where it fits:** Phase 6 activation patching, counterfactual ablation. Cleaner API than raw PyTorch hooks for complex interventions.
- **Integration with Lc0:** **nnsight is the supported interp library for Lc0.** Working glue: `leela_interp.core.nnsight` (see `leela-interp` entry above). Use this pattern rather than reimplementing PyTorch hooks against `lczerolens.LczeroModel` directly. lczerolens supports nnsight as well, but `leela-interp`'s wrapper is the more battle-tested path through Jenner's published patching experiments.

### captum

- **Repo:** `github.com/pytorch/captum`
- **License:** BSD-3 ✓
- **What it does:** PyTorch's official interpretability library. Integrated gradients, layer attribution, neuron attribution.
- **Where it fits:** the `what_features_matter` handler (Phase 3) — gradient-based attribution of policy choice to probe directions.

### zennit

- **Repo:** `github.com/chr5tphr/zennit`
- **License:** LGPL
- **What it does:** attribution for PyTorch via Layer-wise Relevance Propagation (LRP) and related methods.
- **Where it fits:** alternative attribution method if captum's gradient approaches give noisy results.

### tdhook

- **Repo:** search GitHub for `tdhook` — a TensorDict-friendly hook framework.
- **Where it fits:** mentioned in bible as a hook utility. Optional.

### Neuronpedia

- **Site:** `neuronpedia.org`
- **What it does:** public database of SAE features for various LLMs with visualizations.
- **Where it fits:** **inspiration for your Phase 5 labeling UI.** Their interface design is exactly what your labeling tool should aspire to.

### alphasae (Pavan Yellow / pavanyellow)

- **Repo:** `github.com/pavanyellow/alphasae`
- **License:** Apache 2.0 ✓
- **Stack:** torch + numpy + tqdm + wandb (4 dependencies total)
- **What it does:** Trains a small AlphaZero on Othello 6x6, captures Layer-2 residual stream activations (3M examples), trains 25 SAEs across a 5×5 grid of L1 penalty × feature count `[1..5] × [256, 512, 1024, 2048, 4096]`. Includes a static HTML feature browser at `vis/feature_vis.html`. Training time: 1–30 min per SAE on an RTX 4090.
- **Why it's worth keeping:** *Not* a production dependency. Reference implementation only.
  - **Phase 4 pedagogical:** Read `sae/trainer.py` *before* fighting with OpenMOSS, Fudan, or lczero-planning. It's small enough to fit in your head.
  - **Phase 5 UX inspiration:** The static HTML feature browser is a working example of "click a feature, see top-activating positions" — minimum-viable version of the labeling tool.
- **What it does NOT do:** chess. Lc0. BT4. Transcoders. Lorsas. Not a Fudan substitute.

### alphatology (J. Forde — companion to Lovering Hex paper)

- **Repo:** `github.com/jzforde/alphatology` (cite Forde, 2022)
- **License:** MIT ✓
- **Subject domain:** AlphaZero on Hex, not chess.
- **What it gives you:** the actual code behind the Lovering Hex AlphaZero paper (see "Research — Concept probing methodology" entry). Modules:
  - `detect.py`, `encode.py`, `generate.py` — probe + behavioral-test pipeline.
  - `probing.py` — linear probe training.
  - `positive.py`, `negative.py` — generators for concept-positive and concept-negative example positions.
  - `templates/concepts.py`, `templates/positive.py`, `templates/negative.py`, `templates/neighbors.py`, `templates/probing.py` — concept-template generators (Hex-specific: bridge, edge, ladder, dead cells, etc.).
  - Slurm-job scaffolding (less relevant for RunPod).
- **Where it fits:** **Phase 3 reference read.** The Hex-specificity is real — none of the concept templates port to chess directly. But the pattern (positive examples + negative examples + behavioral test as a triplet) is exactly the methodology Phase 3 should follow. Read once during Phase 3 design to internalize the pattern; do not vendor.

---

## Tier 4 — Pre-trained models and weights

### Lc0 networks

- **Primary source:** `storage.lczero.org/files/networks-contrib/`
- **HuggingFace:** `huggingface.co/lczerolens` and similar orgs
- **Recommended pin:** `BT4-1024x15x32h-swa-6147500` (matches Fudan paper)
- **Alternative pin:** `T82-768x15x24h-swa-7464000` (matches Jenner / Iterative Inference)
- **Action:** download, hash with `sha256sum`, record in `docs/DECISIONS.md`. Don't change without re-validating everything downstream.

### Fudan Transcoders and Lorsas

- **Source:** check the Leela-SAEs repo for release links; may also appear on HuggingFace under Fudan group's account.
- **What's there:** pre-trained sparse autoencoders for BT4 layers + labeled feature dictionaries.
- **If available:** integrate directly. If not: train your own with OpenMOSS.

### Maia chess models

- **Repo:** `github.com/CSSLab/maia-chess`
- **Paper:** McIlroy-Young et al., "Aligning Superhuman AI with Human Behavior: Chess as a Model System" (KDD 2020)
- **HuggingFace:** `lczerolens/maia-1100`, `maia-1300`, `maia-1500`, `maia-1700`, `maia-1900`
- **License:** check repo (generally permissive)
- **What they are:** Lc0-compatible networks trained to *mimic human play at specific Elo levels*, not to be strong.
- **Where it fits:** differential analysis (Phase 6). "Maia-1500 would have played X here; you played X too; Leela hates it." Gold for coaching context. "Why doesn't a 1900 player see what BT4 sees here?" becomes a literal computation: diff feature firings between Maia-1900 and BT4 on the same position.
- **Note:** architecture differs from BT4 — separate forward-pass pipeline required, but lczerolens handles it.

### DeepMind 270M chess transformer

- **Paper:** Ruoss et al., "Grandmaster-Level Chess Without Search" (2024)
- **Code/model:** check `github.com/google-deepmind` for release
- **What it is:** 270M-parameter transformer reaching 2895 blitz Elo without any MCTS. Pure network play.
- **Where it fits:** validates that the network alone is hugely strong (supports v2 bible's features-first thesis). Possible comparison baseline ("how does Lc0's network alone compare to this purpose-built one?").

---

## Tier 5 — Datasets

### Lichess open database ✓

- **URL:** `database.lichess.org`
- **Size:** 7.4B+ rated standard games, monthly PGN dumps, ~200 GB compressed per recent month
- **License:** CC0 (public domain) ✓
- **Engine evals:** included on ~6% of games

**Usage:**
- Probe training (Phase 3): sample positions from games
- SAE training data (Phase 4): the Fudan paper trained on 800M tokens from here — you can use the same source
- Max-activation cache (Phase 4): 1M+ positions to find feature firings
- Eval set augmentation (Phase 8): adversarial positions from real games
- Similarity database (Phase 2): 1M+ positions with embeddings for nearest-neighbor

**Tip:** use `pgn-extract` or `python-chess`'s PGN reader to filter by rating bucket, time control, result.

### Lichess puzzle database ✓

- **URL:** `database.lichess.org/#puzzles`
- **License:** CC0 ✓
- **What's there:** ~3.5M puzzles with FEN, solution, themes (tactical motifs), rating, popularity
- **Schema:** CSV columns — PuzzleId, FEN, Moves, Rating, RatingDeviation, Popularity, NbPlays, Themes, GameUrl, OpeningTags

**Usage:** **free labels for tactical concepts.** Themes like `fork`, `pin`, `skewer`, `mateIn2`, `discoveredAttack` directly populate your Phase 3 probe training sets.

### ChessGPT dataset

- **Paper:** Feng et al., "ChessGPT: Bridging Policy Learning and Language Modeling," arXiv 2306.09200 (NeurIPS 2023)
- **Components:**
  - CCRL: 3M computer engine games (Elo 2800–3700)
  - 3.2M Lichess puzzles with theme tags
  - Annotated PGNs (commentary)
  - YouTube chess transcripts
  - Pro-player datasets
  - Chess books
  - Online chess forums

**Usage:**
- The annotated PGNs are gold for training your LLM translator (Phase 1 polish).
- The YouTube transcripts capture how strong players actually *talk* about chess — invaluable for tone tuning.
- Puzzle data labels Phase 3 probes for tactical motifs.

### Jhamtani et al. chess commentary corpus

- **Paper:** "Learning to Generate Move-by-Move Commentary for Chess Games from Hundreds of Thousands of Examples" (ACL 2018)
- **Data:** 300k move-commentary pairs across 11k games
- **Where to find:** check the paper's supplementary, GitHub link in the paper.

**Usage:** older but cleaner than scraped corpora. Useful for Phase 1 prompt examples and few-shot LLM context.

### Lee et al. chess.com Q&A dialogues ⭐

- **Paper:** Lee et al., "Improving Chess Commentaries by Combining Language Models with Symbolic Reasoning Engines," arXiv 2212.08195 (2022)
- **Data:** 32k chess.com Q&A dialogues paired with game state in PGN

**Why starred:** **this is the closest existing public dataset to "user asks question about position, expert responds."** Read the paper carefully — it's a partial spec of your product. Use the data for prompt few-shot examples and for understanding what kinds of questions players actually ask.

### Syzygy tablebases

- **Source:** `tablebase.lichess.ovh` (online) or `chess.cygnitec.com` (download)
- **What's there:** perfect-play data for all positions with ≤7 pieces
- **Usage:** ground truth for endgame probe labels. "Is this position theoretically winning?" answered exactly.

### GM/IM game databases

- **Source:** TWIC (`theweekinchess.com`), chessdb.cn dumps, ChessBase Mega Database (commercial)
- **Usage:** higher-quality positions for eval set than random Lichess games. Good source of "real GM-level decisions" for your alpha test.

### Computer chess opening books (Polyglot format)

- **Source:** various; cerebellum, gm2600, etc.
- **Usage:** optional. Filtering "in-book" positions out of eval set or generating opening-position samples.

### Computer chess rating lists (CCRL, CEGT)

- **Usage:** sanity-check your understanding of where Lc0, Stockfish, and the 270M transformer sit relative to each other.

---

## Research — Chess interpretability (primary)

The papers you build directly on. Read all of them before Phase 4.

### Lin et al. — "Tracing the Thought of a Grandmaster-level Chess-Playing Transformer" ⭐⭐⭐

- **arXiv:** 2604.10158 (April 2026)
- **Authors:** Rui Lin, Zhenyu Jin, Guancheng Zhou, Xuyang Ge, Wentao Shu, Jiaxing Wu, Junxuan Wang, Zhengfu He, Junping Zhang, Xipeng Qiu — Shanghai Innovation Institute / Fudan / Xi'an Jiaotong
- **What it does:** introduces a sparse decomposition framework for Lc0. Transcoders (replacing MLPs) + Lorsas (Low-Rank Sparse Attention, replacing attention). Constructs reasoning pathways via feature-steering causal interventions. Demonstrates "defensive fixation" case study.
- **Why critical:** your Phases 4, 6, and 7 are "build a product on top of this paper." Read it twice. Annotate it. Reproduce the case study before declaring Phase 7 done.
- **Code:** the JacklE0niden/Leela-SAEs repo (verify).
- **Key concepts to internalize:** Transcoders vs SAEs, Lorsas vs standard SAEs (sparse attention decomposition), feature-steering, reasoning pathways as causal graphs, supernode grouping for visualization.

### Davis, Vasquez Ferrer, Sukthankar — "Exploring the Limits of Probes for Latent Representation Edits in GPT Models" ⭐

- **Venue:** MDPI AI Vol 7, 92 (2026), extension of ICMLA 2024
- **Authors:** Austin L. Davis, Robinson Vasquez Ferrer, Gita Sukthankar (UCF)
- **What it does:** trains a 12-layer chess GPT on UCI sequences. Probes the residual stream to read AND edit board state. Compares linear probes vs SAE latents for intervention.
- **Key finding:** **linear probes achieved 88% legal-move success after intervention; SAEs only 41%.** Strong evidence that human-concept probes (Phase 3) are worth doing even when you have SAEs (Phase 4). **Don't skip Phase 3.**
- **Methodology to borrow:** their Markovian-residual-stream test (same board state → same activations regardless of move history) is a clean sanity check for your forward-pass code.
- **Caveat:** their model is a custom GPT on UCI strings, not Lc0. The *engineering* doesn't transfer — only the methodology and findings.

### McGrath, Kapishnikov, Tomašev et al. — "Acquisition of Chess Knowledge in AlphaZero" ⭐⭐

- **arXiv:** 2111.09259, **Venue:** PNAS 2022
- **Authors:** Tom McGrath, Andrei Kapishnikov, Nenad Tomašev, Adam Pearce, Martin Wattenberg, Demis Hassabis, Been Kim, Ulrich Paquet, Vladimir Kramnik
- **What it does:** trains linear probes on AlphaZero's residual stream for ~150 named chess concepts (drawn from Stockfish's evaluation features). Shows when each concept emerges during training. Visualizes concept evolution.
- **Why critical:** your Phase 3 is "apply this methodology to Lc0 with your own concept list." The paper's concept taxonomy is a strong starting point.
- **Reading priority:** first chess-interp paper to read.
- **Note:** DeepMind didn't release AlphaZero weights, but the *methodology* applies directly to Lc0 (the AlphaZero recipe rebuilt openly).

### Jenner et al. — "Evidence of Learned Look-Ahead in a Chess-Playing Neural Network" ⭐

- **arXiv:** 2406.00877 (2024), **Venue:** NeurIPS 2024
- **Authors:** Erik Jenner et al.
- **What it does:** shows Lc0 (without MCTS) internally represents future moves — evidence of "look-ahead" purely in the network. Uses activation patching to demonstrate this.
- **Why critical:**
  - Validates that there's real internal structure to interpret.
  - Open-sources activation patching infrastructure for Lc0 you can adapt.
  - Suggests question types: "what does Leela think will happen in 3 moves?"

### Schut et al. — "Bridging the Human-AI Knowledge Gap: Concept Discovery and Transfer in AlphaZero"

- **Authors:** Lisa Schut, Nenad Tomašev, Tom McGrath, et al. (DeepMind)
- **Venue:** PNAS 2025 (referenced in the Fudan paper)
- **What it does:** identifies chess concepts AlphaZero uses that humans don't have names for, then teaches them to human players. Chess GMs as study participants.
- **Why critical:** this is the *philosophical* paper behind your project, made academic. Provides intellectual cover for the "discover alien concepts" framing and gives a methodology template: identify, name, validate via teaching.

### Karvonen et al. — "Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models"

- **arXiv:** 2403.15498 (2024)
- **Author:** Adam Karvonen et al.
- **What it does:** probes a chess LLM (trained on PGN) for internal board state representation. Shows the model maintains a coherent internal board from move sequences alone.
- **Why relevant:** methodology for "is the world model real?" probing. Cross-references with the Davis MDPI paper.

### Karvonen et al. — "Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models"

- **arXiv:** 2411.18895, **Venue:** NeurIPS 2024
- **What it does:** SAE evaluation methodology on chess (and Othello) models. Shows board-state properties (piece locations) are recoverable from SAE features at >99% accuracy.
- **Why critical:** methodological template for evaluating your own (or Fudan's) SAE features. Read their eval design before judging Phase 4 success.
- **Companion code:** `github.com/adamkarvonen/SAE_BoardGameEval` (no LICENSE file present in repo — check before vendoring). Python 3.11 ✓, Pydantic 2.x ✓, nnsight 0.3.7, transformer_lens 1.17.0.
- **Targets ChessGPT and OthelloGPT, not Lc0.** Methodology transfers; the actual hookups don't.
- **Reusable pieces for Phase 4:**
  - `circuits/eval_sae_as_classifier.py` — supervised SAE quality metric (the paper's contribution).
  - `circuits/eval_board_reconstruction.py` — board-state recovery from SAE features.
  - `circuits/full_pipeline.py` + `circuits/full_pipeline_walkthrough.ipynb` — end-to-end SAE eval pipeline.
  - `circuits/sae_training/` — p-annealing training (the paper's other contribution).
  - **Pretrained chess SAEs on HuggingFace** (`adamkarvonen/chess_saes`, including `chess-trained_model-layer_5-2024-05-23.zip`). Useful as comparison baselines even though they're on ChessGPT, not BT4 — you can run the same probe questions against both and compare quality scores.
- **Where it fits:** Phase 4 evaluation methodology reference. Adopt the eval design, not the code wholesale.

### Toshniwal et al. — "Chess as a Testbed for Language Model State Tracking"

- **Venue:** EMNLP 2022
- **What it does:** earlier work on whether LLMs trained on PGN track board state. Predecessor to the Karvonen and Davis papers.
- **Reading priority:** lower; read after the others for historical context.

### "Iterative Inference" paper

- **arXiv:** 2508.21380 (2025)
- **What it does:** extends logit lens to Lc0's policy network. Shows the network's top move *changes* across layers, non-monotonically.
- **Why critical:** directly informs your `policy_layer_trajectory` extractor (Phase 2). Their methodology IS your handler. "Watch Leela's mind change" is a feature you can build directly from this.

### Poupart — "Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents" ⭐

- **arXiv:** 2406.04028
- **Author:** Yoann Poupart (the lczerolens author)
- **Why included:** the maintainer of your foundation library has a chess SAE paper. Read it for methodology and for understanding what lczerolens was built to support.
- **Companion code:** `github.com/yp-edu/lczero-planning` (see separate Tier 2 entry below). The paper and the repo together are the closest existing model for the Phase 4 / Phase 6 work — Lc0-specific SAE training plus the paired-trajectory framing that maps onto your Phase 6 differential analysis.

### Cruz — "Understanding the learned look-ahead behavior of chess neural networks" ⭐

- **arXiv:** 2505.21552 (May 2025)
- **Author:** Diogo Cruz (Pivotal)
- **What it does:** Direct follow-up to Jenner et al. Extends look-ahead analysis from move 3 to moves 5 and 7. Three findings:
  1. Look-ahead behavior is **highly position-dependent**.
  2. The network considers **multiple move sequences simultaneously**, not just the principal variation.
  3. Specific attention heads respond preferentially to longer-term future moves.
- **Why it matters:**
  - **Phase 6:** Settles a design question for the `did_you_consider` handler — the network *does* hold multiple lines simultaneously. Extracting alternate-line activations is real signal.
  - **Phase 7:** Empirical support for the Fudan reasoning-pathway framing.
- **Model-variant note:** Cruz uses Jenner's history-stripped Leela finetune, same as `leela-interp`. Not BT4.
- **Feasibility signal:** Whole study ran on **RTX 3070Ti in 2 days total** — Phase 6 patching does not require A100 time.

### Davis & Sukthankar — "Decoding Chess Mastery"

- **Authors:** Davis & Sukthankar (UCF) — same authors as the Davis MDPI paper already in PRIOR_WORK.md.
- **Subject model:** Toshniwal et al.'s LCB (GPT-2 small, 12L × 12H). **Not Lc0.**
- **What it does:** Direct Logit Attribution per layer — finds early layers select moves the final layer ranks highly. Attention-pattern taxonomy.
- **Where it fits:** Phase 2 (or late Phase 1) — methodological reference for a `policy_layer_trajectory` extractor. Phase 3 — attention-pattern taxonomy as priors (though derived on GPT-2 chess, not BT4).
- **Caveats:** Smaller, weaker model. Methodology over specific findings.

### Anthropic / DeepMind chess research blog posts

- Watch for any Anthropic blog post on chess or game-playing interpretability; DeepMind's AlphaZero retrospective posts.
- Often contain implementation tips not in papers.

---

## Research — Chess networks and architecture

You need to understand the architecture you're poking at.

### Monroe (Lc0 team) — "Mastering Chess with a Transformer Model" ⭐⭐

- **arXiv:** 2409.12272
- **Author:** Daniel Monroe (Lc0 team)
- **What it does:** describes BT3/BT4 architecture: smolgen attention enhancement, attention policy head, value head structure.
- **Why critical:** the architecture is **not** a vanilla transformer. Smolgen and the attention-policy head are custom. Get hooks in the wrong place → silently corrupted activations. Required reading before serious hook engineering.

### Silver et al. — AlphaZero papers

- "Mastering Chess and Shogi by Self-Play..." (2017), "A general reinforcement learning algorithm..." (Science 2018)
- **Why useful:** foundational. MCTS + policy/value network architecture that Lc0 inherits.
- **Reading priority:** skim. You don't need to implement AlphaZero; you need to interpret a descendant.

### Ruoss et al. — "Grandmaster-Level Chess Without Search"

- **DeepMind 2024**
- 270M-parameter transformer reaches 2895 blitz Elo without any MCTS.
- **Why useful:** validates the network-alone strength thesis. Comparison baseline if released.

### MuZero, EfficientZero, Stockfish NNUE papers

- Adjacent architectures, useful context.
- **Reading priority:** low. Read only for broad context.

### Stockfish NNUE documentation

- Stockfish wiki and NNUE-related papers.
- **Why useful:** different evaluation paradigm. Understanding it helps you explain Stockfish-Leela disagreements in your `compare_to_stockfish` handler.

---

## Research — General mechanistic interpretability (foundational)

Read these before Phase 4. They're the conceptual scaffolding for SAEs and circuits.

### Olah et al. — Circuits thread (Distill 2020+)

- **Site:** `distill.pub` (Circuits articles)
- Foundational essays. Defines neurons, circuits, polysemanticity, superposition.
- **Reading priority:** "Zoom In: An Introduction to Circuits" is mandatory background.

### Elhage et al. — "Toy Models of Superposition" (Anthropic)

- **Site:** `transformer-circuits.pub`
- Establishes the theoretical basis for why neurons are polysemantic and why SAEs are needed.
- **Why critical:** without this, "we need SAEs because of superposition" is just words. Read for intuition.

### Anthropic — "Towards Monosemanticity" ⭐

- **Site:** `transformer-circuits.pub` (October 2023)
- First convincing demonstration that SAEs find interpretable features in a real LLM (one-layer transformer).
- **Why critical:** your Phase 4 mental model. Their feature visualization approach informs your Phase 5 labeling tool.

### Anthropic — "Scaling Monosemanticity"

- **Site:** `transformer-circuits.pub` (May 2024)
- Scales SAE methodology to Claude 3 Sonnet. Discusses feature splitting, computational requirements, evaluation.
- **Why critical:** practical lessons on dictionary size, training compute, what to expect at scale.

### Cunningham et al. — "Sparse Autoencoders Find Highly Interpretable Features in Language Models"

- **arXiv:** 2309.08600 (EleutherAI 2023)
- Independent confirmation of SAE methodology on LLMs. Open-source implementation.
- **Why useful:** alternative reference implementation. Read alongside Anthropic for methodological triangulation.

### Templeton et al. — Anthropic Circuits monthly updates

- **Site:** `transformer-circuits.pub` (monthly)
- Ongoing publications on SAE features, attribution, circuits.
- **Reading priority:** skim each update for chess-relevant techniques.

### Bricken et al. — "Decomposing Language Models With Dictionary Learning"

- **Source:** Anthropic / transformer-circuits.pub
- Detailed SAE training methodology, including dead feature problems and resampling.
- **Why useful:** practical advice. Read before training your own SAEs.

### McGrath et al. — "The Hydra Effect" ⭐

- **arXiv:** 2307.15771 (2023)
- **Authors:** Tom McGrath et al. (DeepMind)
- Shows that when you ablate one network component, others compensate. Naive single-component ablations underestimate importance.
- **Why critical:** Phase 6 lives or dies by this. Plan for multi-component ablations and report uncertainty. Read before you build any ablation experiment.

### Conmy et al. — "Towards Automated Circuit Discovery"

- **arXiv:** 2304.14997
- Automated discovery of circuits in transformers via causal interventions.
- **Why useful:** your Phase 7 reasoning pathways are a special case. Their methodology is generalizable.

### Marks et al. — "Sparse Feature Circuits" ⭐

- **arXiv:** 2403.19647 (2024)
- Combines SAEs with circuit discovery — exactly the methodology Fudan adapts for chess.
- **Why critical:** the LLM precedent for what Fudan did. Read this *before* re-reading Fudan; it'll make the methodology click much faster.

### nostalgebraist — "interpreting GPT: the logit lens"

- **Source:** LessWrong (2020) + follow-ups
- Apply the output head to intermediate residual stream. See what the model "would say" if forced to commit early.
- **Why critical:** your `policy_layer_trajectory` extractor and the Iterative Inference paper depend on this.

### Belrose et al. — "Eliciting Latent Predictions from Transformers with the Tuned Lens"

- **arXiv:** 2303.08112
- Improves on logit lens by learning per-layer translation matrices.
- **Why useful:** better fidelity for intermediate-layer policy estimates. Try if naive logit lens gives noisy results.

---

## Research — Sparse autoencoders (methodology)

If you train your own SAEs in Phase 4, read these.

### Karvonen et al. — "Measuring Progress in Dictionary Learning"

- Already covered above. Re-listed because it's the canonical SAE eval framework.

### Rajamanoharan et al. — "Improving Dictionary Learning with Gated SAEs" / "JumpReLU SAEs"

- **arXiv:** 2404.16014, 2407.14435
- Newer SAE variants with better sparsity-fidelity Pareto frontier.
- **Why useful:** JumpReLU is currently SOTA in many benchmarks. Use if training your own.

### Templeton et al. — "Scaling and evaluating sparse autoencoders" (OpenAI)

- **arXiv:** 2406.04093
- OpenAI's top-k SAE methodology. Alternative to L1-based sparsity.
- **Why useful:** top-k SAEs avoid some training pathologies.
- **Companion code:** `github.com/openai/sparse_autoencoder` (MIT ✓, OpenAI 2023). Targets GPT-2 small; uses `transformer_lens`. Files: `sparse_autoencoder/model.py` (k-sparse SAE), `train.py`, `paths.py` (pretrained model paths), `kernels.py` (custom CUDA kernels), `explanations.py`, plus a `sae-viewer/` TypeScript/React feature browser.
- **Where it fits:** Phase 4 algorithm reference. The k-sparse architecture and training metrics are reusable as algorithm references; the actual hookup code targets GPT-2 not Lc0. If you train your own Phase 4 SAEs and choose k-sparse over JumpReLU or L1, this is the cleanest reference. **Do not adopt the GPT-2 hookup wholesale** — start from `lczero-planning` which is already Leela-shaped.

### Anthropic — "Circuits Updates" on SAE feature labeling

- Automated labeling techniques: max-activating examples → LLM-generated label.
- **Why useful:** direct template for your Phase 5 labeling pipeline.

### Karvonen blog and SAE-specific posts

- **Source:** lesswrong, Karvonen's blog
- Practical SAE training tips from someone who's done it on chess.

---

## Research — Causal interventions, patching, counterfactuals

For Phase 6.

### Meng et al. — "Locating and Editing Factual Associations in GPT" (ROME)

- **arXiv:** 2202.05262
- Causal tracing for locating where facts are stored in transformers.
- **Why useful:** methodology template for "where does Leela store its evaluation of this position?"

### Heimersheim & Nanda — "How to Use and Interpret Activation Patching" ⭐

- **Source:** AI Alignment Forum
- Practical guide. Many subtle variants (zero ablation, mean ablation, resample ablation).
- **Why critical:** read before designing any ablation experiment.

### Goldowsky-Dill et al. — "Localizing Model Behavior with Path Patching"

- **arXiv:** 2304.05969
- Path patching is more precise than activation patching for circuit discovery — relevant for Phase 7 reasoning pathways.

---

## Research — Concept probing methodology

For Phase 3.

### Alain & Bengio — "Understanding intermediate layers using linear classifier probes"

- **arXiv:** 1610.01644 (2016)
- Founding linear-probe paper. Probe intermediate activations with linear classifiers.
- **Why useful:** foundational. Cite when explaining your methodology.

### Belinkov — "Probing Classifiers: Promises, Shortcomings, and Advances"

- **arXiv:** 2102.12452
- Survey of probing methodology, including critiques.
- **Why critical:** read before writing probe code. Saves you from common methodological errors (probe capacity vs information capacity).

### Hewitt & Liang — "Designing and Interpreting Probes with Control Tasks"

- **EMNLP 2019**
- Control tasks for ensuring probes measure representation quality, not their own capacity.
- **Why critical:** your shuffled-labels sanity check (v2 bible §7.4) IS this methodology. Read for rigor.

### Lovering, Forde et al. — "Evaluation Beyond Task Performance: Analyzing Concepts in AlphaZero in Hex" ⭐

- **Venue:** NeurIPS 2022
- **Authors:** Lovering, Forde, Konidaris, Pavlick, Littman (Brown)
- **Companion code:** `alphatology` (see Tier 3 entry above)
- **Three findings that directly affect this project:**
  1. **"MCTS discovers concepts before the neural network learns to encode them."** Empirical justification for Phase 2 (MCTS integration) sitting between Phase 1 and Phase 3.
  2. **Short-term end-game concepts in *final* layers; long-term in *middle* layers.** Affects probe placement in Phase 3.
  3. **"Probing performance alone is insufficient: information may be encoded but not used."** Motivates behavioral tests on top of probes. Should join Phase 3 mandatory reading.
- **Where it fits:**
  - **Phase 2:** MCTS-vs-network signal distinction.
  - **Phase 3:** Required reading. Behavioral-test methodology informs handler design.
  - **Phase 8:** "Evaluation beyond task performance" informs eval grader design.

---

## Research — Chess commentary, NLP, LLM-as-translator

### Jhamtani et al. — "Learning to Generate Move-by-Move Commentary for Chess"

- **Venue:** ACL 2018
- 300k move-commentary pairs.
- **Why useful:** style reference for your LLM prompts. Dataset also in §Datasets.

### Feng et al. — ChessGPT paper

- **arXiv:** 2306.09200, NeurIPS 2023
- Large LLM trained on chess data including PGN, commentary, transcripts.
- **Why useful:** scale data + reference for what a chess-tuned LLM looks like. Open dataset.

### Lee et al. — "Improving Chess Commentaries by Combining Language Models with Symbolic Reasoning Engines" ⭐

- **arXiv:** 2212.08195 (2022)
- 32k chess.com Q&A dialogues paired with game state in PGN.
- **Why critical:** closest existing dataset to your product spec.

### LLM-as-Judge literature

- General body of work on using LLMs to evaluate other LLMs.
- **Why useful:** your eval harness (Phase 8) might benefit from LLM-graded responses on subjective criteria.

### Constitutional AI / Anthropic prompting guides

- **Source:** `docs.claude.com`
- **Why useful:** best practices for the strict prompts your translation layer needs.

---

## Existing chess analysis products (competitive intelligence)

You're not building in a vacuum. Understanding the landscape sharpens your differentiation.

### DecodeChess

- **Site:** `decodechess.com`
- **What it does:** natural-language explanations of chess engine output. Closest existing product.
- **Limitations:** generic templated explanations, no internal-state access (Stockfish wrapper), no novel insights for strong players.
- **Why study:** read 20 random explanations and note the platitudes. Your prompts must beat this.

### chess.com Game Review

- **Site:** chess.com analysis
- **What it does:** move classifications (Brilliant, Great, Best, Excellent, Good, Inaccuracy, Mistake, Blunder), simple explanations, accuracy %.
- **Limitations:** categorical only; templated explanations; surface tactics.
- **Why study:** the classification scheme is the user mental model you inherit.

### Lichess analysis

- **Site:** `lichess.org` (built-in)
- **What it does:** Stockfish-powered, opening explorer, blunder annotations.
- **Limitations:** pure engine output, no commentary.
- **Why study:** reference implementation for engine-integration UX.

### ChessBase

- **Site:** `chessbase.com`
- **What it does:** professional analysis software. Engine integration, opening books, game databases, training tools.
- **Limitations:** targets professionals; no "why" commentary.
- **Why study:** GM workflow tool. Understand it to position for GM adoption.

### Aimchess / chess.com Insights

- **Site:** `aimchess.com`, chess.com Insights
- **What it does:** pattern analysis across your games (weaknesses, time management).
- **Why study:** adjacent product space. Not your target.

### CoSMIC (OpenSI / University of Canberra)

- **Paper:** *Unleashing Artificial Cognition: Integrating Multiple AI Systems*, ACIS 2024, arXiv 2408.04910
- **Repo:** `github.com/TheOpenSI/CoSMIC` (production branch)
- **License:** MIT ✓
- **What it does:** Multi-service LLM router. Five base services — chess, vector DB, RAG, PyCapsule (Python code generation), generic QA — dispatched to by an LLM-based `QueryAnalyser`. Chess service is **Stockfish-only** (no Leela). Chess modules include `chess_qa_puzzle.py` (363 lines), `chess_qa_quality.py`, `chess_genfen.py`, and `chess_gencot.py` — the last performs chain-of-thought *generation* about a position, the canonical LLM-as-inventor pattern forbidden by CLAUDE.md §2 Rule 1.
- **Why it's listed:**
  - **Competitive intelligence:** another LLM-wrapped chess product. Local-LLM emphasis (Llama 3.1 via Ollama).
  - **One useful read in `src/query_analyser/query_analyser.py`** — LLM-based routing logic and per-LLM-family prompt parameterization (`src/maps.py`, `src/query_analyser/user_prompt.py`) is structurally identical to your `CLASSIFIER → HANDLER` design (CLAUDE.md §3). 30-minute read during Phase 1 classifier design.
- **Why it's not adopted:** No Leela, no network internals, no probes, no SAEs. The chess code is the architecture you reject.

### chess-tutor (DVDAGames)

- **Repo:** `github.com/DVDAGames/chess-tutor`
- **License:** **None** (no LICENSE file in repo). Vendoring code is legally murky; treat as read-only reference.
- **Stack:** Next.js 15.1.3 + React 19, `react-chessboard` + `chess.js`, Stockfish WASM client-side, OpenAI gpt-4o as both coach and opponent.
- **What it does:** Beginner-targeted chess tutor. Pure LLM-as-inventor: gpt-4o is asked to "analyze the position and provide feedback."
- **Why it's listed:**
  - **Phase 8 UI mechanics reference:** `react-chessboard@4.7.2` + chess.js + Stockfish WASM is a validated client-side chess UI assembly. Worth knowing when Phase 8 web UI starts.
  - **Wrong-audience example:** explicit beginner targeting; you target 2000+. Wrong-architecture example: LLM as analyzer, not translator.
- **Why it's not adopted:** No license. Wrong target audience. Wrong architecture (violates Rule 1).

---

## Community resources and forums

### Lc0 Discord ⭐

- **Invite:** linked from `lczero.org`
- Active developers respond to questions. Best place to ask about network internals, build issues, BT4 architecture quirks.

### Lc0 GitHub Discussions and Issues

- **URL:** `github.com/LeelaChessZero/lc0/discussions`
- Searchable history of every weird build issue and architecture question.

### AI Alignment Forum / LessWrong (mech interp tag)

- **URLs:** `alignmentforum.org`, `lesswrong.com/tag/interpretability-ml`
- Practical posts on patching, SAEs, probes from people doing the work.

### EleutherAI Discord

- Mech-interp questions and pointers to recent work.

### TalkChess forum

- Chess engine developer community. Build issues, UCI protocol questions.

### Anthropic and DeepMind research blogs

- **URLs:** `anthropic.com/research`, `deepmind.google/research`
- New SAE techniques and interpretability work get blogged before papers go up.

### X (Twitter) accounts to follow

- Lc0 team accounts
- Anthropic interpretability team (Templeton, Bricken, Olah, Cunningham)
- Adam Karvonen (chess SAE work)
- Yoann Poupart (lczerolens, contrastive SAEs)
- Erik Jenner (Lc0 look-ahead)
- Tom McGrath (chess concept probing)
- Neel Nanda (mech interp education)

---

## What's NOT done — your work

To be clear about what you actually have to build:

1. **The question router.** Map free-text questions to signal extractors. Nobody has built this for chess. This is your contribution.
2. **The signal-to-JSON contract for each question type.** What does the LLM receive when the user asks "why not Nf3?" vs "what's the threat?" vs "is this position quiet?" You define this schema.
3. **The LLM translation layer.** System prompts, per-handler instruction templates, validation, fallback behavior. Conceptually simple, requires real iteration.
4. **The differential analysis engine.** Run analysis on (position, position-after-user's-move), diff every signal, rank by magnitude, surface findings. Pieces exist; orchestration doesn't.
5. **The output validator.** Confirms every move mentioned is legal, every eval cited is in JSON, no fabricated lines. Critical and non-obvious.
6. **The MCTS tree extraction with eval trajectories.** Lc0 exposes per-visit info via verbose mode. The Fluri/Paleka/Tramèr `rl_testing` package (`superhuman-ai-consistency`, MIT, arXiv 2306.09983) contains a working parser — `rl_testing/mcts/tree_parser.py` (588 lines) backed by `relaxed_uci_protocol.py`. Decision pending: vendor with Python 3.11 / Lc0 v0.32.1 compat fixes, or clean-room port to our conventions. See ADR-0006.
7. **The evaluation set and harness.** 200+ positions with expert-written expected-answer rubrics. Your chess expertise; nobody else can produce this.
8. **The conversation manager.** Session state, history, position tracking. Standard engineering, specific to chess.
9. **The user interface.** CLI first, web later. Frameworks exist; the work is chess board rendering and conversational UX.

**Roughly 30–40% of the total project.** The remaining 60–70% — engine integration, network ports, SAE training, probe methodology, datasets, board encoding — is solved code you can pull in.

### Phase-by-phase: % existing vs. new

| Phase | % existing tooling | % new code |
|---|---|---|
| 0 (environment) | 95% | 5% |
| 1 (network-first chat) | 70% | 30% |
| 2 (MCTS integration) | 60% | 40% — tree extraction with eval trajectories is yours |
| 3 (probes) | 50% | 50% — lczerolens notebook + McGrath concept list, but your concepts, your labels |
| 4 (SAEs) | 90% (if Fudan works) / 30% (if you train your own) | 10% / 70% |
| 5 (labeling) | 30% | 70% — labels are *your* chess knowledge |
| 6 (differential) | 40% | 60% — diff logic and ranking are novel |
| 7 (reasoning pathways) | 60% (if Fudan releases code) | 40% |
| 8 (polish, UI, alpha) | 50% | 50% |

**Net effect:** the original 6-month plan is closer to 4–5 months if libraries cooperate. They won't always. Budget 50% buffer for integration friction.

---

## Phase-by-phase reading and adoption order

### Before Phase 0 (week 1)

**Read:**
1. v2 Bible cover to cover
2. McGrath et al. AlphaZero concept paper (skim methodology)
3. Lin et al. Fudan paper (your North Star)
4. Davis et al. MDPI probe-vs-SAE paper
5. Jenner look-ahead paper (abstract + conclusions)
6. Monroe BT4 architecture paper (for smolgen + attention policy)

**Install:**
- Python 3.11, uv, CUDA
- lczerolens, python-chess, pytorch
- Lc0 (compiled with `--verbose-move-stats`), Stockfish

**Pin:**
- BT4-1024x15x32h-swa-6147500 (record hash in DECISIONS.md)

**Additions (v3 patch, Phase 0):**
- `superhuman-ai-consistency` consistency-check scripts (forced-move, transformation-invariance) — adversarial eval-position generators with mechanical ground truth. Run once Lc0 v0.32.1 + BT4 are wired up.

### Phase 1 (network-first MVP)

**Read:**
- Monroe BT4 paper (deep dive)
- lczerolens README and notebooks
- Iterative Inference paper
- Lee et al. chess.com Q&A paper (for prompt design)

**Adopt:**
- lczerolens for forward passes
- python-chess for board/move handling
- Frontier LLM API

**Additions (v3 patch, Phase 1):**
- **`leela-interp` `core/uci_to_idx.py` + `core/leela_board.py`** — vendor now during Chunk 4. Resolves CLAUDE.md §8 pitfalls about 1858 move mapping and 112-channel board encoding.
- CoSMIC `src/query_analyser/query_analyser.py` — 30-minute read during classifier design. Reference for LLM-based routing pattern.

### Phase 2 (MCTS integration)

**Read:**
- Lc0 `--verbose-move-stats` documentation
- Hydra Effect paper (early, for awareness)

**Adopt:**
- python-chess engine wrapper for UCI
- Stockfish for comparison handler

**Additions (v3 patch, Phase 2):**
- **`superhuman-ai-consistency / rl_testing.mcts.tree_parser`** — start here. Either vendor with Python 3.11 compat fixes or use as a clean-room port spec. Resolves ADR-0006.
- Lovering, Forde et al. (Hex AlphaZero) — "MCTS discovers concepts before the network does" finding justifies Phase 2 sequencing.
- Davis & Sukthankar "Decoding Chess Mastery" — DLA methodology for the `policy_layer_trajectory` handler.

### Phase 3 (concept probes)

**Read:**
- Alain & Bengio probe paper
- Hewitt & Liang control tasks
- Belinkov probing survey
- McGrath/AlphaZero re-read for concept list

**Adopt:**
- lczerolens probing notebook as starter
- scikit-learn for logistic regression
- Lichess puzzle DB for tactical labels

**Additions (v3 patch, Phase 3):**
- Lovering, Forde et al. — **required**. Behavioral-test methodology on top of probing.
- **`alphatology`** — reference read for the positive/negative/behavioral-test pattern.
- **`leela-interp/tools/probing.py` + `scripts/probing.py`** — starter code for Lc0 probing. Adapt to BT4 + your concept list.

### Phase 4 (sparse features)

**Read:**
- Anthropic "Towards Monosemanticity"
- Anthropic "Scaling Monosemanticity"
- Fudan paper deep re-read
- Karvonen "Measuring Progress in Dictionary Learning"
- Rajamanoharan JumpReLU paper
- Poupart contrastive SAE paper

**Evaluate (week 1):**
- Fudan Leela-SAEs repo — go/no-go decision

**Adopt (if Fudan works):**
- Fudan SAEs and feature dictionary

**Adopt (if Fudan fails):**
- OpenMOSS Language-Model-SAEs or EleutherAI sparsify
- Cloud A100 budget

**Additions (v3 patch, Phase 4):**
- Tang et al. (Skill Adaptation SAEs on Maia-2) — methodology fallback option C.
- **`lczero-planning`** — Phase 4 fallback Option A if Fudan no-goes. **See ADR-0008 for ordering.**
- **`SAE_BoardGameEval`** — Phase 4 evaluation methodology reference (`eval_sae_as_classifier.py`, `eval_board_reconstruction.py`). Pretrained chess SAEs on HF as comparison baseline.
- **`openai/sparse_autoencoder`** — k-sparse SAE algorithm reference if training own SAEs.
- `alphasae` repo — pedagogical reference (read `sae/trainer.py` before touching the others).
- `awesome-sparse-autoencoders` — 30-minute triage at Phase 4 entry.

### Phase 5 (feature labeling)

**Read:**
- Anthropic circuits updates on feature labeling
- Bricken et al. dictionary learning
- Schut et al. concept transfer paper (label quality philosophy)

**Adopt:**
- Build labeling tool (Jupyter or simple web UI)
- Reference Neuronpedia for UX design

**Additions (v3 patch, Phase 5):**
- Tang et al. — feature taxonomy approach.
- `alphasae` repo — `vis/index.html` as minimum-viable feature browser reference.

### Phase 6 (differential / counterfactual)

**Read:**
- Hydra Effect paper (re-read deeply)
- Heimersheim & Nanda activation patching guide
- Meng et al. ROME / causal tracing
- Goldowsky-Dill path patching

**Adopt:**
- nnsight or TransformerLens for intervention code
- Jenner et al. activation patching infrastructure
- Maia models for human-vs-Leela differential

**Additions (v3 patch, Phase 6):**
- Cruz — **required**. Multiple move sequences finding is load-bearing for `did_you_consider`.
- Tang et al. — mediated SAE intervention as Phase 6 counterfactual template.
- **`leela-interp/tools/patching.py` + `scripts/run_global_patching.py` + `scripts/run_single_head.py`** — activation-patching primitives. Required reading alongside Heimersheim & Nanda and Hydra Effect.
- **`leela-interp/tools/piece_movement_heads.py`** — methodology template for BT4-specific named-head taxonomy.
- `superhuman-ai-consistency` evolutionary algorithm — once Phase 6 starts, run it to generate `why_is_my_move_wrong` eval positions where Leela is self-inconsistent.

### Phase 7 (reasoning pathways)

**Read:**
- Fudan paper Algorithm 1 in detail
- Marks et al. Sparse Feature Circuits
- Conmy et al. Automated Circuit Discovery

**Adopt:**
- Fudan's reasoning pathway code (if released)
- Visualization library (Plotly or D3 for graphs)

**Additions (v3 patch, Phase 7):**
- Cruz — supports Fudan reasoning-pathway framing with independent evidence.

### Phase 8 (polish and launch)

**Read:**
- Constitutional AI / Anthropic prompting guides
- ChessGPT and Jhamtani for style references

**Adopt:**
- Web framework (FastAPI + chessground.js)
- Maia networks for "how would a human play" overlays

**Additions (v3 patch, Phase 8):**
- Lovering, Forde et al. — "evaluation beyond task performance" philosophy informs eval grader design.
- chess-tutor — UI mechanics reference. No license, no vendoring; mechanics only.

---

## License survey

Critical if you ever want to commercialize or open-source.

| Asset | License | Implication |
|---|---|---|
| lczerolens | MIT ✓ | Use freely, including commercial |
| python-chess | GPL-3.0 | Source bundling has copyleft; subprocess use fine |
| Lc0 binary | GPL-3.0 | Subprocess use fine for any deployment |
| Stockfish binary | GPL-3.0 | Subprocess use fine |
| Lichess data | CC0 ✓ | Public domain; use freely |
| Lichess puzzles | CC0 ✓ | Public domain |
| Fudan Leela-SAEs | check repo | Verify before integrating |
| OpenMOSS SAEs | check repo (likely permissive) | Verify before integrating |
| EleutherAI sparsify | MIT ✓ | Use freely |
| Maia models | check repo (generally permissive) | Verify |
| TransformerLens | MIT ✓ | Use freely |
| nnsight | MIT ✓ | Use freely |
| captum | BSD-3 ✓ | Use freely |
| Anthropic SAE code | check repo | Verify before integrating |
| superhuman-ai-consistency / rl_testing | MIT ✓ | Use freely, including commercial. Vendoring or porting both fine. |
| leela-interp (Jenner) | GPL-3 | Same license class as Lc0 and python-chess; subprocess/API isolation already planned. No new obligation. |
| lczero-planning (Poupart) | MIT ✓ | Use freely. Note pinned git revs for lczerolens and nnsight forks — verify before depending. |
| SAE_BoardGameEval (Karvonen) | **None visible in repo root** | Check with author before vendoring code. Pretrained SAEs on HF are separately licensed (verify). |
| alphatology (Forde) | MIT ✓ | Use freely. Reference read; no vendoring planned. |
| openai/sparse_autoencoder | MIT ✓ | Use freely. Reference for k-sparse algorithm. |
| alphasae-main | Apache 2.0 ✓ | Use freely, including commercial. |
| awesome-sparse-autoencoders | Apache 2.0 ✓ | Use freely. |
| chess-tutor | **None** | No LICENSE file. Read-only competitive intelligence; do not vendor code. |
| CoSMIC | MIT ✓ | Use freely. Stockfish bundling carries GPL-3 obligations (already known). |
| jax | Apache 2.0 ✓ | Not adopted; framework, not paper code. |
| Skill Adaptation SAE repo (Tang) | unverified (anonymous) | Check on publication |
| Cruz look-ahead follow-up code | unverified | Check arXiv/author homepage |

**Architectural implication:** if you keep GPL components (Lc0, Stockfish, python-chess) as separate subprocess calls behind a thin API server, your application code can ship under any license. If you bundle python-chess source into a closed-source app, GPL-3 obligations apply.

---

## Open threads — re-search periodically

The field moves fast. Check these monthly.

1. **"chess transformer SAE" + 2025/2026** — Fudan follow-ups, competitors, replications.
2. **lczerolens release notes** — new features (especially Lorsa support).
3. **Anthropic transformer-circuits.pub updates** — monthly with new techniques.
4. **Neuronpedia chess SAE features** — they may add chess models to public browser.
5. **DeepMind chess interpretability** — watch for follow-ups to Hassabis/McGrath/Schut.
6. **"reasoning pathways" + interpretability** — Fudan's framing; watch for adoption.
7. **Lc0 BT5 / next-generation networks** — when released, decide migration (probably skip — pinning matters).
8. **JumpReLU / Top-K SAE refinements** — current SOTA may shift.
9. **Hugging Face chess SAE releases** — community uploads accumulate.
10. **Erik Jenner follow-up work** — look-ahead paper had clear next steps.
11. **Yoann Poupart publication stream** — lczerolens author, contrastive SAE author.
12. **awesome-sparse-autoencoders** — `github.com/chrisliu298/awesome-sparse-autoencoders`. Apache 2.0 ✓. Curated paper-and-link list (24 papers, 8 blog posts; last commit ~Sep 2024). Already outdated relative to our stack (no Fudan 2026, no Cruz 2025). Useful as a monthly bookmark — re-scan at Phase 4 entry and on each Anthropic/EleutherAI release. The Gemma Scope, JumpReLU, and Kissane "Interpreting Attention Layer Outputs with SAEs" entries are Phase 4 / Phase 6 relevant and not yet in this PRIOR_WORK.md; worth 30-minute triage when Phase 4 starts.

---

## Quick reference — where to find things

| Need | Go to |
|---|---|
| A Leela network | `storage.lczero.org/files/networks-contrib/` |
| Lc0 source | `github.com/LeelaChessZero/lc0` |
| PyTorch Leela | `github.com/Xmaster6y/lczerolens` |
| SAEs for chess | `github.com/JacklE0niden/Leela-SAEs` (verify) |
| SAE training framework | OpenMOSS or EleutherAI sparsify |
| Puzzles by motif | `database.lichess.org/#puzzles` |
| Real games at scale | `database.lichess.org` |
| Endgame ground truth | Syzygy tablebases |
| Chess commentary data | Jhamtani 300k, ChessGPT, Lee et al. chess.com Q&A |
| Human-mimic networks | `github.com/CSSLab/maia-chess`, HF: `lczerolens/maia-*` |
| Interp toolkit (general) | TransformerLens, nnsight, captum |
| Concept probing template | McGrath AlphaZero paper + lczerolens probing notebook |
| Feature labeling UX inspiration | Neuronpedia |
| Engine community | Lc0 Discord |
| Mech-interp community | transformer-circuits.pub, EleutherAI Discord, AI Alignment Forum |

---

## Final framing

The bible (`BIBLE.md`) was written assuming you'd build most pieces yourself. With this prior work applied:

- Phase 0 (environment): mostly install steps
- Phase 1 (network-first chat): mostly handler glue
- Phase 2 (MCTS integration): the tree extraction with eval trajectories is genuinely your engineering
- Phase 3 (probes): your concept list and your chess expertise are the work
- Phase 4 (SAEs): integration vs. training is the big decision; rest is plumbing
- Phase 5 (labeling): your chess knowledge applied at scale
- Phase 6 (differential): orchestration of existing primitives
- Phase 7 (reasoning pathways): productizing Fudan's research
- Phase 8 (UI and launch): standard product work

**The advice doesn't change:** evaluation set first, validator on every LLM output, never let the LLM invent chess facts. Prior work means more puzzle pieces are pre-made. The picture you assemble is still yours alone.

---

## Maintenance notes

- Re-scan this document at the start of every phase.
- Add new entries as you discover them.
- For every adopted item, link to a `docs/DECISIONS.md` entry with date, version/hash, and a one-line rationale.
- For every rejected item, write a one-line rejection in `DECISIONS.md` so future-you doesn't reopen the question.
- When a paper or repo is superseded by newer work, mark with a strikethrough rather than deleting.

*Last updated: project start. Next review: end of Phase 0.*

## Source locations

Full texts of the papers and source of the reference repos below are not
redistributed in this repository. Use the arXiv IDs and GitHub links given
with each entry.