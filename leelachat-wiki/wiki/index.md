# Wiki Index

A knowledge base for the **Leela Chat** project — a conversational interface to Leela Chess Zero (Lc0/BT4) that grounds every chess statement in signals extracted from the engine's MCTS tree and network internals. It distils the project's research corpus — papers (PDFs in `raw/`, gitignored) and the vendored repos under `update_research/repos/` — into interlinked, source-cited notes. This vault is a **navigation layer into `docs/PRIOR_WORK.md`**, which stays canonical for adopt/skip judgements.

**Layout:** *paper notes*, *concept pages* (cross-paper synthesis), *repo notes* (what a dependency provides + what to adopt vs. skip + pinned commit), and *tool pages*. Claims cite their source; cross-references use `[[...]]` links. Notes are LLM-synthesised secondary summaries — verify load-bearing numbers against the source. ⭐ marks priority items from `PRIOR_WORK.md`.

## Papers — chess interpretability
- [[acquisition-chess-knowledge-alphazero]] ⭐⭐ — human chess concepts emerge at specific layers/training-steps in AlphaZero, read out by linear probes.
- [[bridging-human-ai-knowledge-gap]] — unsupervised discovery of *novel* AlphaZero concepts, verified teachable to four grandmasters.
- [[emergent-world-models-chess]] — a PGN-trained GPT builds a linear board model + a steerable latent Elo vector.
- [[chess-testbed-lm-state-tracking]] — chess as a controlled probe of transformer world-state tracking.
- [[limits-of-probes]] ⭐ — linear probes beat SAEs for board-state editing (88% vs 41% legal-move success).
- [[alphazero-hex-evaluation]] — probing + behavioural tests in Hex: encoded ≠ causally used.
- [[evidence-learned-look-ahead]] ⭐ — Lc0 internally represents and *causally uses* future-move squares.
- [[understanding-look-ahead-behavior]] ⭐ — extends look-ahead to the 5th/7th move; multi-branch planning evidence.
- [[iterative-inference-chess]] — logit lens on Lc0's policy: three-phase progression, late-layer safety overrides.
- [[decoding-chess-mastery]] — attention-pattern taxonomy + layer-wise direct logit attribution in a chess transformer.
- [[contrastive-sae-planning]] ⭐ / [[contrastive-sae-interpreting]] — contrastive SAEs extract Lc0 planning features (Poupart; two versions).
- [[measuring-progress-dictionary-learning]] — board-reconstruction metrics for judging SAE quality on board-game models.
- [[skill-adaptation-saes]] — SAEs on Maia-2 find threat-response features that causally shift effective skill.
- [[fudan-tracing-thought]] ⭐⭐⭐ — **the North Star**: Transcoders + Lorsa decompose Lc0/BT4 into causally-verified reasoning pathways.

## Papers — chess engines & architecture
- [[alphazero-science]] / [[alphazero-arxiv]] — AlphaZero: self-play MCTS + neural net, superhuman from the rules alone.
- [[mastering-chess-transformer-monroe]] ⭐⭐ — the "Chessformer"/BT4-style architecture (smolgen, attention policy head).
- [[grandmaster-chess-without-search]] — a 270M transformer reaches 2895 Lichess blitz Elo with no search.
- [[searchless-chess]] — DeepMind's released code+weights for that searchless model (a concrete network-alone baseline).
- [[maia-aligning-superhuman-ai]] — Maia: an engine retrained to *match* human play at each rating level.
- [[maia-2]] — a single skill-conditioned model spanning the rating spectrum (replaces juggling separate Maia nets; Phase 6).
- [[chessgpt]] — large chess + language dataset and joint policy/language models.

## Papers — general interpretability methods
- [[activation-patching-guide]] ⭐ — practical activation-patching method (denoising/noising, metric choice, pitfalls).
- [[localizing-model-behavior-path-patching]] — path patching: localize behaviour to directed paths + the AUE faithfulness metric.
- [[automated-circuit-discovery]] — ACDC automates circuit discovery by greedy edge-pruning.
- [[sparse-feature-circuits]] ⭐ — SAE-feature causal graphs (the LLM precedent the Fudan work adapts).
- [[hydra-effect]] ⭐ — self-repair: ablated components get compensated downstream — a caveat for all patching.
- [[tuned-lens]] — per-layer affine decoders that fix the brittle logit lens.
- [[scaling-evaluating-saes]] — OpenAI's TopK SAE recipe and clean scaling laws.

## Papers — interpretability methods (2024–2025 gap-fill)
- [[transcoders-find-circuits]] — transcoders: input-invariant sparse MLP replacements for weights-based circuits (the Fudan building block).
- [[circuit-tracing]] — Anthropic attribution graphs + "Biology of an LLM": the LLM-side parent of reasoning pathways.
- [[attention-saes-kissane]] — SAEs on attention-layer outputs (the basis for Fudan's Lorsa).
- [[saebench]] — 8-metric SAE benchmark; proxy gains ≠ practical quality (a Phase-4 go/no-go input).
- [[feature-absorption-matryoshka]] — feature absorption (an SAE pathology) + Matryoshka SAEs (the current fix).
- [[gemma-scope]] — 400+ open JumpReLU SAEs across Gemma 2; the SAE calibration suite.
- [[auto-interpretability]] — explainer→scorer auto-labeling pipeline (delphi); concrete Phase-5 infrastructure.
- [[tcav]] — concept activation vectors: the methodology under concept probing.
- [[amnesic-probing]] — causal ablation of a concept direction: "encoded vs. causally used."
- [[cot-faithfulness]] — verbalized reasoning is systematically unfaithful (the backbone of Rule 1 + the validator).
- [[constrained-decoding]] — FSM-guided generation: the pre-hoc complement to the output validator.
- [[open-problems-mech-interp]] — field roadmap + a skeptic's checklist for the feature-based bet.

## Papers — chess commentary (NLG)
- [[learning-to-generate-move-commentary-jhamtani-2018]] — the canonical 298K commentary dataset + the GAC model.
- [[improving-chess-commentaries-symbolic]] — Lc0 as a symbolic oracle + a controllable LM for grounded commentary.

(See also the commentary concept cluster below; [[chessgpt]] and [[maia-aligning-superhuman-ai]] also touch commentary/human-style.)

## Concepts
**Representations & probing** — [[residual-stream]] · [[world-models]] · [[board-state-probing]] · [[linear-probes]] · [[logit-lens]] · [[concept-vectors]]
**Sparse features** — [[sparse-autoencoders]] · [[feature-labeling]] · [[reasoning-pathways]] · [[transcoders]] · [[attention-saes]] · [[auto-interp]]
**Causal analysis** — [[activation-patching]] · [[path-patching]] · [[circuit-discovery]] · [[self-repair]]
**Chess networks** — [[transformer-chess-architecture]] · [[attention-mechanism]] · [[mcts]] · [[learned-look-ahead]]
**Strength & alignment** — [[superhuman-ai]] · [[maia-human-alignment]]
**Reliability & translation** — [[faithfulness]]

## Chess commentary (concept cluster)
- [[chess-commentary-generation]] · [[chess-commentary-dataset]] · [[gac-model]] · [[commentary-categories]] · [[commentary-features]] · [[content-selection-pragmatics]] · [[bleu-limitations-nlg]]

## Repositories & vendored source
Vendored under `update_research/repos/` (gitignored from the wiki). Each note records what it is, what to adopt vs. skip, the phase it serves, and the pinned commit. `PRIOR_WORK.md` remains canonical.

**Leela / chess engines & interp foundation** — [[lc0]] · [[lczerolens]] ⭐ · [[leela-interp]] ⭐ · [[leela-logit-lens]] · [[lczero-planning]] · [[leela-saes]] · [[lczero-training]]
**SAE training & evaluation** — [[saelens]] · [[sparsify]] · [[openai-sparse-autoencoder]] · [[sae-boardgameeval]] · [[alphasae]] · [[llamascopium]] · [[feature-circuits]] · [[awesome-sparse-autoencoders]]
**Mechanistic-interp toolkits** — [[transformerlens]] · [[nnsight]] · [[zennit]] · [[acdc-repo]] · [[tuned-lens-repo]]
**Chess models, probing & eval** — [[maia-chess]] · [[chess-llm-interpretability]] · [[learning-chess-blindfolded]] · [[alphatology]] · [[superhuman-ai-consistency]]
**Commentary & product reference** — [[chess-commentary-generation-repo]] · [[cosmic]] · [[chess-tutor]]

## Tools and dependencies
- [[python-chess]] — board, FEN, legality, UCI engine wrapper. (Vendored source documented in-page.)
- [[stockfish]] — classical engine for eval scoring and differential analysis. (Vendored source documented in-page.)

## Log
- [[log]] — append-only record of wiki operations.
