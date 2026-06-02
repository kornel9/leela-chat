# Activation patching

**Summary**: Activation patching is a causal-intervention technique that copies internal activations (residual stream, attention-head outputs, or SAE features) from one network run into another and measures the change in output, isolating which components carry a behavior. For chess networks like Lc0/BT4 it is the primary tool for proving that a specific board square, layer, or feature is causally responsible for a move decision rather than merely correlated with it.

**Sources**: How to Use and Interpret Activation Patching.pdf, Localizing Model Behavior with Path Patching.pdf, Evidence of Learned Look-Ahead.pdf, The Hydra Effect.pdf, Sparse Feature Circuits.pdf, Understanding the learned look-ahead behavior.pdf, Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models.pdf, UNDERSTANDING SKILL ADAPTATION IN TRANSFORMERS USING SPARSE AUTOENCODERS.pdf

**Last updated**: 2026-06-02

---

## What it is
Activation patching runs the network on two inputs — a "clean" reference and a "corrupted" counterfactual — then overwrites a chosen activation in one run with the value it took in the other, and measures the resulting change in a chosen output metric. Patching clean activations into a corrupted run (denoising) finds components *sufficient* to restore the behavior; patching corrupted activations into a clean run (noising) finds components *necessary* to maintain it, and the two directions are not symmetric mirrors (source: How to Use and Interpret Activation Patching.pdf, §2.4, §3). It matters for interpreting Lc0/BT4 because it converts a correlational probe finding ("this square is decodable") into a causal claim ("this square's activation changes the move logits"), which is the standard of evidence needed before Leela Chat tells a user *why* the network made a decision.

## In the literature
- **How to Use and Interpret Activation Patching** is the methodology reference: it formalizes denoising vs. noising, recommends logit difference (or logprob difference) as the primary exploratory metric because the residual stream adds to logits roughly linearly while probability-based metrics over-emphasize effects near a decision threshold, and catalogs pitfalls including backup/OR-gate components and negative components; corrupted-prompt patching is preferred over zero/mean ablation because it keeps the model in-distribution (source: How to Use and Interpret Activation Patching.pdf, §2.4, §4.1, §4.2, §2.5). See [[activation-patching-guide]].
- **Localizing Model Behavior with Path Patching** generalizes activation patching from nodes to directed paths in the computational graph, introduces the Average Unexplained Effect (AUE) metric, and shows iterative refinement raising the explained proportion of induction-head behavior from 28.2% to 73.5%; it argues path patching implicates fewer heads than input-corruption causal tracing because the counterfactual stays more on-distribution (source: Localizing Model Behavior with Path Patching.pdf, §3.5, §4). See [[path-patching]].
- **Evidence of Learned Look-Ahead** applies activation patching directly to Leela Chess Zero's policy network: patching the third-move target square at layer 10 reduces the correct-move log odds by 1.88 ± 0.04 on average, versus a maximum of 0.55 ± 0.01 for any of the 61 other squares, proving future-move squares are causally load-bearing; ablating a single attention entry in head L12H12 (third-move target → first-move target) drops log odds by more than 1.5 in over 10% of puzzles (source: Evidence of Learned Look-Ahead.pdf, §3.1, §3.2). See [[evidence-learned-look-ahead]].
- **Understanding the learned look-ahead behavior** extends the same patching methodology to the 5th and 7th future moves in Leela, confirming causally non-negligible effects of those distant squares but with effect size varying sharply by position type (strongest for AAC-pattern puzzles, weakest for ACC); notably, linear probing detects opponent-move encoding even where activation patching shows no strong causal signal, so the two methods give complementary rather than redundant readings (source: Understanding the learned look-ahead behavior.pdf, §3, Appendix C). See [[understanding-look-ahead-behavior]].
- **The Hydra Effect** is the critical caveat for any patching/ablation study: ablating one attention layer causes a downstream layer to compensate (the Hydra effect) and late MLPs to reduce their erasure, collectively restoring ~70% of the lost logit signal, so ablation-based importance and direct-effect importance are substantially uncorrelated and single-component patching *understates* a component's true causal role (source: The Hydra Effect.pdf, §2.5, §6 Conclusion, Figure 2c). See [[hydra-effect]] and [[self-repair]].
- **Sparse Feature Circuits** scales patching to SAE features as graph nodes, estimating each feature's causal importance via attribution patching (a linear, one-forward/one-backward-pass approximation of the full intervention) plus integrated-gradients patching for early layers; this yields circuits needing ~100 feature nodes (vs ~1500 neurons) to explain subject-verb agreement in Pythia-70M (source: Sparse Feature Circuits.pdf, §3.1, Fig. 3). See [[sparse-feature-circuits]].
- **Emergent World Models in Chess** uses a closely related intervention rather than activation-copy patching: subtracting a board-square probe direction from a chess-GPT's residual stream causes the model to treat that piece as absent, raising the legal-move rate on the hypothetical modified board from 40.5% to 92%, demonstrating the internal board representation is causally used by the policy (source: Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models.pdf, §4.2, Table 2). See [[emergent-world-models-chess]].
- **Understanding Skill Adaptation Using SAEs** applies additive feature intervention (the SAE-feature analogue of patching) to Maia-2's residual stream: injecting a "threat-response" feature in transitional positions raises best-move accuracy from 0% to over 50% at higher skill brackets versus near-0% for random-feature controls, with a dose-response relationship as intervention strength increases (source: UNDERSTANDING SKILL ADAPTATION IN TRANSFORMERS USING SPARSE AUTOENCODERS.pdf, §5.2.2, Fig. 4). See [[skill-adaptation-saes]].

## Relevance to Leela Chat
Activation patching is the foundational technique for Phase 6 (differential/counterfactual analysis) and Phase 7 (reasoning pathways); the methodology guide is required pre-reading for Phase 6 (source: How to Use and Interpret Activation Patching.pdf, project relevance). In the extractor layer, a patching-based extractor reports which future-move squares or features are causally active when answering "why is my move wrong," using logit difference between the user's move and Leela's top move as the metric rather than move probability because the residual stream adds to logits linearly (source: How to Use and Interpret Activation Patching.pdf, §4.1). The handler chooses the denoising-vs-noising direction deliberately per the AND/OR circuit structure being localized and pairs single-component patching with multi-component ablation to defeat the Hydra effect (source: How to Use and Interpret Activation Patching.pdf, §2.4; The Hydra Effect.pdf, §6 Conclusion). The validator must block prose that over-claims circuit *completeness* when the patching evidence supports only *sufficiency* (AUE measures sufficiency, not completeness), and the translator should surface patching findings as causal — not merely correlational — claims (source: Localizing Model Behavior with Path Patching.pdf, §7.1).

## Related pages
- [[path-patching]]
- [[circuit-discovery]]
- [[self-repair]]
- [[learned-look-ahead]]
- [[residual-stream]]
- [[attention-mechanism]]
- [[linear-probes]]
- [[sparse-autoencoders]]
- [[logit-lens]]
- [[activation-patching-guide]]
- [[evidence-learned-look-ahead]]
- [[hydra-effect]]
- [[sparse-feature-circuits]]
- [[understanding-look-ahead-behavior]]
- [[emergent-world-models-chess]]
- [[skill-adaptation-saes]]
