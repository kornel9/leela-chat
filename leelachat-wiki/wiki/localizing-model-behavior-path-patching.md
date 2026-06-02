# Localizing Model Behavior with Path Patching

**Summary**: Path patching generalizes activation patching by localizing a model's behavior to specific *paths* in its computational graph rather than to nodes, and scores each localization hypothesis with a quantitative "Average Unexplained Effect" (AUE).

**Sources**: Localizing Model Behavior with Path Patching.pdf

**Last updated**: 2026-06-02

---

## Citation
Goldowsky-Dill et al. (Redwood Research). "Localizing Model Behavior with Path Patching." arXiv 2023. arXiv:2304.05969.

## Contribution
Path patching localizes model behavior to a set of *paths* in the network's computational graph rather than a set of nodes, performing causal interventions that replace "unimportant" paths with values from a counterfactual input while holding "important" paths at the reference input (source: Localizing Model Behavior with Path Patching.pdf). It yields a quantitative *average unexplained effect* (AUE) metric for judging how faithful a circuit hypothesis is, and ships an open-source framework.

## Method
The network is represented as a DAG, and a "Treeify" transformation duplicates shared subtrees so each input→output path has its own copy of the input (source: Localizing Model Behavior with Path Patching.pdf, §2.4 Fig. 3). A hypothesis specifies a set of important paths P; all other paths receive a counterfactual input x_c while important paths receive the reference x_r. The AUE quantifies how much behavior remains unexplained; hypotheses are refined iteratively — inspect attribution, add paths, re-measure (source: Localizing Model Behavior with Path Patching.pdf, §2.3 / §2.5). Experiments run on a 2-layer attention-only transformer (induction heads) and GPT-2 small/XL.

## Key findings
- Path patching is strictly more expressive than node-level ablation because paths capture directed interactions between components, not just which nodes are present (source: Localizing Model Behavior with Path Patching.pdf, §1 / §2.1).
- The AUE provides a quantitative score: AUE=0 means the hypothesis fully explains the behavior on the tested distribution (source: Localizing Model Behavior with Path Patching.pdf, §2.3 Eq. 1).
- The sparse All-Initial induction-head hypothesis (15 paths) explains only 28.2% of behavior; iterative refinement to All-Final reaches 73.5% — a 45.3-point absolute gain (source: Localizing Model Behavior with Path Patching.pdf, §3.5 Table 3).
- On GPT-2 small, marking 8 of 144 heads as important recovers 98.0% of the loss on number-continuation behavior (source: Localizing Model Behavior with Path Patching.pdf, Fig. 11).
- Path patching implicates fewer heads than causal tracing (input-corruption) because the counterfactual stays more on-distribution, producing a sparser abstraction (source: Localizing Model Behavior with Path Patching.pdf, §4).
- AUE measures *sufficiency, not completeness*: low AUE does not prove that minority paths are truly absent (source: Localizing Model Behavior with Path Patching.pdf, §7.1).
- Path patching is a two-sample special case of causal scrubbing, making it computationally cheaper (source: Localizing Model Behavior with Path Patching.pdf, §6.1).

## Relevance to Leela Chat
Path patching is the foundational technique for **Phase 6** (differential / counterfactual analysis): answering "why is my move wrong" means isolating which residual-stream paths carry the decisive evaluation signal — exactly the directed causal intervention path patching provides. The Treeify + AUE framework gives a quantitative faithfulness criterion for any circuit explanation of an Lc0 evaluation, and the iterative refinement loop (hypothesis → attribution heatmap → expand → re-measure AUE) maps onto the planned `why_is_my_move_wrong` handler. Two cautions to surface in the **validator** so prose never over-claims: AUE measures sufficiency not completeness, and the paper's results are on GPT-2 / toy attention-only models — applicability to Lc0/BT4's smolgen-modified attention must be verified first (source: Localizing Model Behavior with Path Patching.pdf, §7.1 / Notes).

## Related pages
- [[path-patching]] — the concept page (this is its source paper)
- [[activation-patching]]
- [[circuit-discovery]]
- [[residual-stream]]
- [[attention-mechanism]]
- [[activation-patching-guide]]
- [[automated-circuit-discovery]]
- [[sparse-feature-circuits]]
- [[hydra-effect]]
- [[tuned-lens]]
