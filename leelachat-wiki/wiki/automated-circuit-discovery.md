# Towards Automated Circuit Discovery for Mechanistic Interpretability

**Summary**: ACDC is a greedy edge-pruning algorithm that automates the activation-patching step of circuit discovery, recovering known circuits in GPT-2 Small with competitive AUC versus gradient-based baselines. It systematises the mechanistic interpretability workflow into a three-step process and replaces manual patching iteration with a KL-divergence-driven subgraph isolation procedure.

**Sources**: Towards Automated Circuit Discovery.pdf

**Last updated**: 2026-06-02

---

## Citation
Conmy et al. "Towards Automated Circuit Discovery for Mechanistic Interpretability." NeurIPS 2023.  arXiv:2304.14997

## Contribution
The paper systematises the mechanistic interpretability workflow into three steps — select behavior/metric/dataset, define the computational graph granularity, and iteratively patch activations to isolate the responsible subgraph — and automates step three with the ACDC algorithm. ACDC iterates from output to input in topological order, greedily removing edges whose deletion raises KL divergence from the full model by less than a threshold τ, leaving a sparse subgraph. The algorithm is evaluated on five previously hand-traced circuits in GPT-2 Small and compiled tracr transformers, successfully rediscovering all five circuit types. ACDC is positioned as a productivity tool that complements rather than replaces human judgment, as performance remains sensitive to the corrupted distribution and threshold choice. (source: Towards Automated Circuit Discovery.pdf)

## Method
ACDC represents the transformer as a directed acyclic graph (DAG) of attention heads, MLPs, and residual-stream edges. Starting from the output node and working backwards in reverse topological order, it temporarily removes each incoming edge to a node, runs a forward pass with that edge replaced by corrupted activations, and permanently prunes the edge if the resulting KL divergence increase over the full model is below threshold τ (Algorithm 1). Two baselines are compared: Subnetwork Probing (SP), which learns a continuous mask via L0-regularised training rounded to binary at test time; and HISP, which ranks heads by gradient magnitude and prunes all below a cutoff k. All three methods use interchange interventions (clean input with corrupted activations on removed edges) rather than zero activations, though zero-activation variants are also tested. ROC AUC against previously published ground-truth circuits and KL-divergence-vs-edges Pareto curves serve as quantitative evaluation metrics. (source: Towards Automated Circuit Discovery.pdf)

## Key findings
- ACDC automates step 3 of the mechanistic interpretability workflow — finding the subgraph of abstract units responsible for a behavior — by replacing manual activation-patching iteration with a greedy edge-pruning algorithm driven by KL divergence. (source: Towards Automated Circuit Discovery.pdf, Abstract)
- On GPT-2 Small's Greater-Than circuit, ACDC rediscovered all 5/5 component types and selected 68 of 32,000 edges, all of which were identified by prior manual work, with zero false positives. (source: Towards Automated Circuit Discovery.pdf, Abstract)
- KL divergence is the most robust optimisation metric for ACDC across tasks; logit-difference metrics can cause instability because the subgraph's logit difference may exceed the model's, leading to inconsistent pruning decisions. (source: Towards Automated Circuit Discovery.pdf, §C.2, Appendix C)
- ACDC systematically misses "negative" circuit components (those actively harmful to task performance) because the greedy KL objective only rewards their absence, never their inclusion. (source: Towards Automated Circuit Discovery.pdf, §4.1)
- Algorithm performance is very sensitive to the corrupted distribution used for interchange interventions; for example, ACDC fails on tracr tasks with corrupted activations but perfectly recovers the circuit with zero activations. (source: Towards Automated Circuit Discovery.pdf, §4.1)
- All three methods (ACDC, SP, HISP) show that KL divergence is significantly lower for trained versus reset networks, confirming the methods extract genuine task-relevant signal rather than artefacts. (source: Towards Automated Circuit Discovery.pdf, §4.2, Appendix L)
- The ground-truth circuits used for ROC evaluation are themselves imperfect — the IOI circuit alone has 1,041 edges from a model containing over an order of magnitude more — so AUC conclusions are bounded by the reliability of prior human-traced circuits. (source: Towards Automated Circuit Discovery.pdf, §4.1)
- ACDC provides early evidence of novel discoveries: it identified a surprising subgraph relevant to gendered pronoun completion in GPT-2 Small that was an early example of the "summarisation motif". (source: Towards Automated Circuit Discovery.pdf, §6, Appendix K)

## Relevance to Leela Chat
ACDC is directly relevant to Leela Chat Phase 7 (reasoning pathways). The three-step workflow it systematises — define behavior metric, specify DAG granularity, patch to isolate subgraph — is the same methodology the Fudan paper applies to Lc0 for move-selection circuits. When implementing pathway extraction in Phase 7, ACDC's Algorithm 1 (or the open-source library at github.com/ArthurConmy/Automatic-Circuit-Discovery) can be adapted to BT4's residual-stream DAG to automatically identify which attention heads and MLP layers drive Lc0's policy for a given position. (source: Towards Automated Circuit Discovery.pdf)

The key practical warning for Leela Chat is ACDC's sensitivity to the corrupted distribution: the choice of "baseline" position (corrupted input) for interchange interventions must be deliberate and chess-semantically appropriate, or the recovered circuit will be an artefact of that choice. ACDC's known failure to capture negative components — heads that suppress incorrect moves — is a limitation worth planning for explicitly, since suppression heads likely play a role in Lc0's move-discrimination logic and would be invisible to a naive ACDC run. (source: Towards Automated Circuit Discovery.pdf)

At the layer level, ACDC operates at the extractor layer: the subgraphs it recovers are the structured signal that a Phase 7 handler would pass to the translator. The translator would then narrate the identified heads and MLP layers as a reasoning pathway, with the validator verifying that every head and layer cited in prose is grounded in the extracted subgraph JSON.

## Related pages
- [[circuit-discovery]]
- [[activation-patching]]
- [[residual-stream]]
- [[attention-mechanism]]
- [[path-patching]]
- [[sparse-feature-circuits]]
- [[activation-patching-guide]]
- [[hydra-effect]]
- [[fudan-tracing-thought]]
- [[reasoning-pathways]]
