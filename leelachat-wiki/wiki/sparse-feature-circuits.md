# Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models

**Summary**: A scalable method to discover SAE-feature circuits via attribution patching, enabling interpretable causal graphs and targeted bias removal without disambiguating labels. The paper replaces coarse polysemantic units (attention heads, neurons) with fine-grained SAE features as the nodes of causal circuits.

**Sources**: Sparse Feature Circuits.pdf

**Last updated**: 2026-06-02

---

## Citation
Marks et al.. "Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models." ICLR 2025.  (arXiv:2403.19647)

## Contribution
The paper introduces sparse feature circuits: causally implicated subnetworks of SAE features that explain specific language model behaviors. Prior circuit-discovery work operated on polysemantic units too coarse for downstream use; this work replaces those units with fine-grained SAE features as graph nodes. The resulting circuits are both faithful (they account for model performance on the behavior) and complete (ablating a small number of nodes destroys the behavior). Two downstream applications are demonstrated: SHIFT, which surgically removes classifier sensitivity to unintended spurious signals, and a fully unsupervised pipeline that auto-discovers thousands of narrow model behaviors and their circuits. (source: Sparse Feature Circuits.pdf)

## Method
SAEs are trained for each sublayer (attention, MLP, residual stream, embeddings) of Pythia-70M and loaded for Gemma-2-2B via Gemma Scope. The model is viewed as a computation graph whose nodes are SAE feature activations and SAE error terms. Causal importance of each node is estimated via indirect effects approximated with either first-order attribution patching (one forward + one backward pass) or integrated-gradients patching (N=10 interpolation steps). Edges are scored with an analogous linear approximation using stop-gradients and Jacobian-vector-product tricks to keep backward-pass cost tractable. Nodes and edges below thresholds T_N and T_E are pruned, yielding a sparse subgraph. For SHIFT, a human annotator inspects each circuit feature via Neuronpedia maximally-activating exemplars and zero-ablates features judged task-irrelevant. (source: Sparse Feature Circuits.pdf)

## Key findings
- Sparse feature circuits require far fewer nodes than neuron circuits to reach the same faithfulness: ~100 feature nodes vs ~1500 neurons for Pythia-70M, ~500 vs ~50,000 for Gemma-2-2B. (source: Sparse Feature Circuits.pdf, Fig. 3)
- Attribution patching accurately estimates indirect effects for SAE features and errors, except for layer-0 MLP and early residual stream where integrated-gradients patching is used instead. (source: Sparse Feature Circuits.pdf, §3.1)
- SAE error terms are incorporated as first-class nodes in the circuit rather than discarded, giving a principled decomposition into interpretable features plus currently-unexplained residual. (source: Sparse Feature Circuits.pdf, §2)
- SHIFT achieves near-oracle worst-group accuracy (89.0% vs oracle 91.9% on Pythia-70M) without access to disambiguating labels or even advance knowledge of the spurious signal. (source: Sparse Feature Circuits.pdf, Table 2)
- Neuron-based SHIFT fails because neurons most implicated in spurious classification also carry ground-truth signal; SAE features are separable enough for clean surgical ablation. (source: Sparse Feature Circuits.pdf, §4)
- Both Pythia-70M and Gemma-2-2B implement subject-verb agreement across a relative clause via two parallel pathways: a number-detection pathway and a PP/RC boundary-detection pathway that carries number information to the verb position. (source: Sparse Feature Circuits.pdf, §3.3)
- A fully unsupervised pipeline auto-discovers thousands of narrow model behaviors (e.g., predicting the next number in a sequence, predicting commas in dates) and produces a feature circuit for each. (source: Sparse Feature Circuits.pdf, §5)
- Small circuit completeness holds even when SAE error nodes are left in place: ablating a small number of feature nodes is sufficient to destroy the behavior, confirming the circuit captures the essential computation. (source: Sparse Feature Circuits.pdf, §3.2)

## Relevance to Leela Chat
This paper is the primary methodological reference for **Phase 7 (Reasoning Pathways)** in Leela Chat, which aims to trace how Lc0's BT4 transformer arrives at a move decision through interpretable computational steps. (source: Sparse Feature Circuits.pdf)

The sparse-feature-circuit discovery algorithm — SAE features as graph nodes, attribution patching for edge weights — is the exact machinery needed to build Leela Chat's **pathway extractor layer**: given a contrastive pair (chosen move vs. rejected move), run the circuit discovery procedure over Lc0's SAEs to identify which features are causally responsible for the preference. (source: Sparse Feature Circuits.pdf)

The **SHIFT technique** is a concrete template for the **Phase 6 counterfactual handler**: if a Leela feature fires spuriously (e.g., a material-count feature active on the wrong side), SHIFT shows how to surgically ablate it and measure the effect on Lc0's move probabilities without retraining. (source: Sparse Feature Circuits.pdf)

Critically for the **extractor layer contract**, the paper establishes that SAE error terms must be kept as first-class nodes rather than discarded. A Leela Chat extractor that silently drops the SAE error term would lose variance the SAE has not yet decomposed, violating the extractor completeness requirement and potentially producing misleading findings JSON for the translator. (source: Sparse Feature Circuits.pdf)

Note: all experiments in this paper use natural language models (Pythia-70M, Gemma-2-2B). Applicability to Lc0/BT4 requires adapting the contrastive-pairs framing and ensuring Lc0 SAEs are trained with compatible architecture. The chess-specific instantiation of these ideas is [[contrastive-sae-planning]] (Fudan Lin et al.). (source: Sparse Feature Circuits.pdf)

## Related pages
- [[sparse-autoencoders]]
- [[circuit-discovery]]
- [[activation-patching]]
- [[feature-labeling]]
- [[residual-stream]]
- [[path-patching]]
- [[automated-circuit-discovery]]
- [[activation-patching-guide]]
- [[measuring-progress-dictionary-learning]]
- [[scaling-evaluating-saes]]
- [[contrastive-sae-planning]]
