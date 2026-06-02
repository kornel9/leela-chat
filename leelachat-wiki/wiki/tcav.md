# Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)

**Summary**: TCAV uses directional derivatives along learned concept vectors to quantify how much a user-defined concept influences a neural net's class predictions, providing a global per-class explanation rather than per-input saliency.

**Source**: arXiv:1711.11279 (Kim et al., TCAV, ICML 2018)

**Verified**: paper-derived, not reproduced

**Last updated**: 2026-06-02

---

## Contribution
TCAV introduces Concept Activation Vectors (CAVs): directions in a network's internal activation space that point toward a user-supplied concept. An analyst provides a small set of positive concept examples (as few as 30) and a set of random counterexamples; a linear binary classifier is trained on layer activations to separate them, and the normal to its decision boundary becomes the CAV. The TCAV score (TCAVQ) for a concept C, class k, and layer l is then the fraction of k-class inputs whose directional derivative with respect to the CAV is positive — a single, interpretable scalar that says "for what proportion of these inputs did this concept push the network toward this prediction?" A multi-run statistical significance test (t-test with Bonferroni correction, typically 500 runs) filters out spurious CAVs before any score is reported. The method requires no model retraining and no access to the original training data, making it applicable post-hoc to any differentiable model.

The paper's central motivation is that per-input saliency maps fail as an explanation medium: in a controlled experiment, saliency maps communicated the correct dominant concept to human raters only 52% of the time — near random chance. TCAV's global, class-level framing sidesteps this failure mode and produces scores that can be directly audited for validity or bias.

## Method
For a target concept, the analyst collects positive examples (images, board positions, or any activation-producing inputs) and a random negative set. A linear classifier is trained on the activations of a chosen layer to separate positives from negatives; its decision-boundary normal is the CAV. The TCAVQ score is computed as the fraction of class inputs where the dot product of the gradient of the class logit (with respect to activations) and the CAV is positive (Eq. 2 in the paper). Relative CAVs extend this to fine-grained comparisons between semantically related concepts by using one concept's positives as the other's negatives. The statistical significance filter trains many CAVs from different random negative sets and uses a t-test (Bonferroni-corrected across concepts) to retain only those whose TCAVQ distribution is distinguishable from the null.

## Key findings
- A CAV is the normal to the decision boundary of a linear classifier trained to separate concept activations from random activations in a given layer. (source: arXiv:1711.11279, Section 3.2)
- TCAVQ (the fraction of class inputs with positive directional derivative toward the concept) is a global, class-level explanation, not a per-input measure. (source: arXiv:1711.11279, Section 3.4, Eq. 2)
- Statistical significance testing via 500-run t-test with Bonferroni correction successfully filters spurious CAVs; e.g., it eliminated 'dotted' as a spurious concept for zebras where 'striped' was the robust concept. (source: arXiv:1711.11279, Section 3.5 and Section 4.2.1)
- Saliency maps communicated the correct dominant concept only 52% of the time (near random chance) in a controlled 50-person human experiment. (source: arXiv:1711.11279, Section 4.3.2)
- TCAV works with as few as 30 concept examples collected from a search engine, as demonstrated for the 'arms' concept in the dumbbell class. (source: arXiv:1711.11279, Section 4.2.1)
- Relative CAVs allow fine-grained comparison between semantically related concepts (e.g., striped vs. dotted vs. meshed) by using one concept set as the other's negatives. (source: arXiv:1711.11279, Section 3.6)
- Linear classifier accuracy on concept activations is higher in deeper layers for abstract concepts (objects, people) but roughly uniform across layers for simple concepts (colors), confirming a known hierarchy. (source: arXiv:1711.11279, Section 4.2.2, Figure 5)
- TCAV is not fooled by adversarial examples in the same way the classifier is: adversarial 'zebras' yield TCAVQ distributions that differ from real zebras even when the classifier is 99-100% confident. (source: arXiv:1711.11279, Appendix A, Figure 11)

## Relevance to Leela Chat
TCAV is the direct methodological ancestor of the linear concept probes the project trains in Phase 3: both derive a direction in activation space via a linear classifier and then measure how much that direction influences network behavior. The key difference is that TCAV computes a class-level influence score via directional derivatives, whereas Phase 3 probes typically report AUC on held-out positions. The project could adopt TCAVQ as a complementary global-influence metric alongside AUC once probes are trained — a TCAV extractor for a named concept (e.g., "knight outpost active") would return a TCAVQ scalar grounded in Leela's residual-stream activations, satisfying Rule 1 (extractor layer, numbers only, no LLM invention).

The concept-example-driven workflow — analyst provides ~30 examples, no model retraining, forward-pass activations only — maps cleanly onto what lczerolens already provides. The statistical significance test (multi-run t-test, Bonferroni) is a mandatory guard against spurious CAVs and should be adopted wholesale if TCAV-style scoring is added, to prevent ungrounded concept claims from reaching the translator layer and failing validation. The BT4 Smolgen attention structure means attention-head activations require care, but residual-stream activations at any layer are straightforward inputs to a CAV classifier. The paper's finding that saliency maps communicate concepts at near-chance accuracy reinforces the project's existing preference for directional/structural interpretability over gradient saliency for chess explanation.

Phase mapping: primarily Phase 3 (concept probe training and evaluation); secondarily Phase 5 (feature labeling, where TCAV-style concept-set curation is directly analogous to the labeling workflow) and Phase 6 (counterfactual analysis, where relative CAVs could support contrastive explanations). The TCAVQ scalar would live at the extractor layer; the significance test output (retained/rejected) would be a field in the extractor's Pydantic return model, not an inline prose judgment.

## Related pages
- [[linear-probes]]
- [[concept-vectors]]
- [[acquisition-chess-knowledge-alphazero]]
