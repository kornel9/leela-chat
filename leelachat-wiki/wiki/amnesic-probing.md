# Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals

**Summary**: Extends probing by causally ablating concept directions via INLP nullspace projection, measuring task-performance drop rather than mere extractability. Shows that standard probe accuracy and actual model usage of a concept are essentially uncorrelated.

**Source**: arXiv:2006.00995 (Elazar et al., TACL 2021)

**Verified**: paper-derived, not reproduced

**Last updated**: 2026-06-02

---

## Contribution
Standard probing only asks whether information about a concept is linearly decodable from a representation — it cannot answer whether the model actually relies on that concept for its predictions. Amnesic Probing closes this gap by applying Iterative Nullspace Projection (INLP) to surgically remove all linearly-predictable directions of a chosen concept from an intermediate representation, then measuring the change in downstream task performance. If performance collapses after removal, the model was genuinely using that concept; if performance holds (or improves), the concept was merely encoded but not used at that layer. Applied to BERT across six linguistic properties, the paper demonstrates that these two questions have essentially uncorrelated answers — exposing a fundamental epistemological gap that any probing-based interpretability claim must address.

## Method
INLP iteratively trains a linear SVM probe to predict concept Z from representation H, projects H onto the nullspace of that probe's weight matrix (erasing the best-separating direction), and repeats until no probe exceeds majority-class accuracy. The result is a representation from which all linearly-present information about Z has been removed. Two controls rule out confounds: a Rand baseline removes the same number of random directions (isolating rank-reduction effects from concept-specific removal), and a Selectivity check verifies that the removed information was genuinely concept-specific by confirming it was recoverable when gold concept features were provided as fine-tuning input. Layer-wise experiments ablate the concept at each intermediate layer and measure both downstream LM accuracy impact and whether subsequent layers can recover the removed information.

## Key findings
- Standard probing accuracy is not correlated with the actual importance of a property for a downstream task: Spearman rho=0.085, p=0.871 in BERT LM experiments. (source: arXiv:2006.00995, Section 4, Table 1)
- Dependency labels and fine-grained POS are actively used by BERT for word prediction: amnesic ablation drops LM accuracy by 87.1 and 81.8 points respectively. (source: arXiv:2006.00995, Section 4, Table 1)
- Phrase-boundary markers show high probing accuracy (~85%) but removing them slightly improves LM performance, indicating they are not used at the queried layer. (source: arXiv:2006.00995, Section 4, Table 1)
- Within POS, closed-class words (conjunctions, particles, determiners) suffer far larger prediction degradation on removal (73.7, 77.7, 65.7 point drops) than open-class words like nouns (8.6 point drop). (source: arXiv:2006.00995, Section 6, Table 3)
- Where a concept is most easily extractable by a probe does not coincide with where its ablation most damages downstream performance — the two localization methods yield different conclusions. (source: arXiv:2006.00995, Section 7.2, Figure 4)
- INLP removes only linearly-present information; some concept directions may be recoverable by subsequent layers, especially for earlier-layer ablations, indicating representational redundancy. (source: arXiv:2006.00995, Section 7.1, Figure 3)
- The behavior of masked vs. non-masked BERT representations differs substantially, and probing conclusions drawn from non-masked representations may not transfer to the model's actual operating mode. (source: arXiv:2006.00995, Section 5, Tables 1-2)
- INLP's random-direction control (Rand) shows that rank reduction alone does not account for task performance drops — targeted concept removal causes significantly larger drops than equal-dimensional random removal. (source: arXiv:2006.00995, Section 4, Table 1, Rand vs. Amnesic rows)

## Relevance to Leela Chat
Phase 3 (linear probes) is the direct application point. High probe AUC on a chess concept — for example, knight-outpost presence — does not imply that Lc0 uses that concept for move selection or policy scoring. The Phase 3 quality gate should include an amnesic sanity check for top-priority concepts, ablating the concept direction and measuring policy/value shift before treating probe accuracy as evidence of causal use.

At the extractor layer, any claim that "Lc0 uses concept X to evaluate Y" requires causal evidence (performance drop on ablation), not just probe accuracy. The McGrath AlphaZero probe work did not fully address this gap, and the same epistemic hole exists in the Leela probe pipeline.

For Rule 1 and the validator (Phase 1 onwards): the validator must not pass LLM prose that frames high probe accuracy as evidence of causal use — that inference is exactly what this paper refutes. The INLP nullspace technique is also the mechanism behind concept-erasure and representation surgery experiments that are likely relevant for Phase 6 counterfactual work.

## Related pages
- [[linear-probes]]
- [[activation-patching]]
- [[board-state-probing]]
