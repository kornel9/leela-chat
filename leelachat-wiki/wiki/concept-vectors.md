# Concept vectors (TCAV) & amnesic probing

**Summary**: Concept vectors are directions in a network's activation space that point toward a human-defined concept; TCAV quantifies how much such a direction influences predictions, and amnesic probing causally ablates the direction to test whether the network actually *uses* it. For Lc0/BT4, both give a principled bridge from "a chess concept is decodable in the residual stream" to "the concept is influential or causally load-bearing."

**Sources**: arXiv:1711.11279 (Kim et al., TCAV, ICML 2018); arXiv:2006.00995 (Elazar et al., TACL 2021)

**Verified**: paper-derived, not reproduced

**Last updated**: 2026-06-02

---

## What it is
A Concept Activation Vector (CAV) is the normal to the decision boundary of a linear classifier trained to separate concept-example activations from random activations in a given layer (source: arXiv:1711.11279 (Kim et al., TCAV, ICML 2018), Section 3.2). TCAV then scores a concept's class-level influence as the fraction of class inputs whose directional derivative toward the CAV is positive — a global, per-class measure rather than a per-input saliency map (source: arXiv:1711.11279 (Kim et al., TCAV, ICML 2018), Section 3.4, Eq. 2). Amnesic probing goes one step further on the causal axis: it uses Iterative Nullspace Projection (INLP) to remove all linearly-predictable directions of a concept from a representation and measures the resulting drop in downstream task performance, because standard probing accuracy shows only whether information is *encoded*, not whether it is *used* (source: arXiv:2006.00995 (Elazar et al., TACL 2021), Section 4, Table 1). This matters for Lc0/BT4 interpretation because a high-AUC chess concept probe does not, on its own, prove Leela relies on that concept for move selection.

## In the literature
- **TCAV (Kim et al. 2018)** derives a concept vector from as few as ~30 positive examples plus random negatives via a linear classifier, then reports TCAVQ — a global class-level influence score validated against ground truth (TCAVQ tracked the known concept while saliency maps were correct only 52% of the time, near chance, in a 50-person study), with a 500-run t-test + Bonferroni correction to filter spurious CAVs (source: arXiv:1711.11279 (Kim et al., TCAV, ICML 2018), Sections 3.4–3.6, 4.2.1, 4.3.2). See [[tcav]].
- **Amnesic probing (Elazar et al. 2021)** ablates concept directions with INLP and shows probe accuracy and task importance are essentially uncorrelated in BERT (Spearman rho=0.085, p=0.871); ablating dependency labels and fine-grained POS dropped LM accuracy by 87.1 and 81.8 points, while high-accuracy phrase-boundary probes were not used at the queried layer, and a random-direction control confirms the drops are concept-specific rather than rank-reduction artifacts (source: arXiv:2006.00995 (Elazar et al., TACL 2021), Section 4, Table 1). See [[amnesic-probing]].

## Relevance to Leela Chat
Both methods land squarely on Phase 3 (concept probes): TCAV is the direct methodological ancestor of the project's linear concept probes — both derive a direction in activation space via a linear classifier — and TCAVQ could serve as a complementary global-influence metric alongside the AUC>0.9 quality gate, with the multi-run significance test adopted wholesale to guard against spurious CAVs (source: arXiv:1711.11279 (Kim et al., TCAV, ICML 2018), project_relevance). Amnesic probing supplies the causal complement: its lesson is that high probe AUC on a chess concept (e.g. knight-outpost presence) does not imply Lc0 *uses* that concept, so the Phase 3 gate should add an ablation-based sanity check measuring policy/value shift, and the INLP nullspace technique connects forward to Phase 6 counterfactual work (source: arXiv:2006.00995 (Elazar et al., TACL 2021), project_relevance). In the pipeline these thread through the extractor layer — a TCAV extractor for a named concept would return a TCAVQ scalar grounded in Leela's layer activations, satisfying **Rule 1** (the LLM never invents chess facts) because the number is derived from Leela's internals rather than asserted (source: arXiv:1711.11279 (Kim et al., TCAV, ICML 2018), project_relevance). The output validator carries the load-bearing epistemological guard: it must not pass LLM prose that frames high probe accuracy as evidence of causal use, since that exact inference is what amnesic probing refutes (source: arXiv:2006.00995 (Elazar et al., TACL 2021), project_relevance).

## Related pages
- [[linear-probes]]
- [[activation-patching]]
- [[board-state-probing]]
- [[tcav]]
- [[amnesic-probing]]
