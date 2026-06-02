# Feature labeling

**Summary**: Feature labeling is the process of assigning human-meaningful chess interpretations (e.g. "rook threat on h7", "own-pawn detection at g2") to the latent features a sparse autoencoder, probe, or concept-vector method extracts from an Lc0/BT4 network, then validating those labels against ground-truth chess rules.

**Sources**: Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models.pdf, Scaling and evaluating sparse autoencoders.pdf, Sparse Feature Circuits.pdf, Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents.pdf, UNDERSTANDING SKILL ADAPTATION IN TRANSFORMERS USING SPARSE AUTOENCODERS.pdf, Bridging the Human–AI Knowledge Gap.pdf, Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf

**Last updated**: 2026-06-02

---

## What it is

Interpretability methods (SAEs, transcoders, linear probes, sparse concept vectors) decompose a network's internal activations into many candidate features, but a raw feature is just a direction in activation space with no name. Feature labeling closes that gap: it identifies what chess concept each feature tracks — a piece on a square, a threat, a planning idea — and measures how reliably the label holds via precision/recall or AUC against rule-based ground truth. For Leela Chat this matters because Rule 1 forbids the LLM from inventing chess facts: only features that carry a validated, human-readable label can be cited in a response, and the label is what makes a number say something a chess player understands.

## In the literature

- **Measuring Progress in Dictionary Learning** uses chess's formally specifiable board state to label and grade SAE features objectively. It defines coverage (max F1 of any feature for a given board-state property) and board reconstruction, finding that strategic-concept labels vary sharply by complexity — check scores 1.00 reconstruction / 1.00 coverage, but fork only 0.13 / 0.38 and pin 0.20 / 0.33 — so some chess concepts are far harder to label cleanly than others (source: Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models.pdf, Table 4). It also notes SAE features usually track properties relative to the player to move (mine/yours) rather than absolute color (source: Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models.pdf, §5). See [[measuring-progress-dictionary-learning]].

- **Scaling and Evaluating Sparse Autoencoders** treats labeling as a measurable quality axis: it scores feature explainability with N2G (n-gram-with-wildcards) precision and recall and with recovery of 61 known binary-classification features, and finds TopK SAEs give >1.5x better N2G recall at slightly lower precision than ReLU, with both metrics improving with scale and no supervised signal (source: Scaling and evaluating sparse autoencoders.pdf, §4.2 and §4.3). It also reports that chess PGN notation surfaces as its own recurring (~20% dense) labeled feature cluster in GPT-2 small SAEs (source: Scaling and evaluating sparse autoencoders.pdf, §E.6). See [[scaling-evaluating-saes]].

- **Sparse Feature Circuits** relies on human feature labels at a decisive step: a person inspects each circuit feature via maximally-activating exemplars (Neuronpedia) and ablates the ones judged task-irrelevant (the SHIFT technique), reaching near-oracle worst-group accuracy (89.0% vs oracle 91.9% on Pythia-70M) without disambiguating labels — and this works only because SAE features are separable enough to label and ablate cleanly, whereas neurons that carry spurious signal also carry ground-truth signal (source: Sparse Feature Circuits.pdf, Table 2 and §4). See [[sparse-feature-circuits]].

- **Contrastive Sparse Autoencoders for Interpreting Planning** labels chess SAE features by activation-maximisation, surfacing king-safety corridor / escape-square features and rook-threat features that fire for both colours, and partitions the dictionary into labeled "common" (root-state) versus "differentiating" (planning) features validated by entropy (d-features H=3.24 vs c-features H=2.81) (source: Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents.pdf, §4.2 and Table 1). It flags a labeling failure mode: oversized dictionaries produce trajectory-specific lookup-table features rather than generalisable, labelable planning concepts (source: Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents.pdf, §5.1). See [[contrastive-sae-planning]].

- **Understanding Skill Adaptation in Transformers** labels features against per-square chess concepts and uses a Chess Concept AUC (CC-AUC) as the labeling/selection metric: the best Maia-2 SAE scores CC-AUC 0.88–0.90 on concepts like "capture possible on e3", and CC-AUC doubles as an early-stopping signal during SAE training (source: UNDERSTANDING SKILL ADAPTATION IN TRANSFORMERS USING SPARSE AUTOENCODERS.pdf, §5.1, Fig. 3). The labeled "square no longer attacked" feature is causally validated: boosting it lifts best-move accuracy from 0% to over 50% versus near-0% for random features (source: UNDERSTANDING SKILL ADAPTATION IN TRANSFORMERS USING SPARSE AUTOENCODERS.pdf, §5.2.2, Fig. 4 left). See [[skill-adaptation-saes]].

- **Bridging the Human–AI Knowledge Gap** labels sparse concept vectors found by convex optimisation and validates them on supervised concept datasets, with constraints holding at 0.99 for piece concepts, 0.92 for the Strategic Test Suite, and 1.00 for opening concepts across layers 19–23 — learnable from as few as 10 labelled examples (source: Bridging the Human–AI Knowledge Gap.pdf, Table 3, §5.1.2). It applies an additional novelty filter (SVD basis comparison of AZ vs human-game latents) so labels can distinguish concepts genuinely new to the engine from ones already in human chess knowledge (source: Bridging the Human–AI Knowledge Gap.pdf, Table 1). See [[bridging-human-ai-knowledge-gap]].

- **Tracing the Thought of a Grandmaster-level Chess-Playing Transformer** (the Fudan paper, on the project's pinned BT4 network) labels Transcoder and Lorsa features by highest-activating squares and validates each with rule-based precision/recall, reporting that these features are far more interpretable than raw MLP neurons (human activation-consistency 4.15 / 3.73 vs 2.05 on a 1–5 scale) and often near 100% precision — though some tactical features show low recall from feature splitting (source: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf, Table 1, Table 2, §7.1). Concretely-labeled features drive causal findings, e.g. ablating the 100%-precision own-pawn-detection feature Tc.0.12663@g2 flips P(Qxh7+) from 28.8% to ~91% (source: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf, Table 4). The paper cautions that auto-interpretation via LLMs is unreliable and that features are easier to label inside a reasoning pathway than in isolation (source: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf, §7 Discussion). See [[fudan-tracing-thought]].

Where the papers diverge: the Fudan paper and the skill-adaptation paper report that well-trained chess SAE/Transcoder features hit near-100% precision and high CC-AUC (source: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf, Table 2; UNDERSTANDING SKILL ADAPTATION IN TRANSFORMERS USING SPARSE AUTOENCODERS.pdf, §5.1), whereas Measuring Progress finds SAEs miss much linearly-encoded board information (reconstruction F1 0.85 vs linear-probe 0.98) and label fork/pin poorly (source: Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models.pdf, §5, Table 4) — so per-feature label reliability is high but coverage of all chess concepts is not yet complete, and linear probes may remain the fallback for concepts SAEs fail to label.

## Relevance to Leela Chat

Feature labeling is the bridge between Phase 4 (Sparse Features) and Phase 5 (Feature Labeling), and it feeds Phase 7 (Reasoning Pathways): an SAE/transcoder firing is useless to the pipeline until it has a validated chess label and a precision/recall number. In the pipeline, labeled features become first-class findings produced by an extractor (a feature ID, its square, its activation strength, its label and validation score), assembled into JSON by a handler, and only then narrated by the translator — the Fudan paper's addressable feature names (Tc.layer.id@square, Lorsa.layer.id@square) map directly onto the pinned BT4 and could become findings-JSON fields (source: Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf, project_relevance). The validator depends on this too: per Rule 1 it can only pass a feature label that is in the firing list, so the label-quality gates (CC-AUC, N2G, coverage, rule-based precision/recall) are exactly the metrics that decide whether a feature is trustworthy enough to expose to the user.

## Related pages

- [[sparse-autoencoders]] — the primary method whose features need labeling
- [[linear-probes]] — alternative concept extraction; the labeling fallback where SAEs miss concepts
- [[reasoning-pathways]] — labeled features become the nodes the LLM narrates as steps
- [[circuit-discovery]] — pathway/circuit construction that consumes labeled features
- [[residual-stream]] — where the features being labeled live
- [[measuring-progress-dictionary-learning]]
- [[scaling-evaluating-saes]]
- [[sparse-feature-circuits]]
- [[contrastive-sae-planning]]
- [[skill-adaptation-saes]]
- [[bridging-human-ai-knowledge-gap]]
- [[fudan-tracing-thought]]
