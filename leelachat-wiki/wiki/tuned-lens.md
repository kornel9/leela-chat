# Eliciting Latent Predictions from Transformers with the Tuned Lens

**Summary**: The tuned lens trains one affine translator per layer to decode intermediate residual-stream states into vocabulary distributions, improving on the brittle logit lens. It addresses systematic bias and representational drift that cause the plain logit lens to fail on models such as BLOOM and GPT-Neo.

**Sources**: Eliciting Latent Predictions from Transformers with the Tuned Lens.pdf

**Last updated**: 2026-06-02

---

## Citation
Belrose et al.. "Eliciting Latent Predictions from Transformers with the Tuned Lens." arXiv 2023.  arXiv:2303.08112

## Contribution
The paper introduces the tuned lens, a technique for decoding transformer hidden states into vocabulary distributions at every layer by training a learned affine transformation (translator) per layer with a KL distillation loss against the final-layer output. This addresses the known brittleness of the plain logit lens, which fails on models like BLOOM and GPT-Neo due to representational drift and systematic bias. The authors show that tuned lens predictions have substantially lower perplexity and are less biased than logit lens predictions. Applications include detecting prompt injection attacks with near-perfect AUROC and measuring example difficulty via prediction depth. (source: Eliciting Latent Predictions from Transformers with the Tuned Lens.pdf)

## Method
For each of the L transformer layers, a small affine transform (A_l, b_l) — the translator — is trained to minimise KL divergence between the tuned lens output and the final-layer logit distribution; the translator is composed with the pretrained unembedding to yield a per-layer vocabulary distribution. Training uses distillation loss and is initialised to the identity. Causal fidelity is evaluated via causal basis extraction (CBE), a greedy orthonormal search for directions with maximal mean-ablation influence on the tuned lens, which are then ablated in the full model to verify stimulus-response alignment. Post-publication experiments with the Muon optimiser dramatically accelerate training and achieve much lower KL divergence than the original SGD-trained lenses. (source: Eliciting Latent Predictions from Transformers with the Tuned Lens.pdf)

## Key findings
- The logit lens is systematically biased, exhibiting ~4–5 bits of KL divergence bias toward certain vocabulary items for most layers of GPT-Neo-2.7B. (source: Eliciting Latent Predictions from Transformers with the Tuned Lens.pdf, Fig. 3)
- The tuned lens trains L affine translators (one per layer) via KL distillation against final-layer logits, correcting both representational drift and the systematic bias of the logit lens. (source: Eliciting Latent Predictions from Transformers with the Tuned Lens.pdf, §3)
- Tuned lens translators can zero-shot transfer to nearby layers with only modest perplexity increase, and transfer well from a base model to its fine-tuned variants (at most 0.3 bits/byte KL increase on RedPajama for LLaMA→Vicuna). (source: Eliciting Latent Predictions from Transformers with the Tuned Lens.pdf, §3, Fig. 13)
- Causal basis extraction finds directions whose influence on the tuned lens correlates strongly with their causal influence on the full model (Spearman ρ=0.89 at layer 18 of Pythia 410M). (source: Eliciting Latent Predictions from Transformers with the Tuned Lens.pdf, §4.1, Fig. 8)
- Stimulus-response alignment is higher at later layers and higher when using principal subspaces defined by the tuned lens rather than the logit lens, validating that the tuned lens finds causally relevant features. (source: Eliciting Latent Predictions from Transformers with the Tuned Lens.pdf, §4.2, Fig. 9)
- Prediction trajectories from the tuned lens can detect prompt injection attacks with near-perfect AUROC on five of nine NLP tasks tested (BoolQ, MNLI, QNLI, QQP, SST-2 all ≥0.99 with LOF). (source: Eliciting Latent Predictions from Transformers with the Tuned Lens.pdf, §5.3, Table 1)
- Tuned lens prediction depth correlates significantly with example difficulty (iteration learned) in all 11 tasks, outperforming the logit lens in 8 of 11 tasks. (source: Eliciting Latent Predictions from Transformers with the Tuned Lens.pdf, §5.4, Table 2)
- Using the Muon optimiser instead of SGD for translator training dramatically accelerates training and achieves much lower KL divergence; original published lenses were severely undertrained. (source: Eliciting Latent Predictions from Transformers with the Tuned Lens.pdf, §3 (Muon optimizer subsection))
- Transformers perform iterative inference in a formal sense: residuals align with the negative gradient at every layer of Pythia 6.9B at least 95% of the time, supporting the per-layer latent prediction view. (source: Eliciting Latent Predictions from Transformers with the Tuned Lens.pdf, Appendix C, Fig. 19)

## Relevance to Leela Chat
For Leela Chat Phase 1, the tuned lens offers a principled way to visualise how Lc0's BT4 transformer builds up its move evaluation layer by layer — each intermediate hidden state can be decoded into a policy/value distribution, providing richer signal for the extractor layer than just the final network output. (source: Eliciting Latent Predictions from Transformers with the Tuned Lens.pdf)

In Phase 3 (linear concept probes), the tuned lens methodology directly informs probe design: training affine translators per layer with distillation loss rather than ground-truth labels avoids the leakage problem that Hewitt & Liang warn about, which is directly applicable to Leela's concept probes. (source: Eliciting Latent Predictions from Transformers with the Tuned Lens.pdf)

The causal basis extraction (CBE) algorithm provides a concrete method for identifying which [[residual-stream]] directions are causally relevant to a particular prediction — useful in Phase 6 for differential/counterfactual analysis without requiring nnsight or full [[activation-patching]] infrastructure. (source: Eliciting Latent Predictions from Transformers with the Tuned Lens.pdf)

Important caveat: BT4 uses smolgen attention (not vanilla transformer attention), so [[logit-lens]] math differs per CLAUDE.md §8; any tuned lens applied to BT4 must account for the non-standard attention policy head before per-layer distributions can be interpreted as policy/value readouts.

## Related pages
- [[logit-lens]]
- [[residual-stream]]
- [[linear-probes]]
- [[activation-patching]]
- [[iterative-inference-chess]]
- [[limits-of-probes]]
- [[activation-patching-guide]]
- [[sparse-feature-circuits]]
- [[hydra-effect]]
