# Bridging the Human–AI Knowledge Gap: Concept Discovery and Transfer in AlphaZero

**Summary**: Unsupervised discovery of novel AlphaZero chess concepts via sparse convex optimisation and MCTS, then verified teachable to four top grandmasters in a controlled study.

**Sources**: Bridging the Human–AI Knowledge Gap.pdf

**Last updated**: 2026-06-02

---

## Citation
Schut et al.. "Bridging the Human–AI Knowledge Gap: Concept Discovery and Transfer in AlphaZero." arXiv 2023.  arXiv:2310.16410

## Contribution
The paper proposes a framework for extracting new chess concepts from AlphaZero's representational space that are absent from human expert games, then validates that those concepts can be taught to humans. Concept vectors are found via sparse L1-regularised convex optimisation over the policy-value network's latent space, using optimal vs. suboptimal MCTS rollouts to define positive/negative examples for dynamic concepts. Four grandmasters (rated 2600–2800, including Kramnik, Gukesh, Hou Yifan, and Vachier-Lagrave) participated in a three-phase study and all improved their performance on concept puzzles after observing AlphaZero's MCTS calculations. This is the first work to demonstrate measurable human learning from super-human AI-extracted concepts in chess. (source: Bridging the Human–AI Knowledge Gap.pdf)

## Method
Concept vectors are represented as sparse vectors in the post-activation latent space of AlphaZero's ResNet policy-value network (layers 18/19/20/21/23). For static concepts, convex optimisation minimises the L1 norm of the concept vector subject to inner-product margin constraints between positive and negative examples. For dynamic concepts, positive examples are the top MCTS rollout and negatives are suboptimal rollouts at multiple depths (up to T-5), contrasted across the chosen trajectory vs. multiple subpar trajectories. Novelty is assessed via singular value decomposition of latent matrices from AlphaZero vs. human games: concepts with lower reconstruction error under AlphaZero's basis vectors than under human-game basis vectors are accepted. Teachability is measured by distilling concepts to a student network via KL-divergence policy matching and evaluating top-1 move agreement on held-out prototype positions, with prototypes defined as the top 2.5% of positions by concept score. After both filters, 97.6% of concepts are removed by teachability and a further 27.1% by the novelty metric. (source: Bridging the Human–AI Knowledge Gap.pdf, §4.1.2, §4.2.1, §4.2.2)

## Key findings
- AlphaZero's games have a higher-rank latent representation than human games at layers 19 and 23 (AZ rank 8269 vs. human rank 7857 at layer 19; AZ rank 6771 vs. 6544 at layer 23), providing evidence that AZ encodes concepts not present in human games. (source: Bridging the Human–AI Knowledge Gap.pdf, Table 1)
- Sparse concept vectors found via convex optimisation can be learned from as few as 10 labelled examples, achieving near-full-set test accuracy on the piece dataset. (source: Bridging the Human–AI Knowledge Gap.pdf, §5.1.2 / Fig. 6)
- Amplifying a discovered concept vector in AZ's latent representation improves its performance on relevant puzzle sets; higher-quality concept vectors (higher test accuracy) produce larger improvements. (source: Bridging the Human–AI Knowledge Gap.pdf, §5.1.3 / Fig. 7)
- 97.6% of initially discovered concept candidates are filtered out by the teachability criterion (student top-1 policy overlap threshold < 0.2 before teaching); a further 27.1% of remaining concepts are filtered by the novelty metric. (source: Bridging the Human–AI Knowledge Gap.pdf, §4.2.1, §4.2.2)
- All four grandmaster participants improved their concept-puzzle performance after the learning phase, with gains of +6 to +42 percentage points; the magnitude of improvement does not correlate with Elo rating. (source: Bridging the Human–AI Knowledge Gap.pdf, Table 4 / §6.1)
- Dynamic concept discovery uses both the policy-value network activations and MCTS statistics: positive examples are the highest-visit rollout, negatives are suboptimal rollouts at multiple MCTS depths, and the convex optimisation finds a sparse vector satisfying margin constraints across all depth pairs. (source: Bridging the Human–AI Knowledge Gap.pdf, §4.1.2 / Equations 3-5)
- Grandmasters report that AZ's discovered concepts often combine familiar human chess ideas in novel ways — e.g., strategic queen sacrifices, prophylactic manoeuvres, and simultaneous play on both sides — and describe them as "clever", "something new", and "not natural". (source: Bridging the Human–AI Knowledge Gap.pdf, §6.2.1)
- AZ differs from humans in its lack of prior biases over concept relevance: it applies concepts flexibly across the whole board, places less weight on material value, and does not optimise for practical winning chances against an imperfect opponent. (source: Bridging the Human–AI Knowledge Gap.pdf, §6.2.3)

## Relevance to Leela Chat
This paper is directly relevant to **Phase 3 (concept probes)** and **Phase 5 (feature labeling)**. (source: Bridging the Human–AI Knowledge Gap.pdf)

The sparse convex optimisation approach for learning concept vectors from as few as 10 labelled examples provides a lighter-weight alternative to the linear-probe methodology when labelled data is scarce — relevant to the extractor layer when building Lc0-specific concept detectors. (source: Bridging the Human–AI Knowledge Gap.pdf, §5.1.2 / Fig. 6)

The novelty-filtering methodology — comparing SVD basis vectors of Lc0 vs. human-game latent matrices at specific layers — could be adapted to identify which Lc0 features are genuinely novel to Lc0 vs. captured in human chess books, directly informing the **feature-labeling layer (Phase 5)** on which features are most valuable to surface in natural-language responses. (source: Bridging the Human–AI Knowledge Gap.pdf, §4.2.2)

The dynamic concept framework (using MCTS optimal vs. suboptimal rollouts to define positive/negative examples) maps directly onto Leela Chat's `why_this_move` and `did_you_consider` handlers: the rollout contrast is exactly the evidence those handlers already extract from Lc0, and the resulting concept vector could be used to cluster similar positions or label recurring strategic themes across the extractor and translation layers. (source: Bridging the Human–AI Knowledge Gap.pdf, §4.1.2)

The concept amplification experiment (§5.1.3) provides a test harness for verifying that extracted concept vectors are causally active in the network rather than merely correlated, supporting the grounding requirement in Rule 2 of the project's non-negotiables (validated responses only). (source: Bridging the Human–AI Knowledge Gap.pdf, §5.1.3 / Fig. 7)

## Related pages
- [[linear-probes]]
- [[world-models]]
- [[mcts]]
- [[residual-stream]]
- [[superhuman-ai]]
- [[feature-labeling]]
- [[acquisition-chess-knowledge-alphazero]]
- [[contrastive-sae-planning]]
- [[alphazero-hex-evaluation]]
- [[maia-aligning-superhuman-ai]]
- [[improving-chess-commentaries-symbolic]]
