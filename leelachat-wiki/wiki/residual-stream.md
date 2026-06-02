# Residual stream

**Summary**: The residual stream is the running sum of vector representations that flows through a transformer's layers — the central "highway" each attention and MLP block reads from and writes to. In chess networks like Lc0/BT4 it holds, per board-square token, the activations that probes decode into concepts and that patching interventions edit to test causation.

**Sources**: Eliciting Latent Predictions from Transformers with the Tuned Lens.pdf, Iterative Inference in a Chess-Playing Neural Network.pdf, Acquisition of Chess Knowledge in AlphaZero.pdf, Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models.pdf, Evidence of Learned Look-Ahead.pdf, Sparse Feature Circuits.pdf, Mastering Chess with a Transformer Model.pdf, How to Use and Interpret Activation Patching.pdf

**Last updated**: 2026-06-02

---

## What it is
The residual stream is the per-token vector that each transformer sublayer adds its output into; later layers read it, transform it, and write back, so it accumulates the network's evolving representation of the position. For interpreting Lc0/BT4 it matters because intermediate residual-stream states (one per board square) are what linear probes read out as chess concepts, what the logit/tuned lens decodes into move distributions, and what activation patching swaps to test which representations are causally load-bearing. Crucially, the residual stream adds to the output logits roughly linearly, which is why it is the natural unit for both probing and patching (source: How to Use and Interpret Activation Patching.pdf, §4.1, §5). In BT4 the residual stream is 1024-dimensional across 15 layers with 32 heads (source: Mastering Chess with a Transformer Model.pdf, Notes).

## In the literature
- **Tuned lens** trains one affine "translator" per layer to decode the intermediate residual-stream state into a vocabulary (here, output) distribution, fixing the brittleness and ~4–5 bit systematic bias of the plain logit lens; it also notes transformers perform iterative inference, with residuals aligning to the negative gradient at every layer ≥95% of the time in Pythia 6.9B (source: Eliciting Latent Predictions from Transformers with the Tuned Lens.pdf, §3 / Fig. 3 / Appendix C, Fig. 19). See [[tuned-lens]].
- **Iterative inference in chess** applies an extended (Post-LN) logit lens to Leela's policy residual stream across all 15 layers and finds move preferences stabilize only late: Kendall's τ between intermediate and final rankings stays near zero through the middle phase and rises sharply around layer 12, so extractors reading the residual stream should target late layers (source: Iterative Inference in a Chess-Playing Neural Network.pdf, §3.2 and Fig. 7). See [[iterative-inference-chess]].
- **Acquisition of chess knowledge in AlphaZero** probes intermediate activations (a ResNet, not a transformer residual stream) and shows human concepts are linearly decodable — e.g. Stockfish total score regressed with r² > 0.75 from layer-10 activations — but warns that some concepts (threats, imbalance) peak at mid-network layers and decline later, so probing only the final layer misses them (source: Acquisition of Chess Knowledge in AlphaZero.pdf, §4.3.2, Fig. 2a/d/g/h). See [[acquisition-chess-knowledge-alphazero]].
- **Emergent world models in chess** trains per-square linear probes on a chess-GPT residual stream and recovers board state at up to 99.6% per-square accuracy (16-layer model, peaking at layer 12); subtracting a piece's probe direction from the residual stream causally erases that piece, raising the modified-board legal-move rate from 40.5% to 92% (source: Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models.pdf, Table 1 / §4.2, Table 2). See [[emergent-world-models-chess]].
- **Evidence of learned look-ahead** patches Leela's policy residual stream at the square level and finds future-move information is causally concentrated: patching the third-move target square at layer 10 drops correct-move log odds by 1.88 ± 0.04 versus at most 0.55 ± 0.01 for any other square, and a bilinear probe reads the two-moves-ahead target from the layer-12 residual stream at 92 ± 1% accuracy (source: Evidence of Learned Look-Ahead.pdf, §3.1 / §3.3, Fig. 8). See [[evidence-learned-look-ahead]].
- **Sparse feature circuits** treats SAE feature activations (and SAE error terms) on each sublayer — including the residual stream — as graph nodes and scores their causal importance with attribution patching, needing far fewer feature nodes than raw neurons (~100 vs ~1500 on Pythia-70M) to explain a behavior (source: Sparse Feature Circuits.pdf, §2 / Fig. 3). See [[sparse-feature-circuits]].
- **Mastering chess with a transformer (Chessformer/CF-240M)** is the architecture BT4 follows (15 layers, 1024 embedding, 32 heads) and shows the residual stream's content per token is shaped by Shaw relative-position encoding; the policy head is itself a 64×64 self-attention traversal, so policy logits are NOT a simple linear projection from the residual stream — a logit-lens-on-policy approach is therefore unsafe here (source: Mastering Chess with a Transformer Model.pdf, Appendix A-C / §V). See [[mastering-chess-transformer-monroe]].
- **Activation patching guide** establishes the residual stream as the coarsest granularity for patching (residual → MLP/attention output → neuron → SAE feature) and argues logit difference is the right metric because the residual stream adds to logits roughly linearly; it also flags the Hydra/backup effect, where compensating components recover ~0.7× of an ablated component's effect (source: How to Use and Interpret Activation Patching.pdf, §4.1 / §3.2). See [[activation-patching-guide]].

## Relevance to Leela Chat
The residual stream is the substrate the extractor layer reads from in Phase 1 (network-first chat) and Phase 3 (linear concept probes): per-square BT4 activations are hooked, decoded by probes or a tuned/logit lens, and emitted as grounded findings for the translator. Because chess networks commit late (source: Iterative Inference in a Chess-Playing Neural Network.pdf, §3.2 and Fig. 7) yet some concepts peak mid-network (source: Acquisition of Chess Knowledge in AlphaZero.pdf, §4.3.2), extractors must expose layer-selectable hooks rather than reading only the final layer. In Phase 6 (differential/counterfactual) and Phase 7 (reasoning pathways), patching or feature-ablating residual-stream activations is how Leela Chat tests which representations cause a move preference — with the Hydra-effect caveat (source: How to Use and Interpret Activation Patching.pdf, §3.2) feeding the validator/handler discipline that a single ablation may understate importance.

## Related pages
- [[logit-lens]]
- [[tuned-lens]]
- [[linear-probes]]
- [[board-state-probing]]
- [[world-models]]
- [[activation-patching]]
- [[sparse-autoencoders]]
- [[circuit-discovery]]
- [[learned-look-ahead]]
- [[attention-mechanism]]
- [[transformer-chess-architecture]]
- [[iterative-inference-chess]]
- [[acquisition-chess-knowledge-alphazero]]
- [[emergent-world-models-chess]]
- [[evidence-learned-look-ahead]]
- [[sparse-feature-circuits]]
- [[mastering-chess-transformer-monroe]]
- [[activation-patching-guide]]
