# GAC: Game-Aware Commentary Generation

**Summary**: The neural commentary model proposed by Jhamtani et al. (2018). Extracts Move/Threat/Score features from board state, encodes feature conjunctions with a BiLSTM, and decodes commentary text with an LSTM that has a selection mechanism over the encoded features.

**Sources**: Learning to Generate Move-by-Move Commentary for Chess.pdf

**Last updated**: 2026-05-30

---

## Architecture overview

Four-stage pipeline (source: Learning to Generate Move-by-Move Commentary for Chess.pdf, §3 and Figure 4):

```
[boards R, C, move M]
        ↓
  Feature Extraction      Move features f_move
                          Threat features f_threat   (via python-chess)
                          Score features f_score     (via Stockfish)
        ↓
  Feature Representation  Embed categorical features directly
                          Bin and embed real-valued features
        ↓
  Conjunction Encoder     BiLSTM over canonical-ordered feature embeddings
                          → produces per-position encoding g_i^enc
        ↓
  LSTM Decoder            At each step t:
                            a_t^(i) = dot(W_c · h_t^dec, g_i^enc)
                            α_t = softmax(a_t)
                            c_t = Σ α_t^(i) · g_i^enc
                            p_t = softmax(W_o · [o_t ; c_t])
```

See [[commentary-features]] for what's in each feature group.

## Variants

The paper benchmarks three model variants:

| Variant | Difference |
|---|---|
| **GAC-sparse** | Sparse binary one-hot feature representation, no embeddings. Decoder selection vector is an affine transform of feature concatenation. |
| **GAC** | Embedded features in shared space + BiLSTM conjunction encoder + dot-product attention decoder. The main model. |
| **RAW** | Same architecture as GAC but encodes raw board (piece-name-per-square) instead of extracted features. Tests whether feature engineering matters. |

GAC outperforms both GAC-sparse and RAW on BLEU and diversity (source: Learning to Generate Move-by-Move Commentary for Chess.pdf, Table 3).

## Why the BiLSTM over features?

The authors hypothesize that **feature conjunctions** matter — what "queen attacks rook" + "white to move" + "+2.3 eval swing" *together* mean is more than the sum of parts (source: Learning to Generate Move-by-Move Commentary for Chess.pdf, §3.3).

The BiLSTM encodes the variable-length feature set by sequencing through a canonical ordering. Different orderings produce similar performance. Running multiple encoders over different orderings did not help.

## Category-aware variants

Two extensions explore per-category modeling (source: Learning to Generate Move-by-Move Commentary for Chess.pdf, §4.3):

- **COMB**: Train separate GAC models per category, use the right one at test time. Best per-category metrics.
- **GAC-all**: Train one model over the merged category dataset. Slightly worse than COMB.
- **CAT**: GAC-all but with the category as a one-hot input to the decoder at every step. Better than GAC-all on BLEU-4.

The interesting finding: a classifier trained to *predict* the category from board features alone achieves only 33.4% accuracy (3-way) — almost random. **The board does not determine what kind of comment to make**; that's a pragmatic choice the commentator brings. See [[content-selection-pragmatics]].

## Results

| Subset | Best GAC variant | BLEU | BLEU-2 | Diversity |
|---|---|---|---|---|
| MoveDesc | GAC (M+T) | 1.85 | 23.35 | 4.72 |
| Quality | GAC (M+T+S) | 16.94 | 47.65 | 1.01 |
| Comparative | GAC (M+T) | 3.51 | 29.48 | 3.64 |

(source: Learning to Generate Move-by-Move Commentary for Chess.pdf, Table 3. M=Move, T=Threat, S=Score)

Threat features help across the board. Score features only help on Quality (since Stockfish eval is a direct proxy for move quality).

## Relation to our extractor-translator architecture

GAC is structurally similar to the project's architecture (CHESSGOD/CLAUDE.md §3):

| GAC component | Our analog |
|---|---|
| `f_move`, `f_threat`, `f_score` | Extractors (`leela_chat/extractors/`) |
| BiLSTM conjunction encoder | Handler JSON assembly |
| LSTM decoder + selection | LLM translator (`leela_chat/llm/`) |
| (none) | Output validator (`leela_chat/validator/`) |

The validator is the layer GAC lacks — and it's exactly what makes the hallucination control in our project possible. GAC's outputs are validated only by post-hoc human evaluation; ours are validated by python-chess legality checks and JSON grounding before reaching the user.

## Related pages

- [[learning-to-generate-move-commentary-jhamtani-2018]]
- [[commentary-features]]
- [[content-selection-pragmatics]]
- [[chess-commentary-generation]]
