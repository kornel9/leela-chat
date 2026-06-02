# Content Selection and Pragmatics in Chess Commentary

**Summary**: The choice of *what to say* about a move is not determined by the position alone — it is a pragmatic choice that depends on context, audience, and commentator style. This is the central NLG challenge in chess commentary.

**Sources**: Learning to Generate Move-by-Move Commentary for Chess.pdf

**Last updated**: 2026-05-30

---

## The problem

For any given move, there are multiple true things to say:

> *"For the move shown in Figure 1, one can comment simply that the pawn was moved, or one may comment on how the check was blocked by that move. Both descriptions are true, but the latter is most salient given the player's goal."* — Jhamtani et al., §1

The system must:

1. **Decide what aspect to comment on** (mechanics, threat, plan, comparison, evaluation).
2. **Render that aspect in fluent prose**.

These are separable problems. Most prior NLG work (recipes, biographies) collapses them because there's only one thing to say. Chess commentary forces the split.

## Empirical evidence that selection is pragmatic, not determined

Jhamtani trains a classifier to predict the commentary **category** ([[commentary-categories]]) from board features (Move + Threat + Score). It achieves 33.4% on a 3-way task — essentially random (source: Learning to Generate Move-by-Move Commentary for Chess.pdf, §4.3).

This means: **the position does not determine what kind of comment is salient**. The same position licenses commentary on mechanics, on plan, or on alternatives — equally well. The choice is a function of the commentator's communicative goal, not of the board.

## Implication for system design

If selection is genuinely pragmatic, a single-output commentary system is *underdetermined*: there's no ground-truth answer to "what should the system say about this move." Two correct responses can both be valid.

Two design responses to this:

**Path A (Jhamtani):** Let the model pick implicitly. Train end-to-end and let the attention/selection mechanism learn what to emphasize. Accept that BLEU will be low because reference variance is high. See [[bleu-limitations-nlg]].

**Path B (Leela Chat):** Let the **question** drive selection. The user asks "why did Leela pick this?" or "what was the second-best move?" — the question makes the pragmatic choice explicit, and the system has a well-defined answer. The classifier becomes the conversation router.

The Leela Chat path is downstream of the framing decision: this isn't a monologue commentary generator, it's a conversational interface. The pragmatic-selection problem is dissolved by making the user state their pragmatic intent.

## Caveat — when the question is ambiguous

If the user asks "what's going on here?" with no specific framing, the pragmatic problem returns. The classifier must guess at intent. Strategies:

- Default to multi-aspect output (describe + evaluate + plan briefly, in fixed order).
- Ask a clarifying question back to the user — at the conversation layer, not the LLM layer.
- Surface the top finding by salience score (e.g., highest-magnitude feature firing).

These belong in the classifier/conversation manager layer, not the extractors.

## Related pages

- [[learning-to-generate-move-commentary-jhamtani-2018]]
- [[commentary-categories]]
- [[chess-commentary-generation]]
- [[bleu-limitations-nlg]]
