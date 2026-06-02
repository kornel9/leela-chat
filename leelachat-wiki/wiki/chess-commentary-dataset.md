# Chess Commentary Dataset (gameknot.com)

**Summary**: 298K move/commentary pairs across 11.5K chess games scraped from the gameknot.com online forum, where players self-annotate their games with move-by-move commentary. First large-scale chess commentary dataset (Jhamtani et al., 2018).

**Sources**: Learning to Generate Move-by-Move Commentary for Chess.pdf

**Last updated**: 2026-05-30

---

## Statistics

| Metric | Value |
|---|---|
| Total games | 11,578 |
| Total moves with comments | 298,008 |
| Avg. annotated moves per game | 25.73 |
| Frequent word types | 39,424 |
| Rare word types | 167,321 |
| Word tokens | 6,125,921 |
| Unigram entropy | 6.88 |
| Avg. comment length (words) | 20.55 |
| Long comments (>5 words) | 230,745 (77%) |

(source: Learning to Generate Move-by-Move Commentary for Chess.pdf, Table 1)

## Source

Forum-scraped from gameknot.com. Each game on the site can have move-level annotations from the player who played it, written in natural English. Some annotations cover a single move; others span a sequence of moves (e.g., 5 consecutive moves under one comment).

Jhamtani et al. release the dataset and code at https://github.com/harsh19/ChessCommentaryGeneration.

## Preprocessing decisions

- Multi-move annotations are dropped for the initial modeling work — only single-move comments are kept.
- Multi-sentence single-move comments are split into separate datapoints sharing the same board/move input.
- A 1K sample is hand-labeled with one of six categories ([[commentary-categories]]); SVM classifiers tagged the rest.

## Train/val/test split

70 / 10 / 20 within each commentary category subset (source: Learning to Generate Move-by-Move Commentary for Chess.pdf, §4).

## Subset sizes (after category tagging)

- Move Description: 28,228
- Comparative: 5,397
- Move Quality: 793

(Quality subset is small enough that 5-fold cross-validation is used instead of a single split.)

## Caveats and limitations

- **Forum quality.** Self-annotated forum games vary wildly in commenter skill — these are not master-level annotations like Informant or ChessBase.
- **Ground-truth contamination.** Jhamtani's human study found ~30% of supposed move-description comments did not actually describe the move alone — commenters smuggled in broader game context. Anyone using this corpus as supervision inherits that noise. (source: Learning to Generate Move-by-Move Commentary for Chess.pdf, §4.5)
- **Style noise.** A meaningful fraction are emotional remarks ("Protect Calvin, Hobbs"), addressed to an opponent, not chess content.
- **English only.**

## Usefulness for Leela Chat

This project does **not** use the gameknot corpus as supervision — we ground LLM outputs in extracted engine/network signals rather than imitate forum prose (CHESSGOD/CLAUDE.md §2 Rule 1). The corpus may still be useful as:

- A source of evaluation prompts (what kinds of moves do humans ask about?).
- A linguistic style reference for prompt design — what phrasing conventions are natural for chess commentary.
- A counter-example: forum prose is exactly the kind of confabulation-prone output we must *not* produce.

## Related pages

- [[learning-to-generate-move-commentary-jhamtani-2018]]
- [[commentary-categories]]
- [[chess-commentary-generation]]
