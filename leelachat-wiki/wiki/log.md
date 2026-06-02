# Wiki Operations Log

Append-only record of wiki edits.

---

## 2026-05-30 — Initial ingest: Jhamtani et al. 2018

**Source**: `raw/Learning to Generate Move-by-Move Commentary for Chess.pdf`

**Pages created** (10):

- `learning-to-generate-move-commentary-jhamtani-2018.md` — source summary
- `chess-commentary-generation.md` — task overview
- `chess-commentary-dataset.md` — gameknot.com dataset
- `gac-model.md` — GAC neural architecture
- `commentary-categories.md` — six-class commentary taxonomy
- `commentary-features.md` — Move/Threat/Score features
- `content-selection-pragmatics.md` — pragmatic content selection
- `bleu-limitations-nlg.md` — why BLEU fails here
- `python-chess.md` — tool reference
- `stockfish.md` — tool reference

**Index**: created `index.md` with all pages linked.

**Notes**:
- This is the first source ingested; wiki was previously empty.
- The `raw/` folder contains ~32 additional PDFs (other papers in the project's reading list) that have not yet been ingested.
- All citations use `(source: filename.pdf)` format per CLAUDE.md.
- Cross-references between pages use `[[...]]` wiki-link syntax.

---

## 2026-06-02 — Bulk ingest: 28 reading-list papers + the Fudan North Star

**Sources**: the 28 remaining `raw/*.pdf` reading-list papers, plus the newly added `raw/Tracing the Thought of a Grandmaster-level Chess-Playing Transformer.pdf` (Lin et al., Fudan, arXiv 2604.10158 — fetched from arXiv this session).

**Method**: two multi-agent workflows. (1) Parallel extraction — one reader per paper over the PDF text layer (extracted with `pypdf`, since the Read tool cannot render PDFs in this environment) → structured, source-cited findings JSON. (2) Parallel synthesis — one writer per note.

**Pages created** (47):
- 29 paper notes (one per source).
- 18 shared concept pages: `residual-stream`, `world-models`, `board-state-probing`, `linear-probes`, `sparse-autoencoders`, `activation-patching`, `path-patching`, `circuit-discovery`, `learned-look-ahead`, `logit-lens`, `attention-mechanism`, `feature-labeling`, `transformer-chess-architecture`, `mcts`, `maia-human-alignment`, `self-repair`, `superhuman-ai`, `reasoning-pathways`.

**Maintenance**:
- Deduped `raw/`: removed 3 MD5-confirmed byte-identical duplicates (Monroe ×2 extra copies, Measuring-Progress ×1 extra). Kept the two *differing* "Contrastive SAE" files (different bytes — two versions of Poupart's paper).
- Added `leelachat-wiki/raw/` and `leelachat-wiki/.obsidian/` to the repo `.gitignore` (raw/ duplicates the already-tracked `update_research/pdfs/`).
- Fixed a slug collision: the Goldowsky-Dill paper note and the `path-patching` *concept* page both targeted `path-patching.md`; the paper note now lives at `localizing-model-behavior-path-patching.md`, with `path-patching.md` reserved for the concept.
- Folded `concept-probing` links into `[[linear-probes]]`.

**Index**: rewrote `index.md` with thematic grouping (papers by area, concept pages, commentary cluster, tools).

**Notes**:
- Every claim cites its source PDF with a section/table/figure locator. Notes are LLM-synthesised secondary summaries — verify any load-bearing number against the source PDF.
- Link-integrity check after this entry: 0 broken links; `index.md` is the intentional root (the only orphan, by design).

---

## 2026-06-02 — Repo ingest: 28 vendored source repos

**Sources**: the 28 readable repos under `update_research/repos/`. `python-chess` and `stockfish` were folded into their existing tool pages (a "Vendored source" subsection), not given new notes. Two repos were re-fetched fresh from GitHub to replace corrupt zips and vendored as full source: `feature-circuits` (saprmarks, MIT, @7fbd82b) and `lczero-training` (LeelaChessZero, no LICENSE file, @a60c7d2).

**Method**: two-phase Workflow mirroring the paper ingest — parallel extract (one reader per repo over README + top-level `ls` + LICENSE + its PRIOR_WORK line range → findings JSON), then parallel synthesize (one writer per note). A mid-run credit exhaustion truncated the first synthesize pass; resumed by re-running only the missing notes (the write-time no-overwrite guard protected the completed ones).

**Pages created** (28): lc0, lczerolens, leela-interp, leela-logit-lens, lczero-planning, leela-saes, sae-boardgameeval, acdc-repo, tuned-lens-repo, maia-chess, chess-commentary-generation-repo, learning-chess-blindfolded, alphatology, chess-llm-interpretability, alphasae, superhuman-ai-consistency, transformerlens, nnsight, saelens, sparsify, openai-sparse-autoencoder, zennit, llamascopium, cosmic, chess-tutor, awesome-sparse-autoencoders, feature-circuits, lczero-training.
**Pages edited**: python-chess + stockfish (Vendored source subsection); index.md (new "Repositories & vendored source" section).

**Provenance**: each note carries `**Vendored at**` (commit/tag; pre-existing snapshots without `.git` say "ref not recorded") and `**Verified**: README-derived, not run`. Adopt/skip lines summarise and cite PRIOR_WORK (canonical).

**Deconfliction**: 4 slugs decorated to avoid colliding with existing notes — `tuned-lens-repo`, `acdc-repo`, `chess-commentary-generation-repo`, `openai-sparse-autoencoder`. No file overwritten.

---

## 2026-06-02 — Research-gap fill: 14 resources + 5 concept pages

**Sources**: 14 resources found absent from `docs/PRIOR_WORK.md` in a corpus audit — fetched from arXiv (12 PDFs) + transformer-circuits.pub (Circuit Tracing / Biology) + 4 companion repos (circuit-tracer, delphi, maia2, searchless_chess). Each note's `**Source**` field records the arXiv id + commit.

**Pages created** (14 resource notes): transcoders-find-circuits, attention-saes-kissane, saebench, feature-absorption-matryoshka, gemma-scope, auto-interpretability, tcav, amnesic-probing, cot-faithfulness, constrained-decoding, maia-2, open-problems-mech-interp, circuit-tracing, searchless-chess.
**New concept pages** (5): transcoders, attention-saes, faithfulness, auto-interp, concept-vectors.

**docs/PRIOR_WORK.md**: added a "Research & tooling — gap-fill additions (2026-06-02)" section (additive; no entry deleted) with ✓ markers (sources read this session), licenses for the code, phase mapping, and a `→ wiki:<slug>` pointer per item. Maia-2, Gemma Scope, and attention-SAEs/Lorsa promoted from incidental mentions to first-class entries.

**CLAUDE.md**: §10 Phase Entry Protocol now says to open the relevant wiki notes (navigation into PRIOR_WORK); §12 adds a pointer to the vault. The wiki is a navigation layer; PRIOR_WORK stays canonical for adopt/skip.

**Notes**: resource notes are LLM-synthesised secondary summaries — `Verified:` states confidence; verify load-bearing numbers against the source. Faithfulness + constrained-decoding directly underpin Rule 1 and the output validator.
