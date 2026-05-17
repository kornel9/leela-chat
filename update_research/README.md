# `update_research/` — reference materials for Claude Code

This folder holds the **full text of papers and the source of reference repos** cited throughout the project. The purpose is to give Claude Code (and any future collaborator) deeper context when an algorithm derivation, paper methodology, original repo implementation detail, or design rationale isn't fully captured by `docs/PRIOR_WORK.md` or by the project's own code.

`docs/PRIOR_WORK.md` is the **curated index** — "which paper is relevant for what, organized by tier and phase." This folder is the **actual files** of those references.

## Structure

```
update_research/
├── README.md     # this file
├── pdfs/         # paper PDFs
└── repos/        # full source of relevant repos (no .git/ inside)
```

## Naming conventions

**PDFs** (`pdfs/`): `lastname_year_short-title.pdf` or `arxiv-NNNN.NNNNN_short-title.pdf`. Examples:

- `lin_2024_tracing-the-thought.pdf`
- `arxiv-2604.10158_tracing-the-thought.pdf`
- `mcgrath_2022_alphazero-knowledge.pdf`

Whichever form makes the file easy to find when grepping for a paper name. Pick one form per paper and stick with it.

**Repos** (`repos/`): keep the project's natural name. Strip any `-main` / `-master` suffix that GitHub zip downloads add if it's distracting (the `Leela-SAEs-main` snapshot keeps the suffix only because we already vendored from that path).

## Hygiene rules

- **Never commit `.git/` folders inside `repos/<X>/`.** If you `git clone <url> repos/<X>`, delete the resulting `repos/<X>/.git/` directory before staging. Otherwise you're embedding another project's full git history inside ours, which bloats this repo forever and can't be cleanly removed.
  - Prefer downloading as a zip from the GitHub repo's "Code → Download ZIP" button — that gives you a clean tree with no embedded `.git/`.
  - Or use `git clone --depth 1 <url> repos/<X>` then `rm -rf repos/<X>/.git/`.
- **Watch single-file size.** GitHub rejects files over 100 MB without Git LFS. Most paper PDFs are 1-10 MB; some textbooks or NeurIPS slidedecks are larger. If a file is huge, store it elsewhere and add a `URL.txt` placeholder linking to the source.
- **Datasets and model weights do NOT belong here.** Those go in `data/` and `weights/` respectively (both gitignored). This folder is for human-readable reference material — papers and source code — not training data or binary blobs.
- **Don't strip files from a vendored repo.** If you copy `repos/<X>/`, copy the whole thing (with the LICENSE file). Trimming makes future re-comparison against upstream harder.

## How Claude Code uses this

See CLAUDE.md §12 "Pointers to Deeper Context" — the rule is: consult `update_research/` when the project's own code + `docs/PRIOR_WORK.md` don't go deep enough. Read **selectively** (abstract / intro / a specific section for papers; README + the relevant module for repos). Avoid reading entire papers when a targeted section would do — context window is finite, and most chunks of most papers aren't load-bearing for the current task.

`docs/PRIOR_WORK.md` should always be the entry point — it tells you which paper is relevant for what. Only then descend into `update_research/` if you need the actual content.

## Known issues

Two zip files in `repos/` are **corrupted downloads** that failed to extract (`End-of-central-directory signature not found`). They've been kept on disk so you remember to replace them; re-download from GitHub when needed:

- `repos/feature-circuits-main.zip` — Marks et al., "Sparse Feature Circuits" (Phase 7 reference). Re-download: <https://github.com/saprmarks/feature-circuits>.
- `repos/lczero-training-master.zip` — Lc0 network training pipeline (89 MB; we don't retrain in this project, so reference-only). Re-download: <https://github.com/LeelaChessZero/lczero-training>.

Until they're replaced, those zips don't contribute usable content. You can ignore them when looking for material to read.
