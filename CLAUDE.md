# CLAUDE.md — Leela Chat

> This file tells Claude Code how to work in this repository. It is read at the start of every session. Strategic context (the "why") lives in `docs/BIBLE.md`. Research and asset references live in `docs/PRIOR_WORK.md`. This file is the "how."
>
> **If you (Claude Code) are about to act on something that contradicts this file, stop and ask.** Drift from these rules is how this project dies.

---

## 1. Project Identity

**What this is:** A conversational interface to Leela Chess Zero. The user loads a position (FEN/PGN) and asks natural-language questions. The system extracts signals from Lc0's MCTS tree and network internals (residual stream, attention, sparse features, probes), then translates those signals to prose via a frontier LLM. Every chess fact in every response traces back to a number Leela produced. The LLM is a translator, never an inventor.

**What this is not:** a chess engine, a UI prototype, a fine-tuned chess LLM, an opening trainer, a tournament tool, a chatbot. Read `docs/BIBLE.md` sections "In scope," "Out of scope," and "Anti-goals" if tempted to add scope.

**Target user:** the project owner (a 2000+ player) for personal use, then strong-player friends, then maybe a public alpha. Quality bar: 2200+ player should approve. GM utility is aspirational, not v1.

**Current phase:** `Phase 1 — Network-First Chat MVP`. Track current phase in `docs/PHASE_NOTES.md`. Update at the start of every work session.

---

## 2. The Three Non-Negotiable Rules

These are the rules that make the project work. Breaking any of them is a regression even if tests pass.

### Rule 1: The LLM never invents chess facts.

The LLM is a translator. It receives a JSON object of extracted findings and writes prose about them. It does NOT:
- Invent moves not in the JSON
- Invent evaluations not in the JSON
- Invent feature/probe firings not in the JSON
- Speculate about lines not given to it
- Use general chess knowledge to fill gaps

If the JSON doesn't contain enough information to answer, the correct LLM response is "I'd need to analyze that — let me run another extractor" (which triggers another extractor call), OR a direct "I don't have that information." Never a confabulation.

### Rule 2: Every response is validated before reaching the user.

The output validator runs on every LLM response. It extracts every move-like string and verifies legality via python-chess. It extracts every eval number and checks it appears in the JSON within tolerance. It extracts every feature label and checks it's in the firing list. If validation fails:
1. Retry with stricter prompt
2. If retry fails, fall back to templated response built from the JSON directly
3. Log the failure
4. Never show an unvalidated response to the user

### Rule 3: The eval set is sacred.

Before ANY change to extraction, translation, or validation code is committed, the full eval set runs and per-handler scores are compared to baseline. A drop of >5% on any handler is a regression — investigate before merging. The eval set lives in `tests/eval/eval_positions.yaml`. Add positions to it, never remove.

**Corollary:** Build the eval set BEFORE the code it evaluates. Don't write a handler without first writing eval entries for it.

---

## 3. Architecture Invariants

This is the contract between layers. Do not violate without updating the bible and writing a `docs/DECISIONS.md` entry.

```
USER → CONVERSATION MANAGER → CLASSIFIER → HANDLER → EXTRACTOR(S) → LEELA → JSON
                                                                              ↓
                                            VALIDATED RESPONSE ← LLM ← PROMPT + JSON
```

**Layer rules:**

- **Extractors return only facts.** No prose, no opinions, no interpretation. Numbers, FENs, UCI moves, feature IDs, activation strengths. The return type is always a Pydantic model.
- **Handlers orchestrate extractors and assemble JSON for the LLM.** They do not contain chess logic. They do not call the LLM directly — they return the findings JSON.
- **The translation layer is the only place the LLM is called.** It takes JSON + question + history + prompt template version. It returns prose. Its output is always validated before returning.
- **The validator is between the LLM and the user.** Nothing bypasses it.
- **Logging is mandatory at every step.** Every analysis writes a JSONL line with timestamp, FEN, question, handler called, full JSON findings, LLM prompt version, raw LLM response, validation result, final response, latency.

**File-level conventions:**
- One handler per file under `leela_chat/handlers/`
- One extractor per logical operation under `leela_chat/extractors/`
- Per-handler LLM prompts versioned at `leela_chat/llm/prompts/{handler}/v{N}.md` — never delete old versions, always increment
- Pydantic models live next to the code that uses them, not in a central `types.py`

---

## 4. Pinned Versions and Foundational Choices

**Do not change these without a DECISIONS.md entry explaining why.** Changing the network invalidates probes, SAEs, and labels — a multi-month rollback.

| Item | Pinned value |
|---|---|
| Python | 3.11.* |
| Package manager | `uv` |
| Lc0 network | `BT4-1024x15x32h-swa-6147500` (hash recorded in `docs/DECISIONS.md`) |
| Lc0 version | tagged release recorded in `docs/DECISIONS.md` |
| Pytorch | matches CUDA driver version, pinned in `pyproject.toml` |
| lczerolens | pinned in `pyproject.toml`; v0.4.0+ |
| LLM provider | Anthropic Claude — `claude-opus-4-7` (eval/prod) + `claude-sonnet-4-6` (iteration); OpenAI as fallback. See ADR-0005. |
| Logging | `loguru` |
| Data validation | `pydantic` |
| Tests | `pytest` + `pytest-asyncio` |

**If asked to switch networks, models, or major dependencies, stop and ask the user first.** This is a common error mode and almost always wrong.

---

## 5. Repository Layout

```
leela-chat/
├── CLAUDE.md                   # This file
├── README.md                   # Short project intro for humans
├── pyproject.toml              # uv-managed, pinned versions
├── .python-version             # 3.11
├── .gitignore                  # excludes /data, /weights, /logs, .env
├── .env.example                # API key template
├── leela_chat/                 # Main package
│   ├── engine/                 # Lc0/Stockfish UCI wrappers, MCTS tree parsing
│   ├── network/                # lczerolens wrappers, forward-pass hooks
│   ├── features/               # SAE loading, feature extraction (Phase 4+)
│   ├── probes/                 # Linear probes (Phase 3+)
│   ├── labels/                 # Feature label DB (Phase 5+)
│   ├── extractors/             # Signal extraction functions
│   ├── handlers/               # Question handlers (one file each)
│   ├── classifier/             # Question → handler routing
│   ├── llm/                    # LLM client, prompts/, translation
│   ├── validator/              # Output validation (legality + JSON grounding)
│   ├── conversation/           # Session state, history
│   ├── eval/                   # Eval harness
│   └── cli/                    # CLI entry point
├── tests/
│   ├── unit/                   # Per-extractor and per-handler unit tests
│   ├── integration/            # End-to-end tests
│   └── eval/
│       └── eval_positions.yaml # THE eval set — protected
├── scripts/                    # One-off data prep, probe training scripts
├── notebooks/                  # EXPLORATION ONLY — never put logic here
├── data/                       # gitignored
├── weights/                    # gitignored — Lc0 nets, SAE weights, probes
├── labels/                     # versioned text — feature label DB
├── logs/                       # gitignored — every analysis logged
├── update_research/            # Full text of cited papers + source of reference repos
│   ├── README.md               # — purpose, structure, hygiene rules
│   ├── pdfs/                   # — paper PDFs (`lastname_year_short-title.pdf`)
│   └── repos/                  # — full source of reference repos (no .git/ inside)
└── docs/
    ├── BIBLE.md                # Strategic vision (read for "why")
    ├── PRIOR_WORK.md           # Papers, repos, datasets reference (curated index)
    ├── DECISIONS.md            # Architecture decision records
    └── PHASE_NOTES.md          # Per-phase notes, current phase tracker
```

**When creating new code:** match this structure exactly. If a new module doesn't fit, ask before adding a new top-level directory.

**Notebooks are for exploration only.** Any logic worth keeping moves to a `.py` file with tests. Do not import notebook code from production.

---

## 6. Code Conventions

### Python style
- Type hints on every function signature
- Docstrings (Google or NumPy style) on every public function and class
- `from __future__ import annotations` at the top of every module
- Prefer `dataclass` or `pydantic.BaseModel` over dicts for structured data
- Async only where it actually helps (LLM calls, parallel engine calls); never speculatively

### Pydantic for layer contracts
- Every extractor returns a Pydantic model, never a raw dict
- Every handler returns a Pydantic model containing the JSON findings
- Models live next to the code that produces them
- Use `model_config = ConfigDict(frozen=True)` for value-like models

### Error handling
- Catch errors at the boundary (CLI, LLM client, subprocess) — let them propagate within a layer
- Never silently swallow errors — log at WARNING or ERROR, then either reraise or return a sentinel
- LLM and validator failures fall back to templated responses, not exceptions to the user

### Logging
- `loguru` everywhere; configure once in `leela_chat/__init__.py`
- Every extractor call logs the FEN and timing at DEBUG
- Every LLM call logs token counts at INFO
- Every validation failure logs at WARNING
- Every analysis writes a structured JSONL line

### Testing
- Every extractor has at least one unit test with a known FEN and expected output
- Every handler has at least one integration test that mocks the LLM
- Eval set runs are integration tests, but separated under `tests/eval/`
- Test data lives next to tests, not in `data/`

---

## 7. Commands

The CLI and dev scripts use `uv` everywhere.

```bash
# Install dependencies
uv sync

# Run the REPL CLI
uv run python -m leela_chat.cli

# Run all unit tests
uv run pytest tests/unit

# Run integration tests (slower, exercises engine)
uv run pytest tests/integration

# Run the eval set
uv run python -m leela_chat.eval.runner

# Compare current eval to baseline (CI-style check)
uv run python -m leela_chat.eval.regression --baseline logs/eval/baseline.json

# Train a probe for a concept (Phase 3+)
uv run python -m leela_chat.probes.train --concept knight_outpost

# Smoke test the engine and network wrappers
uv run python -m leela_chat.scripts.smoke_test
```

**Before committing any change to `leela_chat/extractors/`, `leela_chat/handlers/`, `leela_chat/llm/`, or `leela_chat/validator/`:** run the full eval set. If per-handler score drops >5%, do not commit.

---

## 8. Common Pitfalls — Stop and Ask Before

This list grows as the project finds new ways to fail. Check it before doing anything that sounds like one of these.

### Modeling pitfalls
- Switching the Lc0 network version
- Switching the LLM model in production
- Adding a new SAE layer without retraining max-activation cache
- Changing the board encoding (history planes, side-to-move) — silently breaks every probe and SAE
- Changing the 1858-element move vector mapping

### Architecture pitfalls
- Letting handlers contain chess logic (extractors do that)
- Letting extractors call the LLM (only the translation layer does)
- Bypassing the validator
- Adding a new question handler before existing ones pass eval
- Building a web UI before the engine pipeline works

### Phase-creep pitfalls
- Doing Phase 2 work (MCTS tree parsing) while Phase 1 (network handlers) isn't passing eval
- Doing Phase 4 work (SAEs) before probes (Phase 3) pass quality gates
- Building reasoning pathways (Phase 7) before differential analysis (Phase 6) is stable
- Fine-tuning anything before Phase 6 minimum

### Implementation pitfalls
- Reconstructing positions from board state without preserving FEN metadata (castling rights, en passant, halfmove clock)
- Underpromotion: naive SAN parsing breaks on `e8=N`-type moves
- Threefold repetition: same board with different history has different Leela eval
- Side-to-move: probes must know which side they're trained for
- Smolgen in BT4: not vanilla transformer attention; logit-lens math differs
- Attention policy head: policy isn't linear projection from residual stream
- Capturing all activations for one position is ~150 MB; batch carefully
- **Validator derivation: adding an extractor whose prompt cites derived ratios without extending the validator helper.** The validator's `_collect_derived_percentages` (`leela_chat/validator/__init__.py`) recognises specific field-pair derivations — currently visit-share only (`mcts.children[i].visits / total_nodes_searched` plus the `visits_in_flight` and `root_stats.visits` variants). When you add a new extractor whose prompts cite percentages or ratios that aren't literally in the findings JSON but are structurally derivable from two of its fields (e.g. threat-count ratios, multipv visit share, material-loss percentages, king-safety attacker/defender ratios), you MUST extend `_collect_derived_percentages` (or add a parallel helper if the derivation is not a percentage) with the explicit field-pair mapping AND add one false-positive test per addition, otherwise the LLM's structurally-correct prose is false-negatived as `ungrounded_eval` and falls to the templated fallback. Same shape of bug as the LogLiveStats UCI silent-no-op below — synthetic-fixture tests pass while real-pod prose fails validation. Canonical example: Chunk 6.6 commit `27e7b2d` for the original visit-share derivations. **Do NOT preemptively add derivations** — wait for a real extractor that needs them; broader matching risks false positives.

### LLM pitfalls
- Using a weaker LLM in production to save money (hallucination risk is nonlinear in model strength)
- Deleting old prompt versions (logs reference them by version number)
- Letting prompts drift toward vagueness — enforce specificity rules in every prompt revision
- Adding "be helpful" or "use your knowledge" to prompts — this is a hallucination invitation

### Process pitfalls
- Notebooks containing logic
- Manual testing instead of eval set runs
- Committing without running tests
- Committing weights, large data, or `.env` files
- Working without git, even for "quick experiments"

### External-tool / protocol pitfalls
- **UCI `setoption` silently accepts unknown or hidden options.** python-chess (and any other UCI bridge) sends `setoption name X value Y` to the engine without validating that X is an exposed option. Lc0's `kProOnly` options (e.g. `LogLiveStats`) are *not* listed in `uci` option output and are *not* settable via `setoption`, but `setoption` calls naming them are accepted without error. This means an unverified `engine.configure({"X": True})` call can be a project-wide silent no-op. Lc0 v0.32.1 `LogLiveStats` (Chunk 6a v1, 2026-05-19) is the canonical example — single-search live-stats was actually single-snapshot for ~24 hours because synthetic-fixture tests passed against a stream that real Lc0 wasn't producing. **Required pattern for any new UCI-option-based behavior:** at engine spawn, query the option list (`uci\nquit` to the engine, parse `option name ...` lines) and assert the option is present, OR drive the behavior via a CLI flag treated as the source of truth. Same pattern applies to HookedTransformer hookpoint names (verify the hook string against the actual module's HookPoints before relying on it), Lc0 backend-specific quirks, and Stockfish UCI extensions that aren't portable across builds.

---

## 9. How to Work in This Repo (When You're Claude Code)

**Session start protocol (every time):**
1. Read this file (you're doing it).
2. Read `docs/PHASE_NOTES.md` to determine the current phase and chunk.
3. **If the user is entering a new phase, deliver the Phase Entry Brief per §10 BEFORE writing any code.**
4. If continuing an existing phase, verify any deferred reading from earlier is done before writing code in a new area.
5. Read the relevant section of `docs/BIBLE.md` for the current phase if you haven't this session.
6. Check `docs/DECISIONS.md` for relevant constraints on what you're about to do.

**Default to small, reversible changes.** This is research code. You don't know what will work. Prefer:
- New file over modifying existing
- Pydantic model over dict
- Explicit over clever
- Slower-but-clearer over faster-but-magic

**When in doubt about scope:** ask. "Should I also fix X while I'm here?" is almost always answered "no, separate change."

**When the user asks for a new handler:** check that all existing handlers pass eval first. If not, push back. Adding handlers while old ones fail is a project-killing pattern.

**When the user asks to skip the validator, eval, or Phase Entry Brief:** push back. These are non-negotiable per §2 and §10 of this file.

**When the user asks to change a pinned version:** stop and ask explicitly. Confirm they understand the cost. Update `docs/DECISIONS.md` if they confirm.

**When writing prompts:** start from an existing handler's prompt as template. Version it. Save under `leela_chat/llm/prompts/{handler}/v{N}.md`. Reference the version in code via a constant.

**When writing tests:** unit tests use small fixtures (one or two FENs). Integration tests use real engine calls. Eval tests live in `tests/eval/` and run separately.

---

## 10. Phase Entry Protocol — Proactive Gatekeeping

**Before writing any code, Claude Code must determine the current phase and gate-keep the entry.** This is the project's defense against drifting into a phase without the right context.

A **phase transition** is detected when any of the following is true:
- `docs/PHASE_NOTES.md` says the current phase is higher than the last recorded session
- The user says "let's start Phase N," "I'm moving to Phase N," or equivalent
- The previous phase's exit criteria in §11 are now met
- The user starts asking about work clearly belonging to a later phase

When a phase transition is detected, **Claude Code's first response in that session must be a Phase Entry Brief.** Do not write code until the brief is delivered and the user acknowledges.

### What a Phase Entry Brief contains

Produce these five sections, in order:

#### 1. Phase identification
State the phase number, name, and exit criteria from §11.

#### 2. Mandatory reading — surface specific papers from PRIOR_WORK.md
Pull the "Read" list for this phase and present each as a checkbox the user can confirm or defer. Examples:

- **Phase 0:**
  - v2 Bible cover-to-cover (`docs/BIBLE.md`)
  - McGrath et al., "Acquisition of Chess Knowledge in AlphaZero" (arXiv 2111.09259) — skim methodology
  - Lin et al., "Tracing the Thought of a Grandmaster-level Chess-Playing Transformer" (arXiv 2604.10158) — your North Star
  - Davis et al., "Exploring the Limits of Probes for Latent Representation Edits" (MDPI AI 2026)
  - Jenner et al., "Evidence of Learned Look-Ahead" (arXiv 2406.00877) — abstract + conclusions
  - Monroe, "Mastering Chess with a Transformer Model" (arXiv 2409.12272) — BT4 architecture

- **Phase 1:**
  - Monroe BT4 paper — deep dive this time
  - lczerolens README + walkthrough notebooks
  - "Iterative Inference in Chess Networks" (arXiv 2508.21380)
  - Lee et al., "Improving Chess Commentaries by Combining LMs with Symbolic Reasoning" (arXiv 2212.08195) — closest existing dataset to the product

- **Phase 2:**
  - Lc0 `--verbose-move-stats` documentation
  - McGrath et al., "The Hydra Effect" (arXiv 2307.15771) — awareness pass for Phase 6 prep

- **Phase 3:**
  - Alain & Bengio, "Understanding intermediate layers using linear classifier probes" (arXiv 1610.01644)
  - Hewitt & Liang, "Designing and Interpreting Probes with Control Tasks" (EMNLP 2019)
  - Belinkov, "Probing Classifiers" (arXiv 2102.12452)
  - McGrath AlphaZero paper — re-read for concept list

- **Phase 4:**
  - Anthropic, "Towards Monosemanticity" (transformer-circuits.pub)
  - Anthropic, "Scaling Monosemanticity"
  - Fudan paper — deep re-read
  - Karvonen et al., "Measuring Progress in Dictionary Learning" (arXiv 2411.18895)
  - Rajamanoharan et al., "JumpReLU SAEs" (arXiv 2407.14435)
  - Poupart, "Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents" (arXiv 2406.04028)

- **Phase 5:**
  - Anthropic Circuits monthly updates on feature labeling
  - Bricken et al., "Decomposing Language Models With Dictionary Learning"
  - Schut et al., "Bridging the Human-AI Knowledge Gap" (PNAS 2025)

- **Phase 6:**
  - Hydra Effect paper — deep re-read this time
  - Heimersheim & Nanda, "How to Use and Interpret Activation Patching" (AI Alignment Forum)
  - Meng et al., "ROME" (arXiv 2202.05262)
  - Goldowsky-Dill et al., "Path Patching" (arXiv 2304.05969)

- **Phase 7:**
  - Fudan paper — focus on Algorithm 1 in detail
  - Marks et al., "Sparse Feature Circuits" (arXiv 2403.19647)
  - Conmy et al., "Towards Automated Circuit Discovery" (arXiv 2304.14997)

- **Phase 8:**
  - Constitutional AI / Anthropic prompting documentation (docs.claude.com)
  - ChessGPT paper (arXiv 2306.09200) for style
  - Jhamtani et al. chess commentary corpus (ACL 2018)

#### 3. Mandatory tooling / repo adoption
Pull the "Adopt" list from PRIOR_WORK.md's "Phase-by-phase reading and adoption order" section. For each, ask whether it's installed, evaluated, or needs setup. Key per phase:

- **Phase 0:** lczerolens, python-chess, pytorch, Lc0 (compiled with CUDA + `--verbose-move-stats`), Stockfish, BT4 network pinned with hash in DECISIONS.md.
- **Phase 1:** Frontier LLM API access verified.
- **Phase 2:** Stockfish wrapper, MCTS tree parser.
- **Phase 3:** lczerolens probing notebook adapted, scikit-learn, Lichess puzzle DB downloaded.
- **Phase 4:** **Fudan Leela-SAEs repo evaluation** (see go/no-go below), or OpenMOSS Language-Model-SAEs, or EleutherAI sparsify.
- **Phase 5:** Labeling tool (Jupyter or web UI), Neuronpedia referenced for UX design.
- **Phase 6:** nnsight or TransformerLens, Jenner activation patching infrastructure, Maia models for human-vs-Leela diff.
- **Phase 7:** Fudan reasoning pathway code (if released), graph visualization library.
- **Phase 8:** FastAPI + chessground.js (or equivalent web stack).

#### 4. Go/no-go decisions for this phase
Surface explicit decision points the bible and PRIOR_WORK flag:

- **Phase 0:** confirm BT4 vs. T82 network pin. Record in DECISIONS.md ADR-0001 with SHA-256.
- **Phase 3:** finalize concept list (target ~70–100) BEFORE training any probe. Premature training wastes hours.
- **Phase 4 week 1:** **EVALUATE FUDAN LEELA-SAES REPO.** If it doesn't reproduce in 2 weeks, fall back to OpenMOSS or sparsify. **Do not proceed past week 2 without resolving this.** This is the single highest-variance decision in the project — a working Fudan repo saves 2-3 months.
- **Phase 6:** before any ablation experiment, confirm Hydra Effect mitigations planned (multi-component ablation, mean ablation comparison).
- **Phase 8:** alpha launch criteria (5+ strong-player friends using it weekly).

#### 5. Starred items in PRIOR_WORK.md to prioritize
Surface ⭐, ⭐⭐, or ⭐⭐⭐ items relevant to this phase. For example:
- All phases: **⭐⭐⭐ Fudan paper (Lin et al., arXiv 2604.10158)** is the project's North Star.
- Phase 1: **⭐ Lee et al. chess.com Q&A dataset** is the closest existing dataset to the product.
- Phase 3: **⭐⭐ McGrath AlphaZero concept paper** is the methodology template.
- Phase 4: **⭐ Davis MDPI paper** shows linear probes beat SAEs on legal-move generation — proves Phase 3 was worth doing.
- Phase 6: **⭐ Hydra Effect** and **⭐ Heimersheim & Nanda guide** must be re-read.
- Phase 7: **⭐ Marks et al. Sparse Feature Circuits** is the LLM precedent for Fudan's chess work.

### After delivering the brief

Wait for the user's confirmation before writing code. If they say "skip the brief," do NOT blanket-skip — ask which specific items they want to defer, and note those in `docs/PHASE_NOTES.md` so they're not lost.

### Continuing within a phase (no transition)

When NOT entering a new phase, the briefing is unnecessary. But when about to write code in a part of the system you haven't touched this phase, quickly scan: are there relevant unread items from this phase's mandatory reading? If yes, mention them before writing code.

---

## 11. Phase Exit Criteria

Current phase, exit criteria, and active work live in `docs/PHASE_NOTES.md`. Update at the start of every session. The criteria below are the gate to advancing.

**Phase 0 — Environment and Foundations**
- Exit when: lczerolens smoke test passes, Lc0 with `--verbose-move-stats` runs, Stockfish callable, BT4 downloaded and hashed, repo skeleton committed, 50 eval positions written.

**Phase 1 — Network-First Chat MVP**
- Exit when: 30 eval positions pass on all 3 handlers (`what_is_leela_seeing`, `why_this_move`, `did_you_consider`); zero hallucinated moves in last 100 interactions; <30s latency; you find it useful for your own games.

**Phase 2 — MCTS Integration**
- Exit when: MCTSNode parsing works on 100 random positions; eval trajectories populated; 6 handlers total pass eval; Stockfish comparison surfaces real disagreements.

**Phase 3 — Concept Probes**
- Exit when: 70+ probes pass AUC>0.9 quality gate; `describe_position` agreed-with by 2000+ player on 80% of eval; failed probes documented.

**Phase 4 — Sparse Features**
- Exit when: SAEs loaded for 5+ layers; max-activation cache built; feature extractor produces 30-80 firings per position; <10s extraction latency.

**Phase 5 — Feature Labeling**
- Exit when: ≥500 features labeled with confidence ≥3; ≥200 have rule-based precision ≥0.85; labels queryable by feature ID.

**Phase 6 — Differential and Counterfactual**
- Exit when: `why_is_my_move_wrong` accurate on 80%+ eval-set mistakes; counterfactual handlers work on canonical examples; hydra-effect mitigations in place.

**Phase 7 — Reasoning Pathways**
- Exit when: pathway extraction works on 50+ test positions; LLM narrates pathways as steps; Fudan case study reproduces.

**Phase 8 — Polish and Launch**
- Exit when: 200+ position eval at high quality; web UI usable; 5+ strong-player friends using weekly.

Detail and per-chunk work items belong in `docs/PHASE_NOTES.md`, not here.

---

## 12. Pointers to Deeper Context

**Read for "why":** `docs/BIBLE.md` — the strategic vision, full phase descriptions, philosophy, pitfalls, glossary.

**Read for "what to use":** `docs/PRIOR_WORK.md` — every paper, GitHub repo, dataset, model, library worth knowing about, organized by tier and by phase relevance. Use the table of contents at the top of that file to navigate.

**Read for "what we decided":** `docs/DECISIONS.md` — architecture decision records.

**Read for "what we learned":** `docs/PHASE_NOTES.md` — per-phase retrospectives, current chunk, surprises.

**Read for "the full text":** `update_research/` — full PDFs of cited papers and complete source of reference repos. Use this when `docs/PRIOR_WORK.md` (the curated index) points you to something but you need the actual paper section, an algorithm derivation, or to see exactly how the original authors implemented a step. **Be selective** — read the abstract / intro / a targeted section, not the whole paper. `docs/PRIOR_WORK.md` is always the entry point; descend into `update_research/` only when you need depth. See `update_research/README.md` for the folder's conventions.

**For specific subsystems** (topic pointers — find by heading in each file):

| Topic | Where to look |
|---|---|
| Network internals and BT4 architecture (smolgen, attention policy head) | BIBLE: "Phase 0 — Choose your network" and "Pitfalls: Smolgen in BT4". PRIOR_WORK: "Research — Chess networks and architecture" (Monroe paper, arXiv 2409.12272) |
| MCTS tree parsing and `--verbose-move-stats` | BIBLE: "Phase 2 — MCTS Integration". PRIOR_WORK: "Tier 1 — Lc0" (verbose flags) |
| Probe training methodology | BIBLE: "Phase 3 — Concept Probes". PRIOR_WORK: "Research — Concept probing methodology" + lczerolens probing notebook |
| SAE loading, feature extraction, labeling | BIBLE: "Phase 4 — Sparse Features" and "Phase 5 — Feature Labeling". PRIOR_WORK: "Tier 2 — Major SAE dependencies" and "Research — Sparse autoencoders" |
| Causal interventions and counterfactual analysis | BIBLE: "Phase 6 — Differential and Counterfactual". PRIOR_WORK: "Research — Causal interventions, patching, counterfactuals". MUST read Hydra Effect first |
| Reasoning pathways (Fudan methodology) | BIBLE: "Phase 7 — Reasoning Pathways". PRIOR_WORK: "Tier 2 — Leela-SAEs" plus the Lin et al. Fudan paper |
| LLM prompts, translation, validation | BIBLE: "The LLM Translation Layer (Deep Dive)". PRIOR_WORK: "Research — Chess commentary, NLP, LLM-as-translator" (Lee et al., Jhamtani, ChessGPT) |
| Eval framework and methodology | BIBLE: "The Evaluation Framework". CLAUDE: Rule 3 in §2 |
| Existing chess products (for differentiation) | PRIOR_WORK: "Existing chess analysis products" |
| Datasets for probe/SAE training | PRIOR_WORK: "Tier 5 — Datasets" |
| Community help when stuck | PRIOR_WORK: "Community resources and forums" (Lc0 Discord first) |
| Full text of any cited paper or repo | `update_research/pdfs/` for PDFs, `update_research/repos/` for source. Start from PRIOR_WORK.md to find the right reference; only descend into `update_research/` when you need depth beyond the curated annotation. |

**Starred items in PRIOR_WORK.md** (⭐, ⭐⭐, ⭐⭐⭐) are the highest-leverage entries — adopt or read them first when entering a new phase.

---

## 13. What to Push Back On

You (Claude Code) should respectfully push back when the user asks for any of these:

| Request | Push back because |
|---|---|
| "Skip the Phase Entry Brief" | §10. Without the brief, the project drifts into phases unprepared. Offer to defer specific items instead. |
| "Skip the validator just this once" | Rule 2. No exceptions. |
| "Let the LLM use its chess knowledge for this" | Rule 1. The whole project dies if we relax this. |
| "Add a new handler" while existing ones fail eval | Project-killing pattern. |
| "Switch to a different Lc0 network" | Invalidates probes, SAEs, labels. Multi-month rollback. |
| "Fine-tune the LLM" before Phase 6 | Bible anti-goal. |
| "Build the web UI now" in Phase 0-2 | 3-week distraction. CLI first. |
| "Train your own SAEs" before evaluating Fudan's | Wasted month if Fudan's work. Per §10 Phase 4 go/no-go. |
| "Skip the eval set this commit" | Rule 3. |
| "Use a cheaper LLM in production" | Hallucination risk goes up nonlinearly. False economy. |
| "Implement [feature]" without an eval entry first | Bible says eval entries come first. |
| "Reimplement [Lc0 / python-chess / lczerolens] ourselves" | Bible explicitly forbids this. |

When pushing back, point to this file or the bible. Offer the right alternative.

---

## 14. File Maintenance

This file is updated when:
- A new project-wide rule is adopted (add to §2 or §3)
- A new pinned version is added (update §4)
- A new pitfall is discovered (add to §8)
- A new common request pattern emerges (add to §13)
- A phase transitions (update §10 reading lists if PRIOR_WORK changes)

Anything phase-specific or work-in-progress goes in `docs/PHASE_NOTES.md`, not here. Keep this file under 600 lines.

---

*Read this at the start of every session. When in doubt, ask before acting.*
