# Architecture Decision Records — Leela Chat

> Every meaningful architectural or dependency decision is recorded here with date, status, context, alternatives considered, and consequences. New decisions appended at the bottom. Existing decisions are never deleted — superseded ones are marked with status `Superseded by ADR-NNNN`.
>
> **When to write an ADR.** Pinning a version, choosing between two libraries, adopting a research methodology, ruling out a feature class, making any choice that future-you would question. If you find yourself thinking "why did I do it this way?" three weeks later, that was an ADR moment you skipped.

---

## ADR Format

```
## ADR-NNNN: Short decision title

- **Date:** YYYY-MM-DD
- **Status:** Proposed | Accepted | Superseded by ADR-MMMM | Rejected
- **Phase:** Phase N — name
- **Decision-makers:** [your initials]

### Context
What was the situation? What problem prompted this decision?

### Decision
What did we decide?

### Alternatives considered
What else did we look at? Why did we reject each one?

### Consequences
What does this enable? What does this foreclose? What's the cost if we ever want to reverse it?

### References
Links to PRIOR_WORK.md entries, papers, repos, or commits relevant to this decision.
```

---

## Decisions

### ADR-0001: Pin Lc0 network to BT4-1024x15x32h-swa-6147500

- **Date:** 2026-05-16 (decision accepted; weights downloaded + hashed 2026-05-16)
- **Status:** Accepted
- **Phase:** Phase 0 — Environment
- **Decision-makers:** [initials]

#### Context
We need a single Lc0 network for the entire project lifetime. Switching networks invalidates probes, SAEs, max-activation caches, and feature labels.

#### Decision
Pin **BT4-1024x15x32h-swa-6147500**.

Two file formats are stored side-by-side on the remote box. They represent the same weights — the ONNX is a conversion of the protobuf, with the same `BT4-1024x15x32h-swa-6147500` checkpoint at its core.

| Format | Used by | Source | SHA-256 |
|---|---|---|---|
| `model.pb.gz` (Lc0 protobuf, 335,897,546 B) | Lc0 binary (Phase 2 MCTS) | `https://huggingface.co/shermansiu/lc0_BT4-1024x15x32h-swa-6147500` | `66e6c1b5308c34230801dc171df580d7a66ec916dc2211246a485ef8e545c67e` |
| `BT4-1024x15x32h-swa-6147500.onnx` (741,143,429 B) | lczerolens (Phase 1+ forward passes) | `https://huggingface.co/andyrdt/leela-chess-bt4-onnx` | `d869428f55dc27699c0b252cd288e149607f1fd1a3d421ed2e037a4cd1744c44` |

- **On-disk path (remote box):** `/workspace/weights/BT4-1024x15x32h-swa-6147500/`
- **Architecture confirmed via `config.yaml`:** `embedding_size: 1024`, `encoder_layers: 15`, `encoder_heads: 32`, `use_smolgen: true`, `input_type: classic`, `value: wdl`, `policy: attention`.
- **GPU forward pass verified** 2026-05-16 via lczerolens on the ONNX file (RTX 4090). Output heads: `policy: (1, 1858)`, `wdl: (1, 3)`, `mlh: (1, ...)` — extra moves-left head beyond what Maia-1100 has.

#### Alternatives considered
- **T82-768x15x24h-swa-7464000:** matches Jenner et al. and Iterative Inference papers. Rejected because the Fudan SAEs are trained on BT4, and Fudan is the most product-relevant prior work for Phases 4–7.
- **Train on latest network:** rejected. Pinning is the explicit project rule.

#### Consequences
- Enables direct reuse of Fudan Leela-SAEs if their code reproduces.
- Forecloses easy use of Jenner et al.'s look-ahead activation patching code without porting it to BT4.
- Reversal cost: months of retraining everything.

#### References
- PRIOR_WORK.md → Tier 1 → Lc0 networks
- PRIOR_WORK.md → Tier 2 → Leela-SAEs (Fudan)
- BIBLE.md → Phase 0 → Choose your network and pin it FOREVER

---

### ADR-0002: Local CPU laptop + remote cloud GPU split for development

- **Date:** 2026-05-16
- **Status:** Accepted (cloud-provider sub-choice TBD before Chunk 3)
- **Phase:** Phase 0 — Environment
- **Decision-makers:** [initials]

#### Context
This laptop (WSL2 on Windows) will never have a GPU. The project requires GPU for Lc0 inference, BT4 forward passes, probe training (Phase 3), SAE work (Phase 4), and the eval-set runs that gate every commit per Rule 3 (CLAUDE.md §2). We need an explicit place for GPU work to live.

#### Decision
**Develop on this laptop (CPU-only); run all GPU work on a rented cloud GPU machine.** Code is the single source of truth and lives in git; the remote box pulls and runs.

Local responsibilities: code, prompts, validator, JSON schemas, unit tests on small fixtures, eval-set authoring, translation-layer prompt iteration (calls Anthropic API, not local GPU), smoke tests against tiny Maia nets.

Remote responsibilities: Lc0 binary with CUDA backend doing real MCTS, lczerolens forward passes against BT4, probe training, SAE loading + max-activation cache builds, eval-set regression runs.

Cloud-provider choice (vast.ai vs RunPod vs Lambda vs Modal vs other) is deferred and will be recorded in a follow-on ADR before Chunk 3 begins.

#### Alternatives considered
- **Friend's machine:** rejected for now — adds collab coordination overhead and an ownership ambiguity. Can revisit if cloud cost becomes a sustained blocker.
- **Defer GPU decision entirely:** rejected — Chunk 3 (Lc0 + BT4 download) needs to land *on* the GPU machine. Deferring just pushes the decision into a chunk that depends on it.
- **Local GPU:** foreclosed by user constraint (this laptop will never have a GPU).

#### Consequences
- Single-owner control of the GPU environment.
- `pyproject.toml` must support both local CPU and remote CUDA torch wheels. Workable via uv's per-package index selection — local is pinned to `torch==X.Y.Z+cpu`, `torchvision==A.B.C+cpu` from the `pytorch-cpu` index; the remote box will override these with CUDA-build wheels.
- Operational cost: hourly/monthly cloud GPU spend (~$0.30–1/hr for adequate hardware; manageable for solo research use).
- Need an SSH / dev-loop process: rsync code → run remote → fetch outputs. Or a cloud-native dev environment. Open question.
- Eval-set runs (CLAUDE.md §2 Rule 3) require remote execution. Pre-commit eval must call out to the remote box; cannot be a local hook.
- CLAUDE.md may need a new §15 documenting which subsystem runs where. Defer until after first remote run, when the boundary is concrete.

#### References
- ADR-0001 (BT4 network pin) — BT4 weights live on the remote box, not this laptop.
- PHASE_NOTES.md 2026-05-16 session log — origin of the constraint.
- CLAUDE.md §2 Rule 3 (eval-set gate) — remote execution required.

---

### ADR-0003: Cloud GPU provider — RunPod Community Cloud, RTX 4090 default

- **Date:** 2026-05-16
- **Status:** Accepted
- **Phase:** Phase 0 — Environment
- **Decision-makers:** [initials]

#### Context
ADR-0002 locked the architectural pattern (local CPU + remote cloud GPU split) but deferred provider choice. Chunk 3 (Lc0 build + BT4 download) is next and needs a concrete target machine.

#### Decision
- **Provider:** RunPod.
- **Default tier:** Community Cloud.
- **Default GPU class:** RTX 4090 (24 GB VRAM).
- **Persistent volume:** start at 100 GB, grow as needed.
- **Higher-tier instances** (Secure Cloud, A100/H100) spun up ad-hoc for specific needs (long unattended runs, Phase 4 SAE training from scratch if the Fudan repo doesn't reproduce).

#### Alternatives considered
- **Modal:** strong fit for the Rule 3 "eval on every commit" pattern (serverless function model). Rejected as primary because it ties code to Modal's abstractions and reduces portability if the project ever moves to a friend's box. Worth reconsidering for the eval-CI subsystem specifically once that exists.
- **vast.ai:** cheaper still, but spot-marketplace volatility is the wrong trade-off for reproducible eval runs.
- **Lambda Labs:** good hardware, overkill cost-wise before Phase 4 SAE training.
- **Paperspace, CoreWeave, Crusoe:** wrong fit (overkill or enterprise-oriented).
- **RunPod Secure Cloud as default:** rejected — 2–3× cost without proportional benefit while Phases 0–3 are burst-style. Easy to spin up Secure ad-hoc when needed.
- **RTX 3090:** functionally equivalent for current needs, marginally cheaper, but the 4090's ~1.5–2× speedup on inference is worth the small hourly delta.
- **A100 40 GB:** defer until Phase 4 demonstrates need.

#### Consequences
- Concrete operational model: SSH-based dev loop, rsync code, persistent volume holds BT4 weights + compiled Lc0 binary + SAE caches.
- Need a dev-loop convention (sync code → run job → fetch outputs). Defer to a script in Chunk 3.
- Eval-set runs (Rule 3) eventually need an automated spin-up → run → capture → spin-down. Worth scripting under `scripts/` during Chunk 3 or early Phase 1.
- **Checkpoint discipline is mandatory** for any run > ~1 hour due to Community Cloud reclamation risk. Build this into probe-training and SAE-training scripts from the start.
- Cost ceiling for Phases 0–3 is small (a few USD per active day on Community Cloud). Phase 4 SAE training is the main cost watch-out and likely warrants spinning up Secure Cloud + A100 for the duration.
- Lock-in is minimal — RunPod's SSH + Docker model means the project remains portable (friend's box, different cloud, etc.).

#### References
- ADR-0002 (local/remote split).
- ADR-0001 (BT4 pin — weights live on the RunPod persistent volume from Chunk 3 onward).
- PHASE_NOTES.md 2026-05-16 session log.

---

### ADR-0004: Pin Lc0 engine version to v0.32.1

- **Date:** 2026-05-17
- **Status:** Accepted
- **Phase:** Phase 0 — Environment
- **Decision-makers:** [initials]

#### Context
Per CLAUDE.md §4 the Lc0 engine version is a foundational pin. Lc0 ships no Linux binaries in recent releases (v0.31.x – v0.32.x are Windows + Android only), so the engine has to be built from source on the remote box. Pinning the source tag we build from prevents silent drift.

#### Decision
Pin Lc0 to **v0.32.1** (released 2025-11-23, the latest stable as of build date).

- **Source tag:** `https://github.com/LeelaChessZero/lc0/releases/tag/v0.32.1`
- **Build host:** RunPod Secure Cloud RTX 4090 (per ADR-0003 fallback after Community capacity issues 2026-05-17)
- **Build toolchain:** meson 1.3.2 + ninja 1.11.1 + nvcc 12.8 (preinstalled at `/usr/local/cuda-12.8/`).
- **Backends auto-detected by meson:** `cublas`, `cudnn`, `cudart`. Default runtime backend selected: `cuda-fp16`.
- **Installed binary:** `/workspace/bin/lc0`
- **Runtime env:** `LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64`
- **Verified 2026-05-17:** `--verbose-move-stats` emits the expected `info string <move> N:... P:... Q:... V:...` lines per move candidate. 500-node search from starting position completes in ~250ms after backend warmup. `bestmove d2d4` from start with BT4.

#### Alternatives considered
- **v0.32.0 / pre-release tags:** rejected — no reason to use older stable.
- **HEAD / master:** rejected — non-reproducible. Bug fixes can change `--verbose-move-stats` formatting, which Phase 2 will parse.
- **Pre-built Linux binary:** unavailable; no recent Lc0 release ships one.
- **Ubuntu `nvidia-cuda-toolkit` package:** rejected — ships older CUDA (12.0 on 24.04); we want toolkit major.minor to match the runtime cu128.

#### Consequences
- Reproducible Lc0 behavior across rebuilds (same source tag, same backends).
- `--verbose-move-stats` output format frozen to v0.32.1 — Phase 2 parsing can rely on it.
- Bumping the version is a deliberate ADR amendment, not implicit drift.
- A fresh pod requires a rebuild (~10 min ninja compile if CUDA toolkit is preinstalled, longer if not). All commands captured in PHASE_NOTES.md 2026-05-17 entry.
- `cuda-fp16` backend uses half-precision inference. Numerical parity with lczerolens ONNX (fp32) is approximate, not exact. May matter when comparing engine output to network output in Phase 2 — flag for verification.

#### References
- ADR-0001 (BT4 network pin) — Lc0 binary consumes BT4 `.pb.gz`; lczerolens consumes BT4 `.onnx`. Same weights.
- ADR-0003 (RunPod GPU target) — Lc0 lives on the remote box, not the laptop.
- CLAUDE.md §4 "Pinned versions" — Lc0 version listed as TBD; this ADR fills it.
- CLAUDE.md §8 — "Smolgen in BT4: not vanilla transformer attention; logit-lens math differs." Affects Phase 6, not Phase 0.

---

### ADR-0005: LLM model selection — Opus 4.7 for eval/prod, Sonnet 4.6 for iteration

- **Date:** 2026-05-17
- **Status:** Accepted
- **Phase:** Phase 1 — Network-First Chat MVP
- **Decision-makers:** [initials]

#### Context
CLAUDE.md §4 pins LLM provider to Anthropic Claude (primary) / OpenAI (fallback) but leaves the specific model unspecified. Phase 1 needs a concrete pin for both the iteration loop (frequent calls during prompt design — cost matters, speed matters) and the production / eval-set path (correctness matters, hallucinations are project-killers per CLAUDE.md §8).

#### Decision
Two-tier model selection driven by a flag on the LLM client:

- **Iteration mode (prompt design, exploratory testing):** Claude **Sonnet 4.6** (`claude-sonnet-4-6`). ~$3/1M input, ~$15/1M output. Fast feedback loop while tuning prompts.
- **Eval / production mode (eval-set runs, live user queries):** Claude **Opus 4.7** (`claude-opus-4-7`). ~$15/1M input, ~$75/1M output. The model the eval set is scored against; the model live users get answers from.

The flag (e.g. `LLM_MODE=iter|prod` env var or CLI option) defaults to `prod`. Iteration mode must be explicitly opted into.

#### Alternatives considered
- **Opus only (production from day one):** rejected — prompt-design iteration is many calls per insight; Opus cost ramps fast during exploratory work.
- **Sonnet only (for cost):** rejected — CLAUDE.md §8 pitfall is explicit: *"hallucination risk is nonlinear in model strength."* The model the eval set is graded against must be the same as the production model. Sonnet's hallucination rate is meaningfully higher.
- **GPT-4 / OpenAI primary:** rejected — CLAUDE.md §4 already pins Anthropic as primary. OpenAI remains the fallback if Anthropic API is down.

#### Consequences
- **Cost ceiling (estimated):** eval-set run of 150 calls (50 positions × 3 handlers) on Opus ≈ $12. Daily personal use (~20 calls) on Opus ≈ $1.60. Iteration in Sonnet during a heavy prompt-tuning session ≈ $1–3.
- **Prompt portability:** prompts must work on both models. Sonnet-specific tricks (if any) can't enter the production prompt. Test prompts in both modes before merging.
- **Eval-set baseline lock-in:** baseline scores on the eval set are recorded against Opus. Switching production model later invalidates the baseline — would be a separate ADR.
- **LLM client API surface:** the client wrapper has to accept a mode flag and route to the appropriate model. Minor extra code.

#### References
- CLAUDE.md §4 (LLM provider pinning).
- CLAUDE.md §8 "LLM pitfalls" — *"Using a weaker LLM in production to save money."*
- Phase 1 Entry Brief G1.2 (this session).

---

<!-- Append new ADRs below this line. Number sequentially: ADR-0006, ADR-0007, etc. -->

### ADR-0006: MCTS tree parser — source from `superhuman-ai-consistency`

- **Date:** 2026-05-18
- **Status:** Proposed (final option chosen at Phase 2 Chunk 1)
- **Phase:** Phase 2 — MCTS Integration
- **Decision-makers:** [initials]

#### Context
Phase 2's first hard engineering task is parsing Lc0's `--verbose-move-stats` output into a Python `MCTSNode` tree with per-node eval trajectories. PRIOR_WORK.md's earlier characterization ("no library wraps this") is no longer accurate. The `superhuman-ai-consistency` repo (Fluri/Paleka/Tramèr, MIT, arXiv 2306.09983) ships `rl_testing/mcts/tree_parser.py` (588 lines, networkx-backed) plus `rl_testing/engine_generators/relaxed_uci_protocol.py`, with a full Q/V/N/U/P/WL/D/M/S attribute map. It targets older Lc0 versions; some compat work is required against our pinned Lc0 v0.32.1 (ADR-0004).

#### Decision
**Defer the final option to Phase 2 Chunk 1**, but provisionally commit to **Option B — clean-room port** based on the patch v3 review.

- **Option A — vendor with compat fixes.** Copy `rl_testing/mcts/` + `relaxed_uci_protocol.py` into `leela_chat/engine/`, patch Python 3.11 + Lc0 v0.32.1 + BT4 quirks, add unit tests. Adds `networkx` as a runtime dependency.
- **Option B — clean-room port (recommended).** Treat their parser as a working spec; reimplement `leela_chat/engine/mcts.py` using our conventions (frozen Pydantic, `loguru`, CLAUDE.md §6 style). No heavyweight new dependency unless we want it.
- **Option C — build from scratch without their reference.** Previously planned; now superseded — wastes weeks redoing published work.

#### Alternatives considered
- **Option A** rejected (provisionally) because their code style drifts from our Pydantic + loguru conventions; the port cost is bounded and small compared to the long-term maintenance cost of vendored code that doesn't match house style.
- **Option C** rejected — superseded by their published work.

#### Consequences
- Phase 2 starts from a working reference, not a cold start. MCTS-parser cost ceiling shifts from weeks to roughly a chunk's worth of work.
- `networkx` is NOT added as a dependency unless Option A is later chosen at Phase 2 entry.
- Lc0 v0.32.1 `--verbose-move-stats` output format is the parse target — ADR-0004 stabilizes that.
- Decision actually finalizes at Phase 2 Chunk 1; this ADR captures the option space so the call doesn't have to be re-researched then.

#### References
- PRIOR_WORK.md → `superhuman-ai-consistency` entry (added in patch v3).
- ADR-0004 — Lc0 v0.32.1 pin.
- BIBLE.md §6.1 — Phase 2 MCTS parser description (updated alongside this ADR).
- arXiv 2306.09983 — Fluri/Paleka/Tramèr, "Evaluating Superhuman Models with Consistency Checks."

---

### ADR-0007: Phase 3 exit criteria — add behavioral-test metric

- **Date:** 2026-05-18
- **Status:** Proposed (applied at Phase 3 design entry, before any probe training)
- **Phase:** Phase 3 — Concept Probes
- **Decision-makers:** [initials]

#### Context
Current Phase 3 exit criterion is "70+ probes pass AUC>0.9" (BIBLE.md §7.7, CLAUDE.md §11). That measures whether a concept is *encoded* in the residual stream — necessary but not sufficient. Lovering, Forde et al. ("Evaluation Beyond Task Performance: AlphaZero in Hex," added to PRIOR_WORK.md in patch v3) show empirically that encoding without behavioral use is the dominant failure mode of probe-only methodologies. The pattern parallels CLAUDE.md §2 Rule 1 ("LLM never invents chess facts") one layer deeper: a probe that fires on a concept the network never uses to make decisions is interpretation theatre.

#### Decision
**Add a behavioral-test metric to Phase 3 exit criteria** in addition to the existing AUC>0.9 encoding gate. Provisional formulation:

> For each ⭐ concept (the highest-priority subset of the ~100 probe targets per BIBLE.md §7.1), pass a behavioral test: in ≥5 positions where the concept is decisive, the probe firing must correlate with the expected handler output. Encoding-only probes (AUC>0.9 but no behavioral signal) are flagged as "detected but unused" and do NOT count toward the 70-probe exit gate.

The N=5 floor and the exact "decisive" predicate get sharpened when the concept list lands in Phase 3 Chunk 1.

#### Alternatives considered
- **Option A — add behavioral tests (this decision).** Cost: ~5-10 hand-authored test positions per ⭐ concept. Catches encoding-without-use early.
- **Option B — keep AUC-only, defer behavioral testing to Phase 8 eval framework.** Cheaper now, expensive later: by Phase 8, probe-driven handlers have already shipped. If their probes encode-but-don't-use, the eval grader can't distinguish that from prompt-or-translation failures. Rejected.

#### Consequences
- Phase 3 takes longer (behavioral-test authoring per ⭐ concept), but ships interpretation that survives Lovering's critique.
- `alphatology` (PRIOR_WORK.md addition from patch v3) is the methodology template — a reference read, not a vendoring dependency.
- ⭐ concept identification must precede probe training: the behavioral test is designed *first*, the probe trained *against* a known target predicate. This tightens Phase 3 Chunk 1's order of operations.
- CLAUDE.md §11 Phase 3 exit criterion needs a wording update at Phase 3 entry to reflect this addition.

#### References
- PRIOR_WORK.md → Lovering, Forde et al. "Evaluation Beyond Task Performance" (patch v3 addition).
- PRIOR_WORK.md → `alphatology` repo entry.
- BIBLE.md §7.7 — Phase 3 exit criteria (updated alongside this ADR).
- CLAUDE.md §2 Rule 1 — same anti-pattern at a different layer.

---

### ADR-0008: Phase 4 SAE fallback ordering — `lczero-planning` first if Fudan no-goes

- **Date:** 2026-05-18
- **Status:** Proposed (applied at Phase 4 week 2 — the Fudan go/no-go gate per CLAUDE.md §10)
- **Phase:** Phase 4 — Sparse Features
- **Decision-makers:** [initials]

#### Context
CLAUDE.md §10 Phase 4 go/no-go currently reads: *"if Fudan's code doesn't reproduce in 2 weeks, fall back to OpenMOSS or sparsify."* That ordering predates having `lczero-planning` (Poupart, arXiv 2406.04028, MIT) in hand. `lczero-planning` is *already Lc0-shaped* — uses `lczerolens`, `nnsight`, ships SAE training scripts targeting Leela networks specifically. OpenMOSS and `sparsify` both require porting work to get to Leela. The patch v3 review surfaced this gap.

#### Decision
**Reorder Phase 4 fallback to: `lczero-planning` → OpenMOSS → sparsify**, with Tang et al. (Skill Adaptation SAEs on Maia-2) as methodology-only Option D.

The Fudan code path remains the first attempt per CLAUDE.md §10. Only the fallback ordering changes — the Fudan go/no-go gate at Phase 4 week 2 is unchanged.

#### Alternatives considered
- **Option A — `lczero-planning` → OpenMOSS → sparsify (this decision).** Recommended: already Lc0-shaped, MIT-licensed, smallest porting cost. Risk: Poupart's README says "not a stable codebase"; branch pins on `lczerolens` (`patching` rev) and `nnsight` (Poupart fork's `modules` rev) need verification at adoption time.
- **Option B — keep OpenMOSS first** because of `lczero-planning`'s stability disclaimer. Rejected: the Lc0-shaped head start outweighs the stability risk, given the alternative is porting OpenMOSS to Leela ourselves.
- **Option C — parallel-run `lczero-planning` + OpenMOSS in weeks 3-4 and pick whichever produces the first interpretable feature on BT4.** Rejected: doubles the workload during what should be focused fallback execution.

#### Consequences
- Phase 4 fallback path is shorter by an indeterminate amount of porting time if Fudan no-goes — `lczero-planning` may need only branch-pin verification and BT4-specific adaptation, not a wholesale port.
- Carries risk if `lczero-planning`'s pinned forks (`lczerolens@patching`, Poupart-fork `nnsight@modules`) become unavailable or incompatible with our pinned `lczerolens >= 0.4.0`. **Verification gate before adopting:** confirm those branches still resolve, OR locate archived snapshots.
- CLAUDE.md §10 Phase 4 go/no-go bullet needs updating when Phase 4 entry happens to reflect this fallback order.
- PHASE_NOTES.md open question 9 ("Branch-pin durability in `lczero-planning`") is the gating verification this ADR points at.

#### References
- PRIOR_WORK.md → `lczero-planning` entry (added in patch v3).
- PRIOR_WORK.md → Poupart, "Contrastive SAEs for Interpreting Planning" (arXiv 2406.04028).
- CLAUDE.md §10 Phase 4 go/no-go.
- PHASE_NOTES.md open question 9 — branch-pin durability.
- BIBLE.md §8.1 — Phase 4 decision tree (updated alongside this ADR).

---

### ADR-0009: Stockfish comparison extractor — Phase-2 work pulled into Chunk 5

- **Date:** 2026-05-18
- **Status:** Accepted
- **Phase:** Phase 1 — Network-First Chat MVP (with deliberate Phase 2 carve-out)
- **Decision-makers:** [initials]

#### Context
The Chunk 5 v2 smoke run (10 dev fixtures) produced responses that grounded correctly against findings but were near-content-free chess-wise: descriptions of `eval_cp`, WDL, and policy-pct values. User audit caught a more concrete failure on position 4 (`1r1r2k1/5ppp/p3pn2/4q3/b1PN4/Q4P2/P2RB1PP/KR6 b - - 0 1`): Stockfish's clear best move `Rxd4` did not appear in Leela's policy top-5 (top: `Rxb1+ 33%`). MCTS diagnostic on the same FEN (`lc0` binary, 20k nodes) returned `bestmove d8d4` — confirming that Leela-with-search *also* picks `Rxd4`. The policy network alone undervalues the move; only search (MCTS or classical) discovers it.

This is exactly the failure mode CLAUDE.md §6 (Two kinds of thinking) and BIBLE.md §1.5 (What MCTS gives you that features don't) predict. The fix is to feed a search-derived signal into the findings JSON. Phase 2 plans to do that via MCTS-tree parsing; that's the higher-complexity engineering (parsing `--verbose-move-stats`, eval trajectories, ADR-0006). Stockfish comparison is the *cheap* version of the same idea — one subprocess call, no tree parsing, depth-limited classical search.

CLAUDE.md §8 explicitly lists "Doing Phase 2 work (MCTS tree parsing) while Phase 1 (network handlers) isn't passing eval" as a pitfall. The pitfall is about MCTS *tree parsing* (complex, multi-week). A Stockfish subprocess call is a Chunk-5-class addition, not Phase 2-class.

#### Decision
Add **Stockfish-comparison extraction** to Chunk 5, in the same handler (`what_is_leela_seeing`), as a second extractor feeding the same findings model.

- New extractor: `leela_chat/extractors/stockfish.py` — `extract_stockfish_eval(board, *, engine, depth=15, top_n=3) -> StockfishResult`. Uses python-chess `SimpleEngine.analyse` with `multipv=top_n`.
- `WhatIsLeelaSeeingFindings` gains a required `stockfish: StockfishResult` field.
- Prompt bumped to v3: explicit framing that `eval_cp`/`top_moves` from Leela come from the **policy network alone, no MCTS**, and that `stockfish.*` is an independent classical-search reference. Output template requires comparing the two and noting disagreements >0.5 cp.
- Latency budget: depth 15 on the pod's CPU ≈ 0.5-1 s per position; well under the 30 s Phase 1 ceiling.
- Stockfish version is whatever apt ships (Ubuntu 24.04 → Stockfish 16, currently). Pin at the version-major level in CLAUDE.md §4; not strictly an ADR pin since we use it as a reference, not as a critical dependency for interpretation work.

#### Alternatives considered
- **Wait for Phase 2 MCTS** to provide the search signal. Rejected — Phase 2 MCTS-tree parsing is multi-chunk work (ADR-0006); blocking Chunk 5's usefulness for that long means shipping a system whose Chunk 5 demonstrations are content-thin, which directly hurts Phase 1 exit criterion #4 ("you personally find it useful for your own games").
- **Add Stockfish as a separate handler** (e.g., `compare_with_stockfish`). Rejected — that fragments the user experience (need to call two handlers to get a complete read on a position) and doubles the prompt-engineering surface. The natural place for Stockfish is *alongside* Leela's view in the same response.
- **Lc0 MCTS in-process** instead of Stockfish. Rejected for now — Lc0 MCTS requires the full Phase-2 tree-parsing infrastructure to expose anything richer than a single bestmove. Stockfish's `multipv=3` gives us three candidate moves + evals + PVs out of the box.
- **Leave Phase 1 as-is and accept thin prose.** Rejected based on the user's audit feedback — Phase 1's prose was unmistakably "number paraphrasing." Without an independent search signal, all three planned handlers (`what_is_leela_seeing`, `why_this_move`, `did_you_consider`) would suffer the same content-thinness; the cost compounds across handlers.

#### Consequences
- Phase 1 chat responses now have a second engine's perspective and concrete lines (Stockfish PV). Materially more useful for personal chess analysis.
- Phase 1 latency budget consumes 0.5-1 s extra per question (depth 15). Still well under 30 s.
- The findings shape grows; the validator's grounding check now has more numbers to ground against (more permissive in practice — the LLM has more things it's *allowed* to cite).
- `handle_what_is_leela_seeing` now orchestrates *two* extractors. Sets the pattern for future handlers that will orchestrate three or more (network + MCTS + probes + ...).
- Phase 2's `compare_with_stockfish` handler (originally Chunk 6 in BIBLE.md §6.2) becomes mostly redundant — the comparison already happens in the default handler. Phase 2 can repurpose that handler slot or drop it.
- CLAUDE.md §8 pitfall list should add a clarification distinguishing "Phase 2 MCTS-tree parsing" (the real pitfall) from "any Phase-2-flavored work" (which is fine when bounded and the user clearly benefits). Will surface as a CLAUDE.md edit in a follow-up.
- This ADR is **Accepted** (not Proposed) because it ships in this commit, not at a later phase entry. Reversal cost: rip out the extractor + revert handler + revert prompt → small, since it's contained.

#### References
- PHASE_NOTES.md 2026-05-18 audit findings (user observation on position 4, MCTS diagnostic confirmation).
- ADR-0006 (MCTS tree parser sourcing) — the Phase 2 version of the same "feed search signal into findings" idea.
- BIBLE.md §1.5 — "What MCTS gives you that features don't".
- BIBLE.md §6.2 — `compare_with_stockfish` handler description (will be revisited at Phase 2 entry given this ADR's overlap).
- CLAUDE.md §8 — Phase-creep pitfall list (clarifying edit pending).
- `lc0` binary verified `bestmove d8d4 ponder a3c3` on position 4 at 20k nodes; pod log captured in this session.

---

### ADR-0010: Skip formal eval grading; user is QA; defer Chunks 7 & 10

- **Date:** 2026-05-19
- **Status:** Accepted
- **Phase:** Phase 1 — Network-First Chat MVP (and Phase 2 carryover)
- **Decision-makers:** [initials]

#### Context
CLAUDE.md §2 Rule 3 ("The eval set is sacred") mandates that the canonical eval set runs before any change to extractors / handlers / LLM / validator code, with per-handler scores compared to baseline. The plan was for Chunk 7 to build that grader (`eval_positions.yaml` reader + LLM-as-judge + must_mention/must_not_mention scoring) and Chunk 10 to gate Phase 1 exit on its results.

The system is currently small (one production handler, ~200 lines of orchestration, 49 tests). The user is sole developer and sole user at this scale. The Chunk 5 audit (2026-05-18) made the practical situation explicit: the user can run `scripts/smoke_test_handler.py` against a dev-fixture set and visually verify outputs in minutes. The formal grader would take a chunk of work to build, then sit idle waiting for friends' eval positions to actually grade against. Friends' authoring may or may not arrive.

The architecture-of-efficiency reframe (chosen 2026-05-19): the next high-value capability is Phase 2 MCTS integration, not formal-eval infrastructure. The user opts to take on the QA role manually while the system is auditable by one person.

#### Decision
**Skip Chunk 7 (eval grader) and Chunk 10 (Phase 1 exit verification) indefinitely.** Rule 3 is informally relaxed for Phase 1 and Phase 2:
- The de-facto regression harness is `leela_chat/scripts/smoke_test_handler.py`. Every commit to `leela_chat/extractors/`, `leela_chat/handlers/`, `leela_chat/llm/`, or `leela_chat/validator/` should run the smoke script and the user must visually compare against the prior run.
- Friends' eval positions, if delivered, become **nice-to-have** manual spot-check material — extra positions to feed into the smoke script — not an ML-style regression baseline.
- Phase 1 does NOT formally close. The four Phase 1 exit criteria in CLAUDE.md §11 remain on the record, but criteria #1 ("30 eval positions pass on all 3 handlers") and #2 ("zero hallucinated moves in last 100 interactions") are unenforceable without the grader. Criteria #3 (latency <30 s) and #4 ("you personally find it useful") stay as informal checks.
- This decision is revisited at **Phase 3 entry**. Phase 3 (concept probes) needs a phase-appropriate eval mechanism anyway — ADR-0007 already mandates a behavioral-test metric on top of AUC>0.9. When Phase 3 starts we either build the grader then (against richer findings) or decide it's still skippable.

#### Alternatives considered
- **Option A — build the grader now anyway.** Cost ~1 chunk of work + maintenance overhead. Value gated on friends' eval-position delivery (uncertain) and on the user actually committing changes the grader could catch (low at this scale). Rejected: net-negative ROI at the current system size.
- **Option B — partial automation (just `must_not_mention` checking).** Cheaper than full Chunk 7 but still requires friend-authored positions. Rejected: half-built infrastructure that no one would maintain.
- **Option C — defer to Phase 3 entry.** Functionally identical to this decision, but framed as "delay" rather than "skip." Rejected as a label — calling it a deferral implies a definite return date, which is false. The honest framing is "skipped; revisit at Phase 3 entry if scale demands."
- **Option D — full eval-driven discipline now.** Wait for friends' eval set, then build the grader, then enforce Rule 3 strictly. Rejected: blocks all downstream work on uncertain external delivery; project paralysis.

#### Consequences
- **No mechanical regression detection** for Phase 1 and Phase 2. The user is responsible for noticing regressions via the smoke script. Works while the system fits in one head; reassess at Phase 3 entry when probes / SAEs / labels grow the surface beyond manual auditability.
- **Smoke script gets promoted** from "Chunk 5 dev tool" to "regression harness." Worth making sure the FIXTURES list there covers the kinds of positions where regressions would hurt (which it already mostly does — 10 positions span endgame/middlegame/opening/tactical).
- **Phase 1 stays informally open** indefinitely. Project trajectory has a permanent "Phase 1 incomplete" mark; this is honest given the architecture-of-efficiency reorder.
- **Friends' eval-position work isn't wasted.** Their positions become extra smoke-test fixtures the user can spot-check.
- **CLAUDE.md §11 Phase 1 exit criteria need clarification** — either update them in a follow-up edit, or leave them as historical context with a pointer to this ADR. The latter is cleaner (preserve original intent).
- **ADR-0007 (Phase 3 behavioral tests) is unaffected** — it still applies. Phase 3 needs *some* eval mechanism, and that's when it gets built (correctly, against the behavioral-test methodology).
- **Reversal cost is small** — Chunk 7's design hasn't been written down in code; it stays a design idea. If we want to revive it (Phase 3 entry or earlier), no code is wasted.

#### References
- CLAUDE.md §2 Rule 3 — the rule being relaxed.
- CLAUDE.md §11 Phase 1 exit criteria — informally no longer enforceable.
- ADR-0007 — Phase 3 behavioral-test exit criterion (still in force; defines the *eventual* eval shape).
- PHASE_NOTES.md 2026-05-18 audit — the trigger (user observation that Phase 1 prose was thin pre-Stockfish, and that formal-eval infrastructure isn't the bottleneck on usability).
- `leela_chat/scripts/smoke_test_handler.py` — the de-facto regression harness.
- `tests/eval/eval_positions.yaml` — kept (Rule 3 says never remove positions), just unused for now; friends' contributions still go here if they arrive.

---

### ADR-0011: Quality bar shift — 2200+ player approval → super-GM-tier (2700+) as productionization target

- **Date:** 2026-05-24
- **Status:** Accepted
- **Phase:** Phase 2 — MCTS Integration (forward-looking through Phase 3+)
- **Decision-makers:** [initials]

#### Context
CLAUDE.md §1 originally pinned the project's quality bar: *"Target user: the project owner (a 2000+ player) for personal use, then strong-player friends, then maybe a public alpha. Quality bar: 2200+ player should approve. GM utility is aspirational, not v1."* This framing was set at project start when scope was uncertain.

In the Chunk 7 planning conversation (2026-05-24), the user clarified the actual productionization target: super-GM-tier players (2700+). The shift was buried in a Chunk 7 proposal ("Productionization target: super-GM-tier players (2700+). Quality is prioritised over every other consideration — latency, compute cost, API spend, implementation effort.") but is a project-identity change that needs explicit documentation, not a casual proposal-prompt assumption.

The shift affects: prompt design (must satisfy strong-player-coach voice, not just plausible-coach); subjective scoring (the 2400+ rater bar mentioned in earlier planning is no longer sufficient as a single rater — at minimum one 2500+ rater is needed in the pool, with 2600+ preferred for the productionization gate); latency tolerances (ADR-0012 relaxes the deep-mode cap to <90s on the back of the quality-over-latency stance); MCTS configuration (Chunk 7's deep-mode multi-config ensemble + Stockfish depth-30 multipv=10 are quality choices the new bar justifies).

#### Decision
Shift the project's productionization quality bar from "2200+ player should approve" to "super-GM-tier (2700+) is the productionization bar." The 2200+ approval gate remains as a minimum acceptance bar (anything that fails to satisfy a 2200+ player is unshippable), but the design target is super-GM-grade.

CLAUDE.md §1 amended to match.

#### Alternatives considered
- **Keep the 2200+ bar.** Rejected: doesn't match the user's stated productionization target; would cap prompt-design ambition and MCTS-configuration choices below what the user actually wants.
- **Aspirational super-GM, real 2200+ bar (current CLAUDE.md framing).** Rejected: the aspirational framing was project-start hedging when scope was uncertain. The user has now committed to super-GM as the actual target; treating it as "aspirational" would let prompt and MCTS work drift to the lower bar in practice.
- **Tiered approval (2200+ minimum, super-GM target, no formal middle gate).** This is the chosen framing — minimum gate + design target are different things, both explicit.

#### Consequences
- **Prompt design discipline tightens.** Chunk 7's 7m (prose contract extension + new prompt versions) and beyond must target strong-player-coach voice, not just observational accuracy. Voice criterion in subjective scoring becomes load-bearing.
- **Subjective scoring needs strong raters.** A single 2400+ rater is no longer the bar; at minimum one 2500+ rater for any productionization gate, with 2600+ preferred. The 2500+ recruitment item lands in PHASE_NOTES as a Phase B5 user action item (does NOT block Chunk 7 sub-chunks; tracked).
- **The user's own scoring (a 2000+ player) is no longer sufficient as the SOLE subjective signal for productionization.** It's still valuable as a sanity check (the user catches things AI scoring misses, per the 6.9 plan's reasoning), but the productionization gate requires a stronger reader. Documenting this so the user doesn't over-weight their own scoring.
- **AI scoring** stays as a placeholder signal where no strong rater is available; the 6.9 close used AI scoring per user delegation (see PHASE_NOTES 2026-05-24-cont).
- **Chunk 7 hard gates** (defined in plan 7o) include "deep-mode prose materially better than 6.9 baseline on ≥ 4 of 5 user-scored positions, with at least 5 of 10 cells scored by a 2500+ rater" — the strong-rater requirement is part of the close criterion.
- **No mechanical retroactive effect on Chunk 6.9 or earlier.** ADR-0011 is forward-looking; the 6.9 ship stands per the 2200+ bar that was in force at the time.
- **Reversal cost.** Small — backing this out means amending CLAUDE.md §1 back and dropping the 2500+ rater requirement. Could happen if super-GM-tier turns out to be unreachable for valid product-shape reasons. No code depends on the bar.

#### References
- CLAUDE.md §1 — amended in this commit.
- Chunk 7 plan (`~/.claude/plans/tell-me-and-explain-mutable-axolotl.md`) — Phase B1 + the cost/latency/quality stance that motivates this shift.
- ADR-0012 (latency cap relaxation) — companion ADR; the latency relaxation makes the quality bar achievable.
- ADR-0013 (Tier-2 MCTS deferred) — companion ADR; defers product features that aren't load-bearing for the quality bar.
- BIBLE.md §1 (project identity) — quality framing originates here; reread when re-evaluating this ADR.

---

### ADR-0012: Latency cap relaxation — per-mode caps replace the uniform <30s gate

- **Date:** 2026-05-24 (original); amended 2026-05-25 at Chunk 7c.6 (deep-mode cap relaxed from <90s to <300s)
- **Status:** Accepted
- **Phase:** Phase 2 — MCTS Integration (forward-looking through Phase 3+)
- **Decision-makers:** [initials]

#### Amendment note (2026-05-25, Chunk 7c.6)

Deep-mode cap raised from `<90s` to `<300s`. Rationale: the 7a.2 Stockfish projection (PHASE_NOTES 2026-05-24) revealed that the super-GM (ADR-0011) quality target's preferred Stockfish config — depth=30, multipv=10 — measures at **235s alone** on the walkthrough fixture. Under the original 90s cap that config was vetoed and 7f committed to depth=25/multipv=8 (~55s) — a less-aggressive Stockfish config than the quality target warrants. The original 90s was a project-start under-estimate of "click-and-coffee" tolerance for serious analysis under the super-GM bar; users prepared to wait for super-GM-grade analysis will tolerate substantially more than 90s. The 300s ceiling accommodates the full depth-30/multipv-10 config (235s) plus the rest of the deep-mode stack (BT4 ~0.6s + Lc0 verdict-pass ~7s + Lc0 exploration-pass ~5-15s parallel + LLM translate ~8-15s) comfortably. Quick mode (`<30s`) is unchanged. Deepest mode stays out-of-scope per ADR-0013. The Chunk 7f Stockfish config can revisit depth-30/multipv-10 at land time now that the cap allows it.

#### Context
CLAUDE.md §11 Phase 1 exit criterion #3 pinned the uniform latency cap: `<30s` end-to-end. This was set when there was one mode (default conversational chat) and the extractor surface was small. The Chunk 6.7 commit B promotion explicitly extended this: *"detailed-mode end-to-end latency remains <30s per Phase 1's already-achieved ceiling; the Phase 1 criterion silently re-applies to every later phase that adds extractor cost."*

Chunk 7's planned deep-mode additions (per the plan at `~/.claude/plans/tell-me-and-explain-mutable-axolotl.md`) cannot fit under 30s:
- Multi-config MCTS ensemble (7d): verdict pass at 100k nodes (~30-50s on RTX 3090) + exploration pass at 10-30k nodes (parallel, GPU-contending) ≈ 40-60s.
- Stockfish at depth 30, multipv=10 (7f) ≈ 10-15s.
- Focused MCTS for did_you_consider (7c) ≈ similar to verdict pass.
- LLM translation ≈ 8-15s.
- Even with parallelisation (Chunk 6.7 entry criterion #1, implemented in 7a), the slowest single track sets the floor and that floor exceeds 30s.

Companion to ADR-0011 (super-GM quality bar): the quality choices that make 2700+-grade prose possible (deeper Stockfish, multi-config ensemble, network confidence + chess-content extractors) drive latency up. Relaxing the cap is the trade ADR-0011 implies.

The relaxation needs to be bounded — unbounded latency turns the conversational product into batch analysis (different product shape; see ADR-0013's deepest-mode deferral). The chosen ceiling for deep mode at original codification was 90s (a click-and-coffee upper bound a user is plausibly willing to wait for a deep query); raised to 300s at Chunk 7c.6 per the amendment note above — see the amendment rationale for the trade-off arithmetic.

#### Decision
Replace the uniform `<30s` latency gate with per-mode caps:

| Mode | Latency cap | Rationale |
|---|---|---|
| **quick** | < 30s sustained | Preserves the Phase 1 exit criterion's intent for conversational chat; single MCTS pass + light extractors fit easily under this. |
| **deep** | < 300s sustained (was 90s pre-2026-05-25; raised at Chunk 7c.6) | Bounded relaxation; accommodates multi-config ensemble + Stockfish depth-30/multipv-10 (235s alone) + chess-content extractors + LLM translation; "click-and-coffee-and-something-else" tolerance for super-GM-grade analysis under ADR-0011. |
| **deepest** | EXPLICITLY OUT-OF-SCOPE | Functionally a different product (batch analysis, 10-30 min/query). Requires a future ADR amendment + likely a separate CLI entry point if it ever lands. Deferred per ADR-0013. |

CLAUDE.md §11 amended:
- Phase 1 exit criterion #3: latency cap is quick-mode-only.
- Phase 2 exit criterion (re Chunk 6.7 commit B's silent re-application): per-mode caps replace the uniform criterion; deep mode at <300s (was <90s pre-Chunk-7c.6); deepest mode out-of-scope.

#### Alternatives considered
- **Keep the uniform <30s gate.** Rejected: forces Chunk 7 to cut either Stockfish depth, multi-config ensemble, or chess-content extractors — all of which are foundational for Phase 3+ per ADR-0011's quality bar. The uniform cap was set when scope was small; it doesn't survive contact with the deep-mode design.
- **Unbounded deep mode.** Rejected: turns the conversational chat product into batch analysis; identity drift. The user explicitly waived cost-as-constraint in the Chunk 7 planning conversation but did NOT waive product-shape constraints. 300s (post-7c.6 amendment; was 90s at original codification) is the conversational ceiling; beyond that, it's a different product.
- **Multiple intermediate tiers (quick / medium / deep / deepest).** Rejected: too many modes for one chunk to commit to; complicates per-handler mode selection. Two modes (quick, deep) is the minimum that splits conversational latency from quality-prioritised analysis. Deepest is explicitly out-of-scope per ADR-0013.
- **Adaptive cap (latency = function of position complexity).** Rejected for Chunk 7 scope: this is closely related to the "adaptive search budget" feature deferred per ADR-0013. Possibly Chunk 8+ once Phase 3 reveals which positions benefit.

#### Consequences
- **Chunk 7's deep-mode features (7d ensemble, 7f Stockfish depth-30/multipv-10, 7k+7l chess-content extractors) become buildable within the latency contract.** The deep-mode latency projection in 7a (proactive trade-off-now decision per the original 75s/90s thresholds; thresholds re-keyed to the 300s ceiling post-7c.6 amendment) enforces this before extractors land. Specifically: 7f's Stockfish config can revisit depth-30/multipv-10 now that the cap allows the 235s measurement; was forced to depth-25/multipv-8 (~55s) under the original 90s cap.
- **Conversational product shape preserved.** A 300s ceiling (5 min; post-7c.6 amendment) is recognisably "click-and-coffee-and-something-else" deep analysis; a 10-30 min query would not be. The product identity stays "chat with Leela about positions" + "deep dive on demand."
- **--detailed flag becomes alias for --deep (per Chunk 7 plan 7a).** Existing users of --detailed will see materially richer prose AND longer latency. Documented in the 7a commit message + CLI --help text.
- **Quick-mode discipline tightens.** Quick mode is now the load-bearing conversational floor at <30s. Any extractor added to quick mode must fit; the quick gate is unmoved.
- **Deepest mode is foreclosed (without ADR amendment).** If a 5M-node super-GM study mode is ever wanted, it needs its own ADR + a separate product framing + likely a separate CLI entry. ADR-0013 codifies the deferral.
- **Phase 3+ MCTS-consumer latency expectations.** Probes / SAEs / labels training pipelines that USE the MCTS extractors should configure quick-mode or deep-mode explicitly; data-generation pipelines can take the deep-mode hit since they're batch jobs anyway.
- **Reversal cost.** Modest. Backing out means restoring CLAUDE.md §11 to the uniform cap and dropping deep mode (or relaxing the quality bar in ADR-0011). The two ADRs are coupled — if one reverses, the other likely should too.

#### References
- CLAUDE.md §11 — Phase 1 + Phase 2 latency criteria, amended in this commit.
- ADR-0011 (super-GM quality bar) — the companion decision that makes this relaxation necessary.
- ADR-0013 (Tier-2 deferred) — codifies the deepest-mode deferral.
- Chunk 7 plan (`~/.claude/plans/tell-me-and-explain-mutable-axolotl.md`) — 7a's deep-mode latency projection enforces the cap; 7b-7f's per-extractor costs roll up into the projection.
- Chunk 6.7 commit B (PHASE_NOTES 2026-05-21 (cont.)) — the silent re-application this ADR explicitly amends.

---

### ADR-0013: Tier-2 MCTS features deferred to post-Phase-3 evidence

- **Date:** 2026-05-24
- **Status:** Accepted
- **Phase:** Phase 2 — MCTS Integration (forward-looking through Phase 3-7)
- **Decision-makers:** [initials]

#### Context
The Chunk 7 planning conversation (2026-05-24) surfaced a proposal to bundle Tier-1-foundational MCTS work (multi-config ensemble, MultiPv, focused MCTS, network confidence, Stockfish depth bump) with Tier-2 product features (position perturbation analysis, multi-strength ensemble, adaptive search budget, iterative-search-and-prune, "deepest mode"). The proposal framed all of these as foundational for Phase 3+ because "Phase 3+ will consume MCTS outputs as input data and changing MCTS configuration mid-Phase-3 requires re-running everything downstream."

The framing was partly correct (MCTS configuration is foundational; changing it mid-Phase-3+ is expensive) and partly wrong (only the Tier-1 items are actually load-bearing for Phase 3+ research consumption). The Tier-2 items are product-grade analysis features whose value to the interpretability research pipeline is conditional on Phase 3+ evidence about which positions / position-types the system handles well vs poorly.

Concretely, what each phase actually consumes:
- **Phase 3 probes**: stable MCTS verdict per training position + Q value + visits. That's it.
- **Phase 4 SAE feature characterization**: position characterization = verdict + Stockfish disagreement flag.
- **Phase 5 label validation**: MCTS-decided ground truth (verdict).
- **Phase 6 differential analysis**: consistent MCTS config across pairs of positions. (Perturbation infrastructure MIGHT be useful here, but Phase 6 can build it then.)
- **Phase 7 reasoning pathways**: feature firings + MCTS verdicts.

None of position perturbation, multi-strength ensemble, adaptive search budget, iterative-search-and-prune, or deepest mode is load-bearing for any of these. They are product features that may add user-visible value but don't determine the shape of Phase 3+ training data.

The deferral has a methodological argument too: until Phase 3 produces evidence (which positions get strong vs weak probe coverage; which position-types reveal interpretability gaps), we don't know which Tier-2 features would actually address real failure modes vs which would be guesswork-driven speculation. Front-loading Tier-2 in Chunk 7 commits to product features before the data justifies them.

#### Decision
Defer the following Tier-2 MCTS features from Chunk 7 to Chunk 8+:

1. **Position perturbation analysis** — generating 8-15 minimally-perturbed variants of a position, running MCTS on each, surfacing "razor-edge vs robust" findings. Genuinely novel feature; needs methodology research (legality preservation, castling/en-passant edge cases, what counts as "minimal") before code. Chunk 8+ once Phase 3 surfaces a position class where it would help.
2. **Multi-strength ensemble** — running the same network at 1k / 10k / 100k / 1M visits to approximate human-equivalent strength levels. Needs empirical calibration of human-equivalent-strength mapping for the user's specific hardware + network. Chunk 8+.
3. **Adaptive search budget** — scaling node count based on probe-pass verdict stability + policy entropy. Efficiency optimization, not foundational. Chunk 8+ once Phase 3 identifies which positions probes find hardest.
4. **Iterative search-and-prune** — sequential deep dives on top-N candidates with fresh budget each. Useful for game-review workflows. Chunk 8+ feature.
5. **"Deepest mode"** — 5M+ nodes per query, 10-30 min wall clock, full multi-config + perturbation + multi-strength + Stockfish depth 40+. Functionally a different product (batch analysis, not conversational chat). Requires future ADR amendment to ADR-0012's latency contract + likely separate CLI entry point. Chunk 9+ if ever.
6. **Contempt sweep, Temperature ensemble (stochastic move-selection multi-run), Forced-line MCTS (multi-move forced sequences beyond focal first move), PolicyTemperature tuning experiments, NN cache persistence + tree reuse across session queries, Network policy / value interventions (Phase 6-adjacent), Cross-network consistency at representation level (Phase 4+-adjacent), Maia / multi-network ensemble.** All backlog. Each Chunk-8+ as evidence dictates.

Each Tier-2 feature becomes its own Chunk-8+ work item once Phase 3 has surfaced a concrete failure mode that the feature would address. Captured as a backlog list in PHASE_NOTES.

**Conditional on architectural additive-friendliness:** Chunk 7's mode architecture, findings-JSON Pydantic models, and prompt-loading infrastructure must accommodate Tier-2 additions without rework. See Chunk 7 plan sub-chunk 7a for the extension-friendliness verification (a `tests/integration/test_chunk7_extension_friendliness.py` test exercises this with synthetic stub tracks). If the verification fails, this deferral's premise is compromised and Chunk 7 either does the refactor needed or accepts partial rework for some Tier-2 features.

#### Alternatives considered
- **Land all Tier-2 features in Chunk 7.** Rejected: phase-sized scope (15 sub-chunks, $400-800 API spend, 10-15 weeks of work) for one chunk; commits to product features before Phase 3 evidence exists to validate their value.
- **Land position perturbation only (cherry-pick Tier-2).** Rejected: perturbation methodology is genuinely open (legality preservation, "minimal" definition, en passant / castling edge cases) — it's a research project before it's a feature. Including it in Chunk 7 means either rushing methodology or expanding 7g into 3-4 sub-chunks of methodology research.
- **Defer all Tier-1 + Tier-2 (delay Chunk 7 entirely).** Rejected: chess-content extractors (Chunk 7's 7k+7l) ARE the Phase 2 → Phase 3 bridge deliverable the bible names. Phase 3 needs them to characterize positions for probe training. Delaying Chunk 7 delays Phase 3.

#### Consequences
- **Chunk 7 scope is bounded** (11 sub-chunks instead of 15+; ~$20-40 in API spend vs $400-800). Manageable as one chunk.
- **Phase 3 entry brief follows Chunk 7 close, not a Chunk 7.5.** The Chunk 7 plan's Phase D placeholder commits this direction up-front so post-Chunk-7 work doesn't drift toward Chunk 8 Tier-2 work just because it's mechanically the "next chunk in the same phase."
- **Tier-2 features become evidence-driven.** When Phase 3's probes reveal (say) that the system handles tactical positions well but fails on positional ones, position perturbation analysis on tactical positions becomes wasted work and resources should go to whatever tells us about positional weaknesses instead. Front-loading Tier-2 would have committed to the wrong investments.
- **Extension-friendliness verification (7a) is load-bearing.** If Chunk 7's architecture isn't actually additive-friendly, the deferral's premise breaks; this ADR's argument depends on Chunk 7 leaving the door open architecturally. The 7a integration test is the contract.
- **The product looks less ambitious in v1** — no perturbation analysis, no multi-strength ensemble, no deepest mode. Acceptable per the conversational-product framing in ADR-0012; ambitious Tier-2 features can land later without retrofit pain if 7a holds.
- **Reversal cost.** Low — backing out means landing any of the deferred features in a later chunk. The deferral doesn't burn architecture or close doors; it just sequences.

#### References
- ADR-0011 (super-GM quality bar) — companion; the quality bar Tier-2 features ostensibly serve.
- ADR-0012 (latency cap relaxation) — companion; codifies the deepest-mode deferral specifically.
- Chunk 7 plan (`~/.claude/plans/tell-me-and-explain-mutable-axolotl.md`) — Phase B3 + Out-of-scope section + Phase D placeholder.
- Chunk 7 plan 7a — extension-friendliness verification (the contract this ADR depends on).
- BIBLE.md Phase 3-7 sections — what each phase actually consumes (defines the "load-bearing for research" filter this ADR applies).

---

### ADR-0014: Bundling rule's first behavioral pass — precedent (Chunk 7c)

- **Date:** 2026-05-25
- **Status:** Accepted (precedent established; rule remains permanently in CLAUDE.md §8)
- **Phase:** Phase 2 — MCTS Integration
- **Decision-makers:** [initials]

#### Preamble — what this ADR is, framing-wise

This ADR documents the first behavioral validation of the rule codified in CLAUDE.md §8 (bundling) rather than a new architectural decision; its purpose is to preserve the validation evidence so future amendments to the rule have a reference point. It also hosts a reference subsection of structural-response examples that CLAUDE.md §13 points to (relocated here at Chunk 7c.6 to keep CLAUDE.md focused on rules, not worked examples).

#### Context
The CLAUDE.md §8 bundling rule was codified at Chunk 7b.1 in response to the 7b retro: any sub-chunk that adds a findings-JSON field MUST ship its prose-version bump in the SAME commit chain — bundling-into-future-iteration is forbidden. The rule was added to prevent the silent-plumbing-cycle (N sub-chunks where the CLI looks unchanged) AND the attribution-debt cost (debugging by elimination across 6+ coupled citation patterns at a bundled 7m iteration).

7b.1 itself didn't really test the rule (no handler findings field added — only smoke-battery-runner infrastructure). 7c was the first sub-chunk where the rule's literal-text fired: focused-MCTS added new findings fields (`focused_*`) to `DidYouConsiderFindings` and the rule said the matching `did_you_consider/v4.md` prompt bump must ship in the same commit chain.

This ADR documents the precedent of the first behavioral pass and captures a crucial framing fix from the 7c retro's user pushback.

#### Decision
7c shipped new findings fields AND the matching prompt-version bump (v3→v4) AND the strict-citation guard extension AND the v4 worked-example coverage **all in the same commit chain** (commit `5ead072`, 2026-05-25). The temptation to defer the prompt update to 7m did not arise.

This is the precedent: bundling rule applies; bundling rule holds; future sub-chunks face the same expectation.

#### Crucial framing fix (from the 7c retro pushback)

My initial 7c retro framed the bundling-rule test as "easy" because findings and prompt were tightly coupled in 7c — no realistic scenario where I'd add `focused_pv_san` to the findings model but forget to update the prompt that explicitly cites it. The user pushed back:

> "the rule's value isn't only in hard tests — it's in eliminating the decision point 'is this coupling tight enough to defer?'"

The user is right and the retro's framing was wrong. The bundling rule's primary value is NOT surviving hard tests where the deferral temptation is strong. It's **eliminating the decision-point altogether** — there is no "is this tight enough to defer?" conversation to have, because the rule says ship-in-same-commit-chain regardless.

Pre-discounting the easy-test outcome (as my retro's first framing did) misses the rule's main work. Easy or hard, the rule produces the same outcome: no defer. Easy tests don't validate the rule weaker than hard tests; they validate it just as well — they prove the rule's decision-elimination property still holds.

The hard-test verdict (i.e., does the rule hold when the deferral temptation is actually strong?) lands at the first sub-chunk where findings change is more invisible. Until then, the rule is working as designed.

#### Alternatives considered (preserved from 7b.1's rule codification)
- **No rule (silent scope creep):** rejected. The 6.8 → 6.9 cycle of "we'll iterate at 7m" was the failure mode the rule exists to prevent.
- **Soft guideline rather than hard rule:** rejected at 7b.1 codification because guidelines are negotiable mid-chunk; hard rules aren't. The point is to eliminate the decision-point.
- **Allow opt-out with explicit ADR amendment:** considered for 7c.5; rejected because every deferral would require an ADR, raising the cost of deferral but not eliminating the temptation to litigate.

#### Consequences
- **Future sub-chunks (7d, 7e, 7f, 7k.1, 7k.2, 7k.3, 7l.1, 7l.2)** that add findings JSON fields ship their per-handler prompt-version bumps in the same commit chain. By Chunk 7 close, the bundling rule has been applied ~8 times in a row.
- **7m's role narrows** — was originally "add the prompt citations for every chess-content extractor"; with the bundling rule it's "extend the prose contract with polish + structural reframing" (the chess-content prompt citations land in 7k/7l sub-chunks, not 7m).
- **The attribution-debt cost at 7m drops dramatically.** 7m's smoke battery covers prompts that already cite the new findings cleanly; if a prose-quality regression appears, attribution is to 7m's contract-extension changes specifically (not to 6 coupled citation patterns).
- **The rule's first hard test** lands at the first sub-chunk where findings change is invisible (no prompt-citation-explicit field name to remind me). Likely 7e (network confidence) or 7l.2 (per-handler chess-content wire-up) — the field's prompt-relevance is less obvious than `focused_pv_san`'s.
- **Easy-test outcome ≠ rule-weak validation.** Documenting this so future-me doesn't re-litigate.

#### Reversal cost
Very low — would require amending CLAUDE.md §8 to remove the rule and accepting fragmented commit chains. No code depends on the rule's existence; only the decision-elimination property does.

#### References
- CLAUDE.md §8 Process pitfalls — the rule itself (added 7b.1).
- ADR-0011 / 0012 / 0013 — companion ADRs documenting other project-identity / structural shifts during Chunk 7 planning.
- PHASE_NOTES 2026-05-25 — Chunk 7c retro (user pushback on the "easy test" framing).
- Chunk 7 plan 7c section + 7c commit `5ead072` — the precedent itself.

#### ADR-numbering shift note
ADR-0014 was pre-allocated in the Chunk 7 plan's 7n section for the MCTS-baseline ADR. Renumbered 2026-05-25 because 7c.5.C (this ADR) lands first. 7n's ADR becomes ADR-0015 when 7n lands.

#### Reference: structural-response examples (relocated from CLAUDE.md §13 at Chunk 7c.6)

CLAUDE.md §13's structural-response-to-surfaced-failure-mode meta-rule (added 7c.5.D) points here for worked examples. Reference cases are listed chronologically; each entry shows: the plan's authorized scope, the structural change shipped beyond it, the reasoning, and the retrospective sign-off.

**Example 1 — Chunk 7c's interpretive-claim heuristic (2026-05-25; the canonical case the meta-rule was codified from).**

The 7b.2.E plan refinement said *"Update Rule A's wordlist or Rule B's patterns at 7c if either failure mode surfaces."* When the heuristic under-fired on 7c's spot-check prose, I (Claude Code) extended both Rule B's patterns AND changed the EXCLUSION LOGIC (a structural design change, not just a pattern addition). The plan didn't explicitly authorize the exclusion-logic change. User accepted retrospectively 2026-05-25 with reasoning that pattern-only updates couldn't have addressed the under-fire mode — the plan's framing was the thing that was wrong, not the structural change. The meta-rule was codified at 7c.5.D directly from this experience: structural responses ARE permitted; the discipline is to surface them in the retro for retrospective sign-off, not to silently expand scope OR to halt at the plan's literal text.

(Future structural-response cases get appended here as they accumulate, preserving the worked-example record for the meta-rule.)

---

### ADR-0015: Adaptive MultiPV mapping based on policy distribution character

- **Date:** 2026-05-25
- **Status:** Accepted
- **Phase:** Phase 2 — MCTS Integration (forward-looking through Phase 3+)
- **Decision-makers:** [initials]

#### Context

Chunk 7b committed `multipv=5` as the per-mode default for all deep-mode MCTS calls (all three handlers; whatever the position). The choice was a project-start default chosen before the policy-distribution character was systematically considered. Different positions warrant different multipv values:
- A position with a dominant top move (e.g., the policy assigns 80%+ to one move; the rest is noise) gains nothing from multipv=5 — the bottom three lines have ~no visits and add noise/cost to findings JSON.
- A position with naturally flat policy (e.g., quiet opening / equal middlegame where 8-10 moves share probability mass) benefits from multipv=10+ to surface the real candidate set.
- The walkthrough_rxd4 sacrifice case (top move at ~33%, second at 16%) is in the middle — multipv=5 is genuinely the right pick.

Phase 3+ probes will consume `MCTSFindings.multipv_used` (a new field per 7c.8) as training-data metadata. The mapping function from policy character → multipv is a stable contract Phase 3 reads; ADR documents the contract.

#### Decision

Adopt `compute_adaptive_multipv(top_prob, policy_entropy, mode) -> int` as the canonical mapping. Replaces every hardcoded `multipv=5` in handler call sites.

**Mapping (calibrated against synthetic distributions; see 7c.8 PHASE_NOTES):**

| Policy character    | top_prob   | normalized entropy | Base multipv     |
|---------------------|------------|--------------------|------------------|
| Dominant top move   | > 0.70     | < 0.60             | 2                |
| Moderate top move   | 0.30-0.70  | 0.60-0.90          | 5 (7b default)   |
| Contested           | < 0.30     | 0.85-0.95          | 10               |
| Naturally flat      | < 0.20     | > 0.95             | 15               |

**Normalized entropy.** Raw Shannon entropy is normalized by `ln(N)` where N is the number of policy candidates in the input — making the metric comparable across different `top_n` truncation values. The plan's original thresholds (entropy < 0.5, > 2.5) assumed unbounded distributions; with the typical top-5 truncation, max raw entropy is `ln(5) ≈ 1.6`, so the original thresholds were unreachable. Normalized entropy ∈ [0, 1] resolves this.

**Per-mode caps (post-mapping):**

| Mode      | Cap     | Rationale |
|-----------|---------|-----------|
| `quick`   | 3       | Conservative to fit the <30s quick-mode gate (ADR-0012). |
| `deep`    | **8**   | Calibration (anthropic.messages.count_tokens on synthetic findings dicts at multipv=1/2/3/5/8/10/15): multipv=15 is **2.67x** the token cost of multipv=5 — above the plan's 2x trigger for tightening. multipv=10 is 1.85x (acceptable on its own) but the policy was to cap at 8 once the 2x trigger fired on the higher value. Generous enough for contested positions under ADR-0011's super-GM target; tight enough to bound the prompt-input growth toward the 40k-token threshold (per the PHASE_NOTES token-cost trajectory watch-list). |
| `deepest` | 15      | OUT-OF-SCOPE per ADR-0013. Documented only as design intent for if/when deepest mode lands. |

**Both inputs are checked** because they're partially redundant but NOT identical signals: `top_prob` is a single-move read, `policy_entropy` is distribution-shape. A position with top_prob=0.40 and a sharp second-best (0.35) has lower entropy than one with top_prob=0.40 and flat 0.05 across many — both warrant different multipv values.

#### Alternatives considered

- **Keep hardcoded `multipv=5`.** Rejected: ignores the actual policy character; wastes search budget on dominant positions and undersearches contested ones. Each handler call had the same cost regardless of whether it was warranted.
- **Use `top_prob` alone (drop entropy).** Rejected: top_prob and entropy capture distinct shape information. A position can have top_prob=0.20 with second-best at 0.18 (sharp) vs second-best at 0.05 across many (flat) — different positions, same top_prob, different right answers.
- **Use raw entropy without normalization.** Rejected: thresholds depend on `top_n` truncation; can't compare positions analyzed with different `top_n`.
- **Don't cap deep at 8 (keep 10).** Rejected at calibration time: multipv=15 above the 2x cost trigger triggered the plan's tightening rule; capping at 8 is the chosen response.
- **Drop entropy threshold for "dominant" to <1.0 (looser).** Considered as part of normalization decision; rejected — normalized < 0.60 is the cleaner threshold that matches the actual "dominant top move" intuition with top-5 candidates.

#### Consequences

- **Phase 3 training data carries `MCTSFindings.multipv_used`** as a per-position metadata field — probes can stratify by policy character if needed.
- **Per-handler-per-mode prompt versions bump per CLAUDE.md §8 bundling rule.** what_is_leela_seeing/deep v8→v9; why_this_move/deep v5→v6; did_you_consider v4→v5. Each new version cites `mcts.multipv_used` so prose accurately describes the search scope.
- **Handler refactor: `extract_network_inference` now runs sequentially BEFORE the parallel ThreadPoolExecutor block** so its policy distribution can inform the multipv choice. The slow `extract_layer_emergence` (~3-4s) runs in parallel with Stockfish + Lc0 — net latency vs the pre-7c.8 arrangement is approximately neutral. Documented inline at each handler's wiring site.
- **Quick mode still uses `extract_mcts` (snapshot path) with implicit multipv=1.** Adaptive multipv only applies to `extract_mcts_trajectory` (deep mode). Quick path unchanged.
- **Inline entropy computation in handlers.** 7e (network confidence) may centralize entropy into a shared helper later; for now each handler computes inline from `top_moves[*].policy_pct`. The helper `policy_entropy_normalized` is colocated with `compute_adaptive_multipv` in `leela_chat/extractors/adaptive_multipv.py`.
- **Reversal cost.** Low — revert the helper imports + restore hardcoded `multipv=5` at the three handler call sites; revert prompt-version bumps; revert the parser/extractor `multipv_used` field surfacing. ~30 lines of code rollback.

#### References

- Chunk 7c.8 PHASE_NOTES — calibration measurement + handler-wiring notes.
- ADR-0011 (super-GM quality bar) — motivates the per-mode caps (the 300s deep cap from ADR-0012 amendment makes higher multipv tractable).
- ADR-0012 (latency caps; amended 7c.6: 90s → 300s) — the time budget against which the per-mode caps are sized.
- ADR-0013 (Tier-2 features deferred) — "deepest" mode out-of-scope; deepest's design-intent cap (15) documented but not enabled.
- CLAUDE.md §8 bundling rule — the v9/v6/v5 prompt bumps ship in the same commit chain as the findings field per the rule.
- `leela_chat/extractors/adaptive_multipv.py` — the mapping function + per-mode cap table; module docstring is the source-of-truth for the implementation.
- `tests/unit/test_adaptive_multipv.py` — 18 unit tests pinning each mapping branch + cap behavior.

#### ADR-numbering shift note (2026-05-25, second time)

ADR-0015 was pre-allocated in the Chunk 7 plan for the 7n MCTS-baseline ADR. Renumbered TWICE during Chunk 7's retro chain: first at 7c.5.C when ADR-0014 was taken for the bundling-rule precedent (7n shifted from 0014 to 0015), then again here at 7c.8 when this dynamic-multipv ADR took 0015 (7n shifts to 0016). 7n's MCTS-baseline ADR will land as **ADR-0016** when 7n executes.
