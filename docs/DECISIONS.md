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
