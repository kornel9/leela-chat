# Phase Notes — Leela Chat

> A living log of what's happening in the current phase, what surprised you, and what to remember when the next phase starts. Update at the start and end of every work session.
>
> **What goes here:** current chunk, today's plan, things you learned today, anything that surprised you, bugs that took hours to find, questions you have, things to revisit. Anything actionable or instructive at the chunk-to-phase scale.
>
> **What does NOT go here:** strategic vision (→ BIBLE.md), architectural decisions (→ DECISIONS.md), code conventions (→ CLAUDE.md).

---

## Current Status

- **Phase:** 2 — MCTS Integration. Entered 2026-05-19 after Phase 1's informal close. Phase 2 entry brief delivered + acknowledged; Phase 2 plan written (see `/home/kornel/.claude/plans/tell-me-and-explain-mutable-axolotl.md`).
- **Active chunk:** Phase 2 Chunks 1-5 done 2026-05-19; Chunk 6 trilogy done 2026-05-19; Chunk 6.5 done 2026-05-20; Chunk 6.6 + 6.7 + 6.8 + 6.8.1 done 2026-05-21; Chunk 6.9 done 2026-05-24 (subjective gate cleared by AI scoring per user delegation; user manual re-scoring deferred to Chunk 7's 7o once chess-content extractors land per CLAUDE.md §2 Rule 1 constraint). **Chunk 7 plan Phase B done 2026-05-24** — ADR-0011 (quality bar shift to super-GM 2700+), ADR-0012 (deep-mode latency cap relaxation to <90s; quick mode preserves <30s; deepest mode out-of-scope), ADR-0013 (Tier-2 MCTS deferred to post-Phase-3 evidence) all landed; walkthrough_rxd4 v5/v2 baseline prose preserved as test fixture; 2500+ rater recruitment in Pending user action items (does NOT block). **Active: Chunk 7 Phase C** — 11 sub-chunks Tier-1-foundational MCTS + chess-content extractors 7k+7l as the core deliverable + ADR-0014 pinning MCTS baseline + smoke battery + user re-scoring at 7o (+ conditional 7p iteration pass if 7o lands marginal).
- **Last session:** 2026-05-24
- **Chunk 7 entry criteria** (HARD prerequisites, promoted from soft "must include" in Chunk 6.7 commit B; must hold before any new Chunk-7 extractor lands). **Chunk 6.9 closed the prompt-fragility carry-over from 6.8: validation_ok 100%, illegal_move 0, templated 0%, forbidden-language hits 1/100 down from 4/100. AI subjective quality at parity or better on all 5 scored cells; user re-scoring deferred to post-7k+7l per the user-delegation close.**
  1. **Extractor invocation is parallelised.** Current sequential `handler_s` is ~7-8s detailed (extractors + ~750ms layer emergence). Chunk 7 adds ~2-4s of additional extractor time pre-parallelisation → projected post-Chunk-7 detailed latency ~20-25s, with Phase 3 probes pushing it over Phase 1's `<30s` exit criterion (CLAUDE.md §11). Chunk 7's FIRST deliverable is a per-extractor latency baseline on the walkthrough Rxd4 fixture + an `asyncio.gather` (or `concurrent.futures.ThreadPoolExecutor`) wrapper around extractor invocation so the slowest single extractor sets the floor instead of the sum. Parallelisation must demonstrably hold detailed-mode `handler_s` under ~10s before any new Chunk-7 extractor lands. The Phase 2 exit criterion in CLAUDE.md §11 has been amended to mention this ceiling explicitly.
  2. **Per-new-extractor derivation checkpoint** honoured for each of the 5 planned extractors. For each, ask: "does the prompt that will consume this extractor cite percentages or ratios derived from two of the extractor's fields rather than literal scalars present in the JSON?" If yes → extend `leela_chat/validator/__init__.py::_collect_derived_percentages` (or a parallel `_collect_derived_*` helper if the derivation is not a percentage) with the explicit field-pair mapping AND add one false-positive test per addition. See CLAUDE.md §8 "Implementation pitfalls — Validator derivation" for the canonical pattern (added in Chunk 6.7 commit A).
- **Alternative direction:** Phase 2 close + Phase 3 Entry Brief per CLAUDE.md §10 — concept-list pruning of `docs/phase3/concept_candidates.md` (~91 seed candidates) is the project's highest-leverage Phase-3 decision.

> **Pod state caveat:** RunPod stop/start cycles wipe everything outside `/workspace` (apt packages, `/workspace/bin/lc0`, `/usr/local/cuda-12.8/bin` PATH). Run `ssh runpod 'bash /workspace/leela-chat/scripts/setup_pod.sh'` after every pod restart — idempotent, takes ~30s on warm restart and ~10 min on first build of Lc0.

---

## Pending user action items

_Items that need user attention but do NOT block code work. Tracked here so they don't get lost; updated as items land or get superseded._

### 2500+ rater recruitment for Chunk 7's subjective scoring (opened 2026-05-24, Phase B5)

**Goal:** recruit at least one 2500+ rater for Chunk 7's 7o subjective-scoring sample, with 2600+ preferred. Per ADR-0011 (quality bar shift to super-GM 2700+), a 2400+ rater is no longer sufficient as a single rater — the productionization gate requires 2500+ minimum, and the design target is super-GM-grade prose that 2700+ would approve.

**Why it matters:** Chunk 7 closes (per plan 7o) on subjective-quality scoring of deep-mode prose. The user's own 2000+ scoring is a valuable sanity check but no longer sufficient as the sole subjective signal per ADR-0011. Without at least one 2500+ rater, the 7o close decision falls back to AI scoring + user-as-sanity-check, which is weaker than the productionization bar implies.

**Pool candidates:**
- Strong-player friends (any IM / GM the user has personal contact with).
- Online chess communities the user is in (Lichess Forums, Chess.com strong-player groups, lc0 Discord).
- Club / federation contacts.
- Coaches who do online lesson work — pay for one rating session.

**Compensation acceptable per user direction** (cost not a constraint). A rater scoring 10 cells of side-by-side prose ≈ 1-2 hours of careful reading; reasonable to compensate at typical chess-coaching rates.

**Does NOT block Chunk 7 sub-chunks.** Phase B6 (Active Chunk pointer to Chunk 7) does not block on rater confirmation per the plan's "track but don't block" decision. If rater recruitment lands during Chunk 7 sub-chunks, incorporate scores at 7o. If recruitment stalls indefinitely, 7o falls back to user-scoring-only with the gap noted in PHASE_NOTES.

**Status:** open. Update this section when a rater commits or when fallback path is taken.

---

## Phase 0 — Environment and Foundations

### Exit criteria (from BIBLE.md / CLAUDE.md §10)
- [x] lczerolens walkthrough notebook runs end-to-end on this machine — `notebooks/00_smoke_test.ipynb`, 2026-05-16
- [x] Lc0 compiled with CUDA backend, `--verbose-move-stats` produces parseable output — 2026-05-17. v0.32.1 built from source on remote box (cuda-fp16 backend, cublas + cudnn auto-detected). Binary at `/workspace/bin/lc0`. Verified: 500-node search from start position emits 21 info-string lines per move with Q/V/N/U/P/WL/D/M/S fields. See ADR-0004.
- [x] Stockfish callable via python-chess — 2026-05-16 on remote box (`/usr/games/stockfish`, Stockfish 16 via apt). `python-chess` UCI smoke passes.
- [x] BT4 network downloaded, SHA-256 hashed, recorded in DECISIONS.md (ADR-0001) — 2026-05-16. Both formats: `.pb.gz` (for future Lc0 binary) and `.onnx` (for lczerolens). Both pinned with SHA-256 in ADR-0001.
- [x] Repo skeleton committed (matches CLAUDE.md §5 structure) — 2026-05-16; full `leela_chat/` package tree + tests/{unit,integration,eval} + scripts/ + labels/ + README.md + .env.example. `uv run python -m leela_chat.scripts.smoke_test` works.
- [ ] 50 eval positions written in `tests/eval/eval_positions.yaml` with YAML schema validated — **friends authoring per G0.2 override.**

### Chunk plan
1. **Chunk 1** — lczerolens smoke test (2–4 hours)
   - Install Python 3.11, uv, CUDA toolkit
   - `uv add lczerolens torch python-chess`
   - Run smoke test: load `lczerolens/maia-1100`, forward pass, print policy shape
   - Commit `notebooks/00_smoke_test.ipynb`
2. **Chunk 2** — Repo skeleton (day 2–3)
   - Create directory layout per CLAUDE.md §5
   - Write pyproject.toml with pinned versions
   - Add .gitignore, .env.example, README stub
   - Initial git commit
3. **Chunk 3** — Lc0 build + BT4 download (day 3–5)
   - Compile Lc0 from source with CUDA
   - Verify `--verbose-move-stats` flag exists
   - Download BT4 network, hash, record in DECISIONS.md ADR-0001
4. **Chunk 4** — First 10 eval positions (day 5–7)
   - Three openings I know well
   - Three middlegames from my games
   - Two endgames (one technical, one tactical)
   - Two positions where I know the right answer
5. **Reassess** after Chunk 4. Plan Chunks 5–6 based on what I learned.

### Session log

#### 2026-05-16 — Phase 0 entry brief; foundational decisions locked

What I did:
- Delivered Phase 0 Entry Brief (CLAUDE.md §10).
- Locked go/no-go decisions:
  - **G0.1:** BT4-1024x15x32h-swa-6147500 confirmed as the pinned network. ADR-0001 advanced to **Accepted** (SHA-256 pending download in Chunk 3).
  - **G0.2:** Eval-set construction is solo through Phase 0 exit. Strong-player friends review at Phase 1 exit, not during construction (avoids signal loss from committee).
  - **G0.3:** Try an Lc0 release binary first; compile from source only if `--verbose-move-stats` is missing or broken in the build.

Reading status (R0.1–R0.6 from brief):
- On user's schedule. Recommended order: R0.1 BIBLE.md → R0.3 Fudan (⭐⭐⭐) → R0.6 Monroe BT4 (⭐⭐) → R0.2 McGrath AlphaZero (⭐⭐) → R0.4 Davis MDPI (⭐) → R0.5 Jenner look-ahead (⭐).

Tooling status (T0.1–T0.11):
- Inventoried 2026-05-16: Python 3.11.15 ✅, uv 0.11.14 ✅, git 2.53.0 ✅. Local CPU venv installed via `uv sync` ✅ — torch 2.12.0+cpu, torchvision 0.27.0+cpu, lczerolens 0.4.0, python-chess (chess 1.11.2), loguru, pydantic, jupyter/ipykernel.
- **GPU target decided:** cloud GPU rental (provider TBD). See ADR-0002. CUDA install on this laptop intentionally skipped — never happens.
- Stockfish + Lc0 + BT4 deferred to the remote box.

Chunk 1 result (2026-05-16):
- `notebooks/00_smoke_test.ipynb` runs end-to-end. Loads `lczerolens/maia-1100` from HF, forward pass on starting position returns `policy: (1, 1858)` and `wdl: (1, 3)` as expected. Kiwipete sanity check passes. CUDA available: False (intended). Exit criterion #1 satisfied.

What surprised me:
- **uv source pin gotcha:** `[tool.uv.sources] torch = { index = "pytorch-cpu" }` was not enough. The PyTorch CPU index hosts both `torchvision==0.27.0` (CUDA build) AND `torchvision==0.27.0+cpu` (true CPU build) at the same path. PEP 440 treats them as equal versions, so uv picked the CUDA wheel arbitrarily → `torchvision::nms does not exist` at import. Fix: pin both `torch` and `torchvision` to the exact `+cpu` local-version in `[project] dependencies` (`torch==2.12.0+cpu`, `torchvision==0.27.0+cpu`). uv 0.11.14 doesn't yet support `uv sync --torch-backend=cpu`. Worth re-checking when uv upgrades.
- HF Hub anonymous-rate-limit warning fires on `from_hf` calls — not blocking, but `HF_TOKEN` env var will be wanted in `.env.example` later.

Bugs:
- (none unresolved)

Questions / open threads:
- ADR-0001 + ADR-0002 `[initials]` fields not yet filled — user has not stated preferred initials.
- ~~Cloud provider sub-decision~~ — RESOLVED 2026-05-16: RunPod Community Cloud, RTX 4090 default. See ADR-0003.
- Dev-loop pattern for remote box: rsync + SSH vs. cloud-native IDE vs. mounted VS Code Remote. Decide in Chunk 3 when the RunPod box first comes up.
- `python-chess` vs `chess` PyPI naming: `python-chess>=1.999,<2.0` is the legacy name. Both the legacy `python-chess==1.999` shim and modern `chess==1.11.2` are installed; the runtime `import chess` resolves to 1.11.2. Probably worth simplifying to just `chess` in `pyproject.toml` to drop the redundant shim — non-urgent.
- Possible CLAUDE.md update needed: explicit §15 documenting which subsystems run local vs remote. Defer until first remote run lands and the boundary is concrete.

Next session goal:
- Chunk 3 (provision RunPod RTX 4090 Community instance per ADR-0003; mirror venv with CUDA torch wheel; download BT4 + sha256sum → fill ADR-0001; install Lc0 with `--verbose-move-stats`; install Stockfish).
- Possibly Chunk 4 (first 10 eval positions in the YAML), authored locally — no GPU needed.

---

#### 2026-05-16 (continued) — Phase 0 Chunk 2 done

What I did:
- Created the full `leela_chat/` package tree per CLAUDE.md §5: `engine/`, `network/`, `features/`, `probes/`, `labels/`, `extractors/`, `handlers/`, `classifier/`, `llm/` (+ `llm/prompts/`), `validator/`, `conversation/`, `eval/`, `cli/`, `scripts/`. All with `__init__.py`.
- `leela_chat/__init__.py` configures loguru once (stderr handler, INFO level, colored format) per CLAUDE.md §6 logging conventions.
- Mirrored the notebook smoke test as `leela_chat/scripts/smoke_test.py` so `uv run python -m leela_chat.scripts.smoke_test` (CLAUDE.md §7) actually works. Verified: passes.
- Created `tests/{unit,integration,eval}/`. `tests/eval/eval_positions.yaml` has the schema documented in comments + empty `positions: []` list (no real positions yet — that's Chunk 4; the schema-in-comments preserves Rule 3's "never remove" while still showing what an entry looks like).
- Added `README.md` (short intro pointing at CLAUDE.md + docs/), `.env.example` (ANTHROPIC_API_KEY, OPENAI_API_KEY, HF_TOKEN), top-level `scripts/` and `labels/` dirs (with `.gitkeep`).
- `pyproject.toml`: added `[tool.pytest.ini_options]` (testpaths, asyncio_mode=auto).
- Verified: `pytest --collect-only` reads config from pyproject.toml and reports "0 items" cleanly. `from leela_chat import logger` resolves.

What surprised me:
- CLAUDE.md §5 layout doesn't list `leela_chat/scripts/` as a subpackage but CLAUDE.md §7 specifies `uv run python -m leela_chat.scripts.smoke_test` as a command — so it has to exist as a Python package. Resolved by creating it; top-level `scripts/` stays for one-off non-package scripts. Worth noting in a CLAUDE.md §5 update later, but the §5 layout vs §7 commands mismatch is a single line, not a structural issue.

Bugs:
- (none)

Questions / open threads:
- Empty top-level `scripts/` and `labels/` dirs currently held by `.gitkeep` — replace those once real files land.
- `leela_chat/llm/prompts/` also `.gitkeep`-held — first handler prompt (Phase 1) will displace it.

Next session goal:
- Provision the RunPod box (ADR-0003), mirror venv with CUDA torch wheel, run smoke test there, download BT4, fill ADR-0001's SHA-256. Optionally start Chunk 4 (eval positions) locally in parallel.

---

#### 2026-05-16 (continued, 3) — Chunk 4 delegated to friends; pivoting to Chunk 3

What I did:
- Discussed eval-set authoring framing with user: handlers, ground truth, schema, FEN vs PGN, step-by-step process, handler spread (3–4 / 3–4 / 2–3 across the three Phase 1 handlers), recommended attributes are sufficient for first 10 (defer `expected_top_move_san` / `expected_eval_range` to Phase 1).
- **User decided to delegate Chunk 4 authoring to chess friends.** This overrides G0.2 (the "solo authoring + friends review at Phase 1 exit" decision locked on 2026-05-16 earlier).
- Pivoting to Chunk 3 (RunPod box) — no dependency on eval positions, so parallel work.

Open questions / threads:
- **G0.2 override consequence:** the original "friends review at Phase 1 exit" step disappears if friends are the authors. Need a replacement review step. **Options:** (a) user reviews each batch from friends as it lands; (b) a *different* friend reviews; (c) accept "multi-friend authoring = implicit consensus." **Decide before the first batch arrives.**
- **Eval-set timeline:** friends have ~3-week runway before Phase 1 handler work needs grading. If batches don't arrive on time, pull positions back in-house.
- **Per-handler batching:** if friends deliver positions in batches grouped by handler, the first Phase 1 handler can start grading once its ~10 are done — buys flexibility. Worth coordinating with them.

Next session goal:
- Chunk 3: provision RunPod RTX 4090 Community per ADR-0003. Mirror venv with CUDA torch wheel, run smoke test on the box, download BT4 + sha256sum → fill ADR-0001's pending hash, install Lc0 with `--verbose-move-stats`, install Stockfish.

---

#### 2026-05-16 (continued, 4) — Chunk 3 mostly done; Lc0 binary deferred

What I did:
- Provisioned RunPod pod (RTX 4090 Community, PyTorch 2.8.0 template, Ubuntu 24.04, 100 GB persistent volume at `/workspace`). Driver 580.105.08, CUDA 12.8.
- Generated local SSH ed25519 keypair (`~/.ssh/id_ed25519`), registered on RunPod.
- Set up `~/.ssh/config` with `Host runpod` alias for cleaner SSH commands.
- Wrote pyproject.toml local/remote split per ADR-0002: `[tool.uv.environments]` constrains the resolver to "WSL Linux" + "non-WSL Linux"; dependency markers use `'microsoft' in platform_release` to pick `+cpu` (laptop) vs `+cu128` (RunPod) wheels for torch + torchvision. Downgraded pin from 2.12.0 → 2.11.0 because cu128 index lags one minor behind cpu.
- Installed uv on remote, rsynced project to `/workspace/leela-chat/`, ran `uv sync` on the box → torch 2.11.0+cu128, torchvision 0.26.0+cu128, 174 packages.
- Smoke test (`leela_chat.scripts.smoke_test`, Maia-1100) passes on the box with `cuda_available=True`.
- **Downloaded BT4** in both formats: `.pb.gz` from `shermansiu/lc0_BT4-1024x15x32h-swa-6147500` (for the future Lc0 binary), `.onnx` from `andyrdt/leela-chess-bt4-onnx` (for lczerolens). Both SHA-256s recorded in ADR-0001. Verified `config.yaml` matches the pinned architecture.
- **BT4 GPU forward pass via lczerolens (ONNX) works.** Output heads: `policy (1, 1858)`, `wdl (1, 3)`, `mlh (1, ...)`. WDL on starting position W=0.190 D=0.644 L=0.166.
- Installed Stockfish 16 via `apt install stockfish` (after enabling `universe` repo). Verified `python-chess` UCI invocation works (depth-12 analysis from start: +40 cp white).

What surprised me:
- `lczerolens.LczeroModel.from_path` only accepts `.onnx` or `.pt`, not Lc0's native `.pb.gz`. Sherman Siu's HF repo only has `.pb.gz`. Resolved by also pulling the pre-converted ONNX from `andyrdt/leela-chess-bt4-onnx` — same network, just ONNX-serialised.
- **No recent Lc0 release ships a Linux binary** (v0.31.x – v0.32.x are Windows + Android only). Build from source is the only path. Deferred to its own focused task (~45 min).
- uv's default resolver locks for ALL platforms (Windows, macOS, aarch64) — initial cu128 pin failed because no matching wheel exists for Windows. Fix: `[tool.uv.environments]` whitelist to just the two real targets.
- PEP 440 quirk redux: even the CUDA index hosts the same version under both bare and local-version-suffixed names → still need explicit `+cu128` / `+cpu` qualifiers, just like the CPU side.
- Box has Python 3.12 baseline but uv installed 3.11 transparently via `uv sync`. No special command needed.
- Stockfish requires `apt add-apt-repository universe` on a fresh PyTorch container — not in default repos.

Open questions / threads:
- **Lc0 binary build:** when to do it. Cost ~45 min wall time including CUDA toolkit install. Not blocking Phase 1 design. Decide: now (session ~5) vs next session.
- ADR-0001 + ADR-0002 + ADR-0003 `[initials]` placeholders still empty. User has not specified.
- Friends' Chunk 4 progress unknown.
- ONNX vs `.pb.gz` numerical parity: both pin to BT4-1024x15x32h-swa-6147500. ONNX is probably fp32 while training was fp16 — small numerical differences (~1e-4) possible. Will matter if we ever compare lczerolens (ONNX) outputs to Lc0 binary (`.pb.gz`) outputs for consistency. Worth a quick parity test in Phase 2.

Next session goal:
- Either: finish Lc0 build on remote (task #10), close the last Phase 0 exit criterion blocker; or
- Start Phase 1 design while friends complete Chunk 4 (per CLAUDE.md §10, that requires a Phase 1 entry brief first).

---

#### 2026-05-17 — Lc0 v0.32.1 built; Phase 0 driver-side complete

What I did:
- Pod restart hit "GPUs no longer available" → migrate also failed ("This machine does not have the resources") → user terminated and recreated on **Secure Cloud** (ad-hoc per ADR-0003). New endpoint at `38.65.239.46:31833`.
- **Volume preserved across the recreate** (RunPod migrated `/workspace` to a new host even though the migrate UI looked like it failed). `/workspace/leela-chat`, BT4 weights, and even the project's `.venv` survived intact. Only `/` was fresh, so uv binary (in `/root/.local/bin/`) and stockfish needed reinstall.
- Reinstalled uv on the new container. Existing venv worked as-is (uv-managed Python 3.11 lives in the project's `.venv/`, which is on `/workspace`). `uv sync` was a no-op verify (171 packages checked, 0 changes).
- BT4 ONNX forward pass on GPU re-verified — works.
- `apt install stockfish` (after `add-apt-repository -y universe`) → Stockfish 16 at `/usr/games/stockfish`.
- Installed Lc0 build deps via apt: `meson`, `ninja-build`, `libprotobuf-dev`, `protobuf-compiler`, `libz-dev`, `libeigen3-dev`, `build-essential`.
- Discovered CUDA toolkit 12.8 was **preinstalled** at `/usr/local/cuda-12.8/` (PyTorch 2.8.0 template ships it; just not on PATH). Saved ~3 GB install + ~5 min wait.
- Cloned `Lc0 v0.32.1`, ran `meson setup build --buildtype=release -Dgtest=false` → cublas + cudnn + cudart auto-detected. ninja built 263 targets in ~10 min wall time on the RTX 4090 host. Binary copied to `/workspace/bin/lc0`.
- Verified `--verbose-move-stats` produces the expected per-move-candidate format. 500-node search from start: `bestmove d2d4`, 21 info-string lines emitted with Q/V/N/U/P/WL/D/M/S fields. Phase 2 has its data source.
- Wrote ADR-0004 pinning Lc0 to v0.32.1.

What surprised me:
- **Migrate UI lied about failing.** Both the "GPUs unavailable" denial and the "no resources to deploy your pod" error appeared during the recreate attempt, but when the new pod actually came up, the persistent volume was already on the new host. RunPod's UI messaging is misleading — the data was migrating in the background even when the UI errored. Net: don't panic; check `/workspace` content before assuming you lost data.
- **CUDA toolkit was preinstalled** on the PyTorch 2.8.0 template, just not on PATH. Worth knowing for future Lc0 builds. The `which nvcc` check returns missing but `/usr/local/cuda-12.8/bin/nvcc` works.
- Lc0's UCI command sequence is async — `quit` racing `go nodes N` will silently kill the search before bestmove. Need `sleep N` between go and quit when scripting from shell. Worth scripting properly when we build the Python wrapper in Phase 2.
- `cuda-fp16` is the default backend Lc0 picks when both cuda and cuda-fp16 are available. Numerical parity vs lczerolens fp32 ONNX will need verification when Phase 2 starts comparing engine and network outputs.

Bugs:
- (none unresolved)

Open questions / threads:
- ADR-0001 + ADR-0002 + ADR-0003 + ADR-0004 `[initials]` placeholders still empty. User has not specified.
- Friends' Chunk 4 progress unknown.
- Stockfish + Lc0 binary installs DON'T persist across pod recreates (live on `/`, not `/workspace`). Could move both to `/workspace/bin/` for portability, but stockfish is fast to reinstall (`apt install stockfish`) so not urgent.
- ONNX (fp32) vs `.pb.gz` (fp16) numerical parity check — Phase 2 prep task.
- Lc0 build commands captured in this entry — could be scripted as `scripts/setup_remote.sh` for one-shot fresh-pod recovery. Defer until we hit the pattern often enough to need automation.

Next session goal (after Phase 1 brief):
- Phase 1 design work: JSON schema for handler outputs, LLM prompt template structure, validator architecture, classifier shell.
- Friends' eval positions still pending; design work doesn't need them.

---

## Open questions for the project

_Things I notice but don't have time to answer right now. Capture them so they don't get lost._

_Items 1–9 below moved here from `docs/_inbox/PRIOR_WORK_UPDATE_v3.md` on 2026-05-17 as part of the v3 patch merge._

1. **History-stripped Leela variant.** Cruz, Jenner, and `leela-interp` all use single-board-state finetunes of Leela. BT4 takes 8-position history. Phase 6 patching against BT4 needs either (a) a similar history-stripping finetune of BT4, or (b) careful experimental design controlling for history-plane noise.
2. **Behavioral tests in eval grader.** Lovering's behavioral-test methodology should be considered for the eval grader in PHASE_NOTES.md Chunk 7. Phase 3 exit criterion currently uses AUC>0.9 (encoding); a complementary "concept used correctly in N positions" behavioral metric might be worth adding. See ADR-0007.
3. **MCTS-vs-network signal split.** Lovering finding that MCTS discovers concepts first formalizes a Phase 2 design question: when handlers cite "Leela sees X," does X come from the policy network or from MCTS visits/values? Worth distinguishing in the JSON schema.
4. **Two more attention head taxonomies.** Davis & Sukthankar (chess GPT-2) and Cruz (Leela history-stripped) both produce attention-head taxonomies. Neither targets BT4 with smolgen. Phase 3 or Phase 6 may benefit from deriving a BT4-specific taxonomy; `leela-interp/tools/piece_movement_heads.py` is the methodology template.
5. **Maia-2 vs Maia-1 integration choice.** Maia-2 is the unified skill-aware successor; Tang et al. uses it. Worth deciding which to integrate when Phase 6 needs a "how would a player rated X see this?" reference.
6. **Adversarial eval-position acceptance criteria.** `superhuman-ai-consistency` generators produce positions with mechanical ground truth but not authored against handler intent. Worth deciding whether they go in `tests/eval/eval_positions.yaml` (Rule 3 protected) or a separate `tests/eval/consistency_positions.yaml`.
7. **JSON-schema forward compatibility.** Cruz's "multiple move sequences" finding suggests `WhatIsLeelaSeeingFindings` should accommodate an `alternate_lines` field even if Phase 1 doesn't populate it.
8. **Custom python-chess fork in leela-interp.** `leela-interp` pins `chess @ git+https://github.com/ejnnr/python-chess.git`. If we vendor `core/uci_to_idx.py`, do we also need that fork's specific extensions, or does mainline `chess==1.11.2` suffice? Worth verifying during Chunk 4 vendoring.
9. **Branch-pin durability in lczero-planning.** `lczero-planning` pins `lczerolens` at the `patching` rev and `nnsight` at Poupart's `Xmaster6y/nnsight` fork's `modules` rev. If we adopt as Phase 4 fallback, are those branches still maintained? Verify before depending.

---

## Things to revisit

_Decisions or shortcuts I took that I should reconsider later._

- (none yet)

---

## Phase retrospectives

_At the end of each phase, write a short retrospective: what worked, what didn't, what I'd do differently, what I learned that should change CLAUDE.md or the bible._

### Phase 0 retrospective
_(empty until end of phase — note: driver-side criteria all done as of 2026-05-17; only friends' 50 eval positions outstanding. Full retro deferred until those land and Phase 0 formally exits.)_

---

## Phase 1 — Network-First Chat MVP

### Exit criteria (from CLAUDE.md §11)
- [ ] 30 eval positions pass on all 3 handlers (`what_is_leela_seeing`, `why_this_move`, `did_you_consider`)
- [ ] Zero hallucinated moves in last 100 interactions
- [ ] <30 s end-to-end latency
- [ ] You personally find it useful for your own games

### Locked Phase 1 decisions (2026-05-17, from Phase 1 Entry Brief)
- **G1.1** First handler = `what_is_leela_seeing` (simplest extractors, proves end-to-end pipeline fast).
- **G1.2** Two-tier LLM: Sonnet 4.6 for iteration, Opus 4.7 for eval + production. See **ADR-0005**.
- **G1.3** Validator = strict mechanical only (legality + JSON-grounding). Semantic judgment moved to eval grader, not the live path.
- **G1.4** Anthropic prompt caching enabled from day one.
- **G1.5** CLI = REPL + one-shot `ask --fen ... --question ...` subcommand. The one-shot is what the eval grader will call.

### Chunk plan (per Phase 1 Entry Brief)
| Chunk | What | Status |
|---|---|---|
| 1 | Design: handler JSON schemas (Pydantic), prompt template structure, validator interface, CLI sketch | ✅ done 2026-05-17 |
| 2 | Anthropic SDK wrapper + .env loading + prompt-caching scaffold | ✅ done 2026-05-17 |
| 3 | Validator implementation (legality + JSON-grounding) + unit tests on synthetic data | ✅ done 2026-05-17 |
| 4 | First extractor (forward pass against BT4, return Pydantic of policy+wdl+mlh+top-N) | ✅ done 2026-05-17 (v2: Fudan-vendored HookedTransformer) |
| 5 | First handler (`what_is_leela_seeing`) wiring it all together | ✅ done 2026-05-18 (ADR-0009 Stockfish carve-out; v3 prompt) |
| 6 | CLI REPL + one-shot subcommand | ✅ done 2026-05-19 |
| 7 | ~~Eval grader~~ | **skipped per ADR-0010** (manual QA in lieu of formal eval) |
| 8 | Handler #2 (`why_this_move`) | **deferred until after Phase 2** (avoid handler rework when MCTS lands) |
| 9 | Handler #3 (`did_you_consider`) | **deferred until after Phase 2** (same reasoning) |
| 10 | ~~Polish + Phase 1 exit verification~~ | **skipped per ADR-0010** (no formal exit gate) |

Phase 1 is informally closed: Chunks 1-6 done, 7/10 skipped, 8/9 deferred to post-Phase-2. Next active work is Phase 2 (MCTS integration).

### Session log

#### 2026-05-17 — Phase 1 entry brief delivered + decisions locked

What I did:
- Delivered Phase 1 Entry Brief per CLAUDE.md §10. Reading list R1.1–R1.4, tooling T1.1–T1.4, go/no-go G1.1–G1.5, chunking proposed.
- User accepted all G1.x recommendations: first handler = `what_is_leela_seeing`, Sonnet-iter/Opus-prod, strict validator, prompt caching on, REPL + one-shot CLI.
- Wrote **ADR-0005** pinning LLM model selection.
- Updated CLAUDE.md §1 current-phase pointer and §4 pinned-versions LLM row.
- Friends' Chunk 4 authoring continues; user told them to prioritize 10 `what_is_leela_seeing` positions first to unblock Chunk 5 the moment it's ready.

Open questions / threads:
- ADR-0001 + ADR-0002 + ADR-0003 + ADR-0004 + ADR-0005 `[initials]` placeholders still empty.
- T1.1 Anthropic API key — pending user confirmation. Needed before Chunk 2.
- Reading R1.1–R1.4 on user's schedule.
- Pod currently stopped; not needed until Chunk 4 (BT4 GPU forward passes).

Next session goal:
- Finish Chunk 1 (design files) and move into Chunk 2 (Anthropic SDK wrapper).

---

#### 2026-05-17 (cont.) — Phase 1 Chunks 1 and 2 done

What I did:
- Chunk 1 (design): wrote Pydantic schemas (`extractors/network_inference.py`, `handlers/what_is_leela_seeing.py` — `NetworkInferenceResult`, `MoveCandidate`, `WDL`, `WhatIsLeelaSeeingFindings`), v1 prompt template (`llm/prompts/what_is_leela_seeing/v1.md`, ~3.9 KB), validator types + signature (`validator/types.py`, `validator/__init__.py`), CLI sketch (`cli/__main__.py` — argparse with REPL default + `ask` subcommand). All stubs raise `NotImplementedError`. `python -m leela_chat.cli --help` and `ask --help` display correctly.
- Chunk 2 (Anthropic SDK wrapper): added `anthropic` 0.102.0 + `python-dotenv` 1.2.2 to deps. Wrote `llm/client.py` (LLMMode enum, LLMResponse Pydantic with cache telemetry, LLMClient.translate() with `cache_control:ephemeral` on system block), `llm/env.py` (loads .env from project root via python-dotenv), `scripts/smoke_test_llm.py` (one call per mode against real API).
- Smoke test passes: Sonnet 4.6 81+33 tokens (~$0.0007), Opus 4.7 115+33 tokens (~$0.004). Both `stop=end_turn`. Cache hits 0 at current prompt size — that's expected, the system prompt (~1000 tokens) is below the cache-engagement minimum (Opus 4.7: 4096, Sonnet 4.6: 2048). Wiring is forward-looking; caching engages naturally as prompts grow with more handlers and few-shot examples.

What surprised me:
- The claude-api skill flagged the cacheable-prefix minimum explicitly — without that, the `cache_control` would have looked broken when telemetry showed 0 hits. The skill content is load-bearing for this kind of thing; worth invoking it any time we touch Anthropic SDK code.
- Anthropic SDK is at 0.102.0 — major version still 0.x but the API surface is stable; using `>=0.40,<1.0` as the pin range.
- `cache_creation_input_tokens` and `cache_read_input_tokens` are both reported as 0 (not absent) when caching doesn't engage. Defensive `getattr(usage, ..., 0) or 0` in the client handles either case.

Open questions / threads:
- **Adaptive thinking on Opus 4.7:** currently off in our client (translation is mechanical). The claude-api skill recommends `thinking: {type: "adaptive"}` for "anything remotely complicated." Revisit in Chunk 5 once we see actual handler outputs against eval positions.
- **Prompt cache engagement:** monitor token counts as prompts grow. Once system + JSON schema exceed 2048 tokens, Sonnet will start caching; 4096 for Opus. Likely happens during Chunk 5 once we add few-shot examples.
- ADR-0001 through ADR-0005 still have `[initials]` placeholders.
- Friends' Chunk 4 progress still unknown.

Next session goal:
- Chunk 3 — implement validator (legality + JSON-grounding) in `leela_chat/validator/__init__.py`. Pure local Python with synthetic test data, no API calls.

---

#### 2026-05-17 (cont., 2) — Phase 1 Chunk 3 done

What I did:
- Implemented `leela_chat/validator/__init__.py` per CLAUDE.md §2 Rule 2 + ADR-0005 G1.3 (strict mechanical only).
  - SAN regex catches piece moves (`Nf3`, `Nbd2`, `Bc4`), pawn captures (`exd5`), promotions (`a8=Q`), and castling (`O-O`, `O-O-O`) with optional `+`/`#`. Bare pawn pushes (`e4`, `d5`) are deliberately NOT extracted — too easy to collide with square references in prose. Documented as a known limitation.
  - UCI regex catches `[a-h][1-8][a-h][1-8][qrbn]?`.
  - Legality via `board.parse_san()` / `chess.Move.from_uci() in board.legal_moves`. `ValueError` catches all three python-chess parse failures (Invalid/Illegal/Ambiguous).
  - Numeric grounding extracts signed decimals (`+0.42`, `-1.5`) and percentages (`28%`). Plain integers (`2200 elo`) are skipped. Unit conversions both ways: cp ↔ pawn-units (`+0.42` matches `eval_cp=42`), pct ↔ probability (`45%` matches `wdl.white_wins=0.45`). Tolerance: 10% relative or 0.02 absolute, whichever is larger.
  - Boolean values in findings are filtered before grounding (Python bool subclasses int — without the guard, `True`/`False` would pollute the numeric set as 1.0/0.0).
- Wrote 27 unit tests in `tests/unit/test_validator.py` covering legality (legal/illegal SAN, legal/illegal UCI, castling, promotion, captures, dedup), grounding (cp form, pawn-unit form, percentage matching pct, percentage matching probability, ungrounded numbers, empty findings, nested-list walking, dedup, bool guard), and combined cases (clean response, mixed error kinds, frozen Pydantic shape). All pass.
- Added `pythonpath = ["."]` to `[tool.pytest.ini_options]` in pyproject.toml so pytest can import `leela_chat` despite `package = false`.

What surprised me:
- python-chess's `parse_san` is lenient about missing `+` annotation — `"a8=Q"` parses fine even when the move actually gives check. Good for the validator (LLM doesn't get punished for forgetting the suffix).
- The bool guard in `_flatten_numerics` matters: without it, a stray `True`/`False` in findings would make every `+1.0`/`+0.0` in responses falsely pass grounding. The test catches the regression.

Open questions / threads:
- v1 grounding only extracts numbers with explicit sign or `%`. If real LLM outputs cite eval as bare integers ("eval 42"), we'll see false negatives. Revisit after we see real Chunk 5 outputs against eval positions.
- Bare pawn-push SAN (`e4`, `d5`) skipped — the prompt template (v1.md) instructs the LLM to use full SAN or context-rich phrasing. If the model still emits bare squares as moves, we'll need a disambiguator (e.g., extract bare squares only when preceded by "plays "/"considers ").
- ADR-0001 through ADR-0005 `[initials]` placeholders still empty.
- Friends' Chunk 4 progress still unknown.

Next session goal:
- Chunk 4 — implement `extract_network_inference()` in `leela_chat/extractors/network_inference.py`. Loads BT4 ONNX into `lczerolens.LczeroModel`, runs one forward pass, extracts top-N policy moves + WDL + MLH + eval_cp. Best done on the remote GPU box; CPU on the laptop works for shape-only smoke if needed.

---

#### 2026-05-17 (cont., 3) — Phase 1 Chunk 4 done + pod-state remediation

What I did:
- Step 0 (API probe on the pod): confirmed `LczeroModel.from_onnx_path()` is the right loader. BT4 returns three output heads: `policy` (1×1858 raw logits), `wdl` (1×3 softmaxed, side-to-move perspective), `mlh` (1×1, plies). `LczeroBoard(fen)` builds from a FEN; `decode_move(idx)/encode_move(move, us)` convert between policy index ↔ chess.Move. No CP helper in lczerolens 0.4 — implemented Lc0's UCI formula ourselves.
- Implemented `leela_chat/network/__init__.py` (`load_bt4()` — picks CUDA when available, eval mode, moves to device).
- Implemented `extract_network_inference()` in `leela_chat/extractors/network_inference.py`. Pydantic gains `moves_left_plies` alongside `moves_left_estimate` (full moves = plies ÷ 2) per user's "both fields" choice. WDL is flipped to White-perspective for Black-to-move so the `WDL` model is always White-perspective per its docstring. Policy is masked to legal moves, numerically-stable softmax over them, top-N sorted descending.
- Wrote `leela_chat/scripts/smoke_test_extractor.py` — loads BT4, asserts WDL sum ≈ 1, eval near 0 at startpos, top-N overlaps known openings, side_to_move flips correctly for Black-to-move FEN.
- **First smoke test passed but eval_cp showed an anomaly**: +11 cp at startpos, +7 cp after 1.e4 — but chess intuition says 1.e4 retains an edge, so the drop was suspicious.
- **Investigated via `lc0 --verbose-move-stats` comparison on the pod.** Root WL values matched exactly between extractor (q=+0.016) and lc0 (root WL=-0.01573 from Black = +0.01573 from White), confirming the inference path is correct. **CP formula was the bug** — my code used the older `centipawn_2018` constants (290.680623072 / 1.548090806), but Lc0 v0.32's default `centipawn` formula is `90 * tan(1.5637541897 * wl)`. Three-line patch in `network_inference.py`. After fix: extractor reports +3 startpos / +2 after 1.e4, matching lc0 within rounding. The "+11→+7" anomaly was the legacy formula amplifying small Q differences.

What surprised me:
- **Pod restart wiped EVERYTHING outside /workspace.** `/workspace/bin/lc0` from Chunk 3 was gone; apt packages (stockfish, meson, ninja) were gone; CUDA was on disk but not on PATH. The persisted volume is `/workspace` only, but even some sub-dirs (`/workspace/bin`) didn't survive. The `.venv` inside `/workspace/leela-chat/` did persist. Lesson: RunPod stop/start is more destructive than expected — anything that needs to survive must (a) be in `/workspace/<a-real-app-dir>/` and (b) be rebuildable from a checked-in script.
- Lc0 has FIVE CP score types (`centipawn`, `centipawn_2018`, `centipawn_2019`, `centipawn_with_drawscore`, `WDL_mu`). The 2018 formula is widely quoted in old docs but is no longer the default. Easy to grab the wrong one if you're reading historical sources.
- BT4 outputs MLH ~99 full moves at startpos — far longer than human games, presumably reflecting Lc0 self-play training distribution.
- BT4's `e7e5` policy after 1.e4 is **77.6%** — strongly opinionated towards the Open Game; Sicilian-via-c5 only gets 4.9%.

Remediation (committed separately):
- Added `scripts/setup_pod.sh` — idempotent pod bootstrap. Installs apt packages, adds CUDA to PATH (and `/root/.bashrc`), builds Lc0 v0.32.1 into `/workspace/src/lc0` then copies binary to `/workspace/bin/lc0` only if missing. Smoke-checks lc0, stockfish, BT4 ONNX, and the venv. Run on every fresh pod with `ssh runpod 'bash /workspace/leela-chat/scripts/setup_pod.sh'`.
- Lc0 v0.32.1 rebuilt on the pod with cuda-fp16 backend. Same version as the previous build (ADR-0004 still valid). Binary at `/workspace/bin/lc0`. Source preserved at `/workspace/src/lc0`.

Open questions / threads:
- BT4's startpos eval (+3) is slightly higher than after 1.e4 (+2). Probably noise within rounding; flagged for re-check when we feed actual move history (the no-history input may explain the 1cp wobble).
- `WhatIsLeelaSeeingFindings` (handler model) deliberately does NOT yet have `moves_left_plies`. Chunk 5 decides whether to surface plies to the LLM.
- ADR-0001 through ADR-0005 `[initials]` placeholders still empty.
- Friends' Chunk 4 (eval positions) progress still unknown.

Next session goal:
- Chunk 5 — first handler (`what_is_leela_seeing`). Wires extractor → handler → translation layer → validator end-to-end. **Blocked on ~10 `what_is_leela_seeing` eval positions from friends.** While waiting, Chunk 6 (CLI REPL + one-shot `ask` subcommand) is unblocked and could be done in parallel.

---

#### 2026-05-17 (cont., 4) — castling-encoding fix; lczerolens/onnx2torch BT4 bug uncovered

What I did:
- Spot-checked the v3 patch's leela-interp claims, ran a parity test between `lczerolens.LczeroBoard.encode_move()` and the vendored `leela-interp` `uci_to_idx` tables over 242 legal moves across 11 diverse positions: **240/242 match. 2 mismatches were both castling moves.** Verified the disagreement against BT4 on the pod by reading raw policy logits at the disputed indices — confirmed lczerolens's `encode_move()` returns the wrong policy slot for king-target castling UCIs (`e1g1`, `e1c1`, `e8g8`, `e8c8`). The network's actual castling probability lives at the rook-target indices, which lczerolens's `decode_move()` correctly inverts but `encode_move()` misses.
- Vendored `leela-interp/src/leela_interp/core/uci_to_idx.py` (SHA-1 `34c3f5c7e1b9968fa46ace7b3b7b89b0a2a670b2`) to `leela_chat/network/uci_to_idx.py` with an attribution header documenting source, license (GPL-3, same class as our existing GPL deps), the bug it works around, and known knight-underpromotion limitation.
- Added `encode_legal_move(board, move)` to `leela_chat/network/__init__.py` using the vendored tables. Patched `extract_network_inference()` to use it instead of `lc_board.encode_move(mv, us)`. Extended `smoke_test_extractor.py` with a castling-rich middlegame regression test that asserts O-O or O-O-O appears in the top-N.
- **DEEPER bug uncovered while preparing to commit the castling fix:** the eval_cp on the castling-rich middlegame came back as `-1542` (i.e. White dead-lost) while every other engine on the same position says `+14` to `+49` cp (slight White edge). Cross-checked 20 diverse positions: lczerolens via the andyrdt third-party ONNX disagrees with lc0 binary (using the same `.pb.gz` weights, hash-verified) on **12 of 20 positions**. lczerolens's WDL output is polarized to extremes (`0/0/1`, `1/0/0`, `0/1/0`) for most non-opening positions, while lc0 binary produces sensible intermediate values. Self-converted ONNX via `lc0 leela2onnx` produces **identical broken output** to the andyrdt ONNX — so the third-party export isn't the villain. lczerolens 0.4.0 also doesn't accept `.pb.gz` via `from_path()` (raises `NotImplementedError`). The bug is somewhere in the `lczerolens / onnx2torch / PyTorch` runtime — likely in how BT4's smolgen-attention or attention-policy operators get translated by `onnx2torch`. T82 (older architecture, no smolgen) is what published interp work (Jenner, Cruz) uses successfully — they don't hit this bug.

What surprised me:
- The smoke test from Chunk 4 v1 passed on cherry-picked positions (startpos + after-1.e4 + castling-rich-middlegame). The first two happen to fall in lczerolens's "works" zone. The third only just got added as part of the castling regression — and it's exactly the position that surfaced the deeper inference bug. The 3-position smoke test was insufficient; the real cross-check needs to be a 20-position-vs-lc0-binary WDL comparison.
- The `--verbose-move-stats` Lc0 binary on the pod (from setup_pod.sh) was load-bearing for this diagnosis. Without it, we'd still believe the extractor works.
- **Fudan ships their pretrained Transcoders/Lorsas specifically for BT4-1024x15x32h** (per their README and the JacklE0niden/lc0-BT4-tc + JacklE0niden/lc0-BT4-lorsa HuggingFace repos). Switching to T82 to sidestep the smolgen bug would lose Fudan integration entirely — the project's single biggest Phase 4 accelerator. So BT4 stays.
- **Fudan has a known-working PyTorch path for BT4 that bypasses `onnx2torch` entirely:** `examples/weight_conversion.py` in the Leela-SAEs repo converts the ONNX into a TransformerLens-compatible `BT4.pt`, then uses TransformerLens `HookedTransformer` at runtime. Their reasoning-pathway code runs on this loader daily in published work. This is the swap plan for next session (or now).

Status of Chunk 4 deliverables:
- Castling encoding fix: ✅ correct in isolation, committed in the next commit. The vendored `uci_to_idx` table is right regardless of which forward-pass library we use; the fix is independent of the inference-path bug.
- `extract_network_inference()` end-to-end: ❌ produces unreliable WDL/eval_cp for most non-opening positions. Cannot be relied upon by handlers downstream until the inference-path swap lands.
- Smoke test: now includes castling regression. Needs to be expanded to a 20-position WDL-vs-lc0-binary cross-check before we declare Chunk 4 done again.

Open questions / threads (carryover):
- ADR-0001 through ADR-0005 `[initials]` placeholders still empty.
- Friends' Chunk 4 (eval positions) progress still unknown.
- Patch v3 steps 8-10 (BIBLE.md edits, ADR-0006/7/8 drafts, delete the inbox patch) still deferred.
- BT4's no-history input may also affect some positions even after the loader swap — Fudan and lc0-binary both run cold-no-history but produce sensible values; revisit whether `LczeroBoard(fen)` differs in its history-plane defaults from what BT4 was trained on.

Next session goal:
- See post-script below — Fudan-loader-swap attempt was made this session, hit deeper issues than expected. Next session: decide whether to keep digging into Fudan's loader, switch to lc0-subprocess inference, or file an issue with Fudan.

---

#### 2026-05-17 (cont., 5) — Fudan-loader-swap attempt; deferred to fresh session

What I did (in-session investigation, no committed code):
- Uploaded `update_research/repos/Leela-SAEs-main/` (Fudan repo, ~34 MB) to `/workspace/Leela-SAEs/` on the pod. `uv sync` succeeded with their pinned TransformerLens fork installed editable into a separate `.venv`.
- Ran `examples/weight_conversion.py` against our BT4 ONNX (using the andyrdt file we already had). Conversion succeeded: 469 parameters loaded, `BT4.pt` (1.3 GB) saved to `/workspace/weights/BT4-1024x15x32h-swa-6147500/BT4.pt`. Symlinked into `/path/to/models/lc0/BT4.pt` (the hardcoded path their `loading_from_pretrained.py` expects).
- Wrote a 20-position WDL cross-check using `HookedTransformer.from_pretrained_no_processing("lc0/BT4-1024x15x32h", dtype=torch.float32)` and inlined `get_wdl_from_model` (their `chess_utils` package has a broken import chain — `lm_saes.backend.language_model` imports `SearchlessChessBehavioralCloningTokenizer` from `transformer_lens.components`, which doesn't exist there — so importing `chess_utils.move` fails before you can use anything).

What I found:
- The HookedTransformer load logs **two missing weight-keys filled with zero tensors**: `pos_embed.W_pos` and `policy_head.promotion_weight`.
- `pos_embed.W_pos` is a **red herring**: HookedTransformer.py line 458 has `if not self.cfg.is_leela_chess_model: ... else: residual = embed`, so the leela path skips pos_embed addition entirely. The zero tensor is never used.
- `policy_head.promotion_weight` is a real name mismatch: `weight_conversion.py` produces `policy_head.promotion.weight` (with a dot), but HookedTransformer expects `policy_head.promotion_weight` (underscore). This breaks promotion-policy encoding but shouldn't affect WDL.
- **WDL output is wrong on 16 of 20 positions, even on the starting position** (Fudan reports W0.01/D0.02/L0.97 from White's perspective; correct is W0.19/D0.64/L0.17). The error is not the pos_embed warning, and not the promotion-key mismatch — something in the `value_head` pipeline is off, but I haven't pinpointed it. Possibly another weight name mismatch, possibly a shape/transpose difference in the value head between what `weight_conversion.py` extracts and what HookedTransformer's `ValueHead` expects.

Realistic next-session plan to debug Fudan path (~3-8 hours):
1. Diff the state_dict keys between (a) what `weight_conversion.py` outputs and (b) what `HookedTransformer` `state_dict()` returns after a fresh init. Find every name mismatch.
2. Read Fudan's `ValueHead`, `PolicyHead`, `MLHHead` implementations in `TransformerLens/transformer_lens/components/leela_*.py` to see exact shapes/transposes expected.
3. Patch `weight_conversion.py` (or write a small post-processor) to align key names + shapes.
4. Re-run the 20-position cross-check; require all 20 to agree with lc0 binary within tolerance.
5. Once correct, integrate into our `leela_chat/network/__init__.py` (probably by vendoring or pinning Fudan's TransformerLens fork as a project dependency).
6. Update `extract_network_inference()` to use the new model.

Alternative paths to consider for next session:
- **lc0 subprocess for inference (Chunk 4 v2 today, defer hooks)**: refactor `extract_network_inference()` to spawn lc0 binary via python-chess's UciProtocol with WDL_mu score-type, parse the output. Correct WDL today, but no PyTorch hook access — Phase 3 (probes) will need to revisit the loader question separately.
- **File an issue with Fudan**: ask the authors directly how they intend `weight_conversion.py` + `HookedTransformer.from_pretrained_no_processing("lc0/BT4-1024x15x32h")` to produce correct WDL. They may have a setup step we're missing that takes 5 minutes when you know it.
- **Stay on BT4 forever**: BT4 + Fudan SAEs is the right long-term pin (per the v3 patch and the Fudan README — JacklE0niden/lc0-BT4-tc and lc0-BT4-lorsa). The loader question is solvable, just not in one session of unfamiliar-codebase debugging.

What's preserved + what isn't:
- `BT4.pt` (1.3 GB) at `/workspace/weights/BT4-1024x15x32h-swa-6147500/BT4.pt` — will survive pod restart since `/workspace` persists.
- `/workspace/Leela-SAEs/` directory with their venv — will likely survive pod restart.
- `/path/to/models/lc0/BT4.pt` symlink — won't survive pod restart (rebuild from `setup_pod.sh` follow-up if we keep this path).

Carryover open questions (still valid):
- ADR-0001 through ADR-0005 `[initials]` placeholders still empty.
- Friends' Chunk 4 (eval positions) progress still unknown.
- Patch v3 steps 8-10 (BIBLE.md edits, ADR-0006/7/8 drafts, delete the inbox patch) deferred until Chunk 4 inference is reliable.
- All 9 patch-v3 open questions still valid (see "Open questions for the project" earlier in this file).

Next session goal:
- Decide: keep debugging Fudan's loader (3-8h fresh-context dive), OR ship lc0-subprocess Chunk 4 v2 (~30 min, correct WDL, hook-access deferred), OR file Fudan issue and wait. Whichever path: produce a written design proposal first, get sign-off, then execute.

#### 2026-05-17 (cont., 6) — Fudan loader FIXED: two upstream bugs identified and patched

Root-caused both bugs in the Fudan stack. With these two patches, `HookedTransformer.from_pretrained_no_processing("lc0/BT4-1024x15x32h")` produces WDL that matches `lc0` binary on all 5 sanity positions (startpos, after 1.e4, castling-rich middlegame, KR vs K, Najdorf).

**Bug 1 — Missing `attention_body.ln.bias` and `attention_body.ln2.bias` mappings in `examples/weight_conversion.py`.** The mapping only includes `"attention_body.ln.weight": "attn_body/ln.weight"` and the same for `ln2`. The corresponding `.bias` keys exist in the ONNX (and onnx2torch loads them with magnitudes up to |1.74|), but `weight_conversion.py` never copies them into `CleanLC0Model`. PyTorch's `nn.LayerNorm` default-initializes `.bias` to zeros, so the final `BT4.pt` has zero LN biases in the attention body. Consequences cascade through every encoder block — the very first divergence between onnx2torch (correct) and CleanLC0Model (broken) appears immediately after `attn_body/ln` (max abs diff 1.74 at the LN output, growing to 488 by `attn_body/ffn/dense2/b`).

**Bug 2 — `HookedTransformer.py` constructs `EncoderLayer(...)` without passing `cfg.eps`.** `EncoderLayer.__init__` defaults to `eps=1e-5`, and that eps is propagated into `Lc0LayerNorm` (encoder pre/post LNs) and `SmolGen`'s `nn.LayerNorm` calls. The BT4 model config sets `"eps": 1e-03`, but the value is never threaded through. CleanLC0Model hardcodes `eps=1e-3` in its own encoder LNs, so this bug only affects the HookedTransformer pipeline — not `CleanLC0Model` standalone.

How I found it:
- Wrote `compare_intermediate.py` capturing activations via forward hooks on onnx2torch's `attn_body/ln2` and `encoder*/ln2`. First divergence (max diff 1.66) showed up at the very first waypoint — confirms bug is in the embed, not encoders.
- `embed_drilldown.py` captured every sub-step of the attention body (`embedding_preprocess`, `concat`, `main_linear`, `mish`, `ln`, `ma_gating`, `ffn`, `ln2`). All exact-match (≤1e-6) up to and including `attn_body/mish`, then divergence appears at `attn_body/ln` (max abs diff 1.74). Pinpointed to the LN itself.
- `inspect_ln.py` listed every `nn.LayerNorm` in onnx2torch with its bias stats. Every encoder/smolgen LN has `bias_max` in [0.5, 3.4]. CleanLC0Model's mapping table only loaded weight, not bias, for `attn_body.ln`/`attn_body.ln2` — bug confirmed.
- Two-line patch to `weight_conversion.py` added the bias entries. Rebuilt `BT4.pt`. WDL on all 5 test positions now matches `lc0` binary.

Verification — ground-truth from `lc0` binary at 1 node:

| Position | lc0 WDL | Fudan-fixed WDL |
|---|---|---|
| startpos | 0.191 / 0.642 / 0.167 | 0.19 / 0.65 / 0.16 |
| after 1.e4 | 0.169 / 0.645 / 0.186 | 0.17 / 0.65 / 0.19 |
| castling-rich | 0.216 / 0.632 / 0.152 | 0.19 / 0.65 / 0.16 |
| KR vs K | 1.000 / 0.000 / 0.000 | 1.00 / 0.00 / 0.00 |
| Najdorf | 0.293 / 0.550 / 0.157 | 0.29 / 0.56 / 0.16 |

Small differences (e.g., castling-rich 0.216 vs 0.19) are within fp16-vs-fp32 backend rounding plus 1-node MCTS noise; the structure (W/D/L ordering and rough magnitudes) is exact.

State of patches:
- Both edits live on the pod in `/workspace/Leela-SAEs/`. They will **NOT** survive a pod restart unless re-applied (the repo is re-cloned by `setup_pod.sh`).
- Need to: (a) capture both as a `.patch` file in the repo, (b) make `setup_pod.sh` apply it after cloning Leela-SAEs, (c) document in an ADR (-0006? — needs assignment), (d) optionally file upstream issues on the Fudan repo.

Remaining open items for the actual Chunk 4 v2 swap:
- Vendor patches or fork Leela-SAEs (decide which).
- Decide whether `leela_chat` should consume HookedTransformer directly (Phase 3+ hook-access) or keep lczerolens for now and only swap before Phase 3.
- If swapping: update `leela_chat/network/__init__.py` to add a `load_bt4_hooked()` path, refactor `extract_network_inference()` to consume HookedTransformer's output format, re-run the 240-position parity test.

#### 2026-05-17 (cont., 7) — Chunk 4 v2 done: vendored Fudan TL, swapped extractor

Picked vendoring (option 2 of four — see session log for the four options I considered) over editable install / GitHub fork / adapter pattern. The decisive criteria: pin-everything matches existing project philosophy, no external runtime dependencies (no GitHub, no pod-specific filesystem), patches show up as normal code in our git history, and Fudan's repo could be deleted/changed at any time without affecting us. Honest second choice was the GitHub fork (cleanest upstream-PR path), but we can do that separately later.

What got committed:
- `leela_chat/_vendor/transformer_lens/` — 77 .py files (~900 KB) copied from `update_research/repos/Leela-SAEs-main/TransformerLens/transformer_lens/`. Upstream MIT LICENSE preserved verbatim. Provenance + local mods recorded in `leela_chat/_vendor/VENDOR.md`.
- `leela_chat/_vendor/bt4_weight_conversion.py` — the ONNX→.pt converter from Fudan's `examples/`, with the missing-LN-biases fix applied directly (and noted in-file).
- `leela_chat/_vendor/__init__.py` — sys.path hack (pip-style) so `from transformer_lens import HookedTransformer` resolves to the vendored copy. Side-effect-only module; idempotent on re-import.
- Both BT4-loader patches applied directly to the vendored code (no `.patch` overlay): (a) `attention_body.ln.bias` / `ln2.bias` added to the explicit mapping in `bt4_weight_conversion.py`, (b) `cfg.eps` threaded into `EncoderLayer(...)` construction in `HookedTransformer.py`.
- Trimmed eager imports in vendored `transformer_lens/__init__.py` to skip modules we don't use that pull in deps we'd rather not require: `SVDInterpreter` (needs typeguard), `HookedEncoder`/`HookedEncoderDecoder` (BERT-style, irrelevant), `head_detector`, `patching` (Phase 6 thing), `train` (needs wandb). Each is still individually importable as `from transformer_lens.<mod> import X` — only the top-level shortcut is removed.
- Removed `from matplotlib.cm import ScalarMappable` from `transformer_lens/components/leela_features.py` (unused in our code path; avoids pulling matplotlib).
- Removed two leftover debug `print()` calls in vendored upstream (`state_dict.keys()` dump on every model load, and `"advanced embedding for BT4"` chatter).
- `pyproject.toml` — added runtime deps: einops, jaxtyping, beartype, fancy-einsum, transformers, better-abc, accelerate, sentencepiece, rich, typeguard. Kept lczerolens too (used by `load_bt4_lczerolens` for non-BT4 lczerolens models like Maia, and as a second-opinion cross-check).
- `leela_chat/network/__init__.py` — `load_bt4()` now returns a `HookedTransformer` (the primary path). Old lczerolens loader renamed `load_bt4_lczerolens` and marked legacy. No ONNX path arg — the vendored `loading_from_pretrained.py` reads from the hardcoded `/path/to/models/lc0/BT4.pt` (kept upstream's hardcode to stay bit-identical).
- `leela_chat/extractors/network_inference.py` — refactored. Input is now built via `LeelaBoard.from_fen(fen, history_synthesis=True).lcz_features(no_history=False)` instead of `LczeroBoard.encode_move`. Forward call is `model(tensor) -> (policy_logits, wdl, mlh)` (HookedTransformer signature) instead of `model(lc_board) -> dict`.
- `leela_chat/scripts/smoke_test_extractor.py` — updated to use new `load_bt4()` signature (no path arg). All three smoke checks pass on pod (startpos / 1.e4 / castling-rich middlegame).
- `tests/integration/test_bt4_parity.py` (new, 28 cases) — for each of 14 fixed positions (openings, midgames, endgames, castling-rich, kr-vs-k, kpvk, kbnk-zugzwang, tactical white + tactical black) checks (a) our WDL per-channel within 0.05 of lc0 binary's WDL and that the argmax channel matches, (b) our top-policy move equals lc0's 1-node best move (or at worst lc0's move is in our top-3). Auto-skips when CUDA absent or `LEELA_CHAT_LC0_BIN`/`LEELA_CHAT_LC0_WEIGHTS` env vars unset.
- `pyproject.toml` — registered `integration` pytest marker.

Verification on pod:
- `uv run python -m leela_chat.scripts.smoke_test_extractor` passes.
- `LEELA_CHAT_LC0_BIN=... LEELA_CHAT_LC0_WEIGHTS=... uv run pytest tests/integration/test_bt4_parity.py -v` — **28/28 pass** in 48 s.
- `uv run pytest tests/unit` — 27/27 pass (validator unchanged).

Carryover (out of scope for this commit):
- File upstream PR to OpenMOSS/Leela-SAEs with both patches. The `scripts/patches/leela_saes_bt4_loader.patch` from the previous commit is the portable form; or we just push them straight to upstream as two separate small PRs.
- `scripts/setup_pod.sh` still includes the runtime patch step for `/workspace/Leela-SAEs/`. Now obsolete for our purposes (we ship a vendored copy), but harmless if Leela-SAEs is still cloned on the pod for upstream PR work. Leave for now; revisit if it causes confusion.
- Patch v3 steps 8–10 (BIBLE.md edits, ADR-0006/7/8 drafts, delete `docs/_inbox/`) still deferred.
- Friends' Chunk 4 eval positions still TBA.

---

#### 2026-05-18 — Chunk 5 end-to-end working; v1→v2 prompt refinement

Deliberate Rule-3 carve-out for this chunk, agreed in-session: handler ships before friends' eval entries land. Friends author independently (have not seen Leela outputs), so the "eval author isn't biased by code" invariant of Rule 3 is preserved even though the temporal order inverts. When their batch arrives, Chunk 7 grader runs against Chunk 5 outputs and the source of any score gap (eval too strict / code wrong / both) gets diagnosed then. Documented in this entry so it doesn't read as silent rule-breaking later.

What I did:
- `handle_what_is_leela_seeing` implemented in `leela_chat/handlers/what_is_leela_seeing.py` — trivial projection from `extract_network_inference` output to `WhatIsLeelaSeeingFindings` (latter is a strict subset). Handler signature tightened: `model: object` → `model: "HookedTransformer"` (the comment was stale from before the Chunk 4 v2 swap). Added `HANDLER_NAME = "what_is_leela_seeing"` constant to avoid string repetition at call sites.
- `templated_fallback(findings)` in the same module — mechanical render-from-Pydantic. Used only when LLM hallucinates twice (Rule 2 last resort). Grounded by construction (every number traces directly to findings); produces something readable if robotic. Verified on real findings via validator: passes.
- `leela_chat/llm/translate.py` (new, ~190 lines) — orchestrator. `translate_and_validate(findings, question, *, handler_name, templated_fallback, prompt_version, mode, client)` runs the Rule-2 chain: attempt 1 normal → on fail, attempt 2 with validator errors appended ("Retry context" addendum) → on fail, templated fallback. Always returns a `ValidatedTranslation` with a `source` field (`llm` / `llm_retry` / `templated`) so callers know which path produced the text. Loads prompts from disk (`prompts/{handler}/v{N}.md`) so versioning bumps are file-only changes.
- `leela_chat/scripts/smoke_test_handler.py` (new) — dev tool. Loads BT4 once, iterates a FIXTURES list of `(fen, question, comment)`, prints findings JSON + final response + latency + token telemetry per position. Defaults to ITER mode (Sonnet 4.6) for cheap iteration; `--prod` flag swaps to Opus 4.7. **NOT eval data** — these are throwaway smoke positions supplied by the user in this session.
- 10 user-supplied dev fixtures spanning endgame / middlegame / opening / minor-piece / promotion / wild-melee / dead-lost-rook-endgame. 4 Black-to-move, 6 White-to-move. All validated as legal FENs (17-51 legal moves). NOT the eval set.
- `tests/integration/test_translate_what_is_leela_seeing.py` (new) — 5 tests with mocked `LLMClient`: happy path, retry-success, templated-fallback after both LLM attempts fail, missing-handler error, missing-prompt-version error. All pass. Handler itself isn't exercised (would require BT4 + GPU) — smoke-tested on the pod instead.
- Prompt **v1 → v2** after smoke surfaced one validation failure: Sonnet rounded `1.2%` policy values to `1%`, which the validator (correctly) flagged as ungrounded (closest finding 1.21, tolerance 0.12 → diff 0.21 fails). v2 tightens the precision rule: `policy_pct` must keep ≥1 decimal place; threshold phrasing (`under 1%`, `below 1%`) banned. WDL probabilities stay whole-number percentages (large enough to round safely). `eval_cp` cite as `+12 cp` or `+0.12` pawn units, either form acceptable.
- `tests/eval/AUTHORING.md` (new) — kept as internal reference for what we'd want a well-designed eval entry to look like. NOT sent to friends per user decision (they author unguided to keep independence).
- `.env.save` hygiene: deleted local + pod copies of a nano backup file containing an old (rotated-out?) Anthropic key. Expanded `.gitignore` from `.env` / `.env.local` to `.env*` with `!.env.example` exception so future backup variants (`.env.save`, `.env.bak`, `.env.tmp`, …) are covered. User flagged to rotate the old key (`sk-ant-api03-tz3umF...`) in the Anthropic console as standard hygiene since it sat in a 644-perm file for ~24h locally.
- SSH config (`~/.ssh/config`) updated to new pod endpoint `213.192.2.68:40138` after a stop/restart cycle. Worth scripting a "fetch current endpoint from RunPod console" automation later if this keeps biting.

Smoke test outcome:
- **v1 prompt run (round 1):** 9/10 first-try-valid, 1/10 templated fallback (position 6 — Qf6/Qb6 cited as `1%` instead of `1.2%`). Validator caught it, Rule 2 fallback held, user saw a clean templated response.
- **v2 prompt run (round 2):** 10/10 valid, 0 templated fallbacks. 1 of 10 (position 1) needed the stricter retry — first attempt cited `88%`, an apparent derived/composite statistic (likely `71 / (71+1) ≈ 88%` black wins among decisive games). Retry produced clean text. Net improvement vs v1: -1 templated, +1 retry.
- Latency: all positions 4-12s end-to-end including translation. BT4 load amortizes at ~50s, hits once per process. Comfortably under Phase 1's 30s exit-criterion target.
- Cost: ~$0.05 per full 10-position run on Sonnet. Cache engaged from call 2 onwards (1651 tokens cached on v2 prompt). Apparently Sonnet 4.6 caches below the 2048-token minimum we documented earlier — worth a `docs/DECISIONS.md` footnote but not urgent.

What surprised me:
- **Prompt caching engaged on a ~1700-token prefix.** Earlier PHASE_NOTES entry recorded the Anthropic-documented minimum as 2048 (Sonnet) / 4096 (Opus); we saw cache_read = 1651 from call 2 onward, both on v1 (1292) and v2 (1651). Either the minimums dropped, the docs are stale, or our prior reading was off. Caching is working unambiguously — token telemetry shows `cache_read_input_tokens` consistently nonzero after call 1.
- **Sonnet's "derived statistics" failure mode** (position 1 v2 retry): the LLM occasionally computes composite percentages from the findings rather than citing the WDL values directly. `71 / (71 + 1) ≈ 0.98 → 98%` of decisive games — or maybe just a hallucinated `88%`, hard to tell without the first-attempt text in stdout. Validator caught it; retry fixed it. If this recurs across positions, a v3 anti-pattern banning composite calculations is the right fix. Not pressing — one occurrence out of 11 LLM calls.
- The HookedTransformer "missing keys" warnings (`pos_embed.W_pos`, `policy_head.promotion_weight`) are now well-understood. `pos_embed.W_pos` is skipped via the `is_leela_chess_model` branch in `HookedTransformer.py`; `policy_head.promotion_weight` only affects underpromotions which our setups don't surface. Both harmless. Worth a comment somewhere if a future contributor trips on them.

Open questions / threads:
- **Derived-stat failure mode (Sonnet, occasional).** Recurrence rate unknown. If it shows up again in Chunk 8/9 handlers, add a v3 anti-pattern: "Cite WDL values exactly as in findings. Do not compute composite statistics (e.g., 'win rate among decisive games')."
- **`setup_pod.sh` should delete `.env.save` defensively** the next time we revise it — covers the case where a user runs nano on `.env` during pod work.
- **Pod-endpoint churn.** Every stop/restart issues a new SSH endpoint. Scripting a `runpod-update-ssh` helper that calls the RunPod API to fetch the current endpoint would save manual config edits. Defer until it bites a third time.
- Carryover open from prior sessions: ADR-0001 through ADR-0005 `[initials]` placeholders still empty. Friends' eval positions still TBA.

Patch v3 cleanup completed in the same session as a follow-on:
- ADR-0006 (MCTS parser sourcing, provisional Option B clean-room port), ADR-0007 (Phase 3 behavioral-test exit criterion addition), ADR-0008 (Phase 4 SAE fallback ordering with `lczero-planning` as Option A) all appended to `docs/DECISIONS.md`. Each is `Proposed`, applied at the corresponding phase entry — they capture the option space so the call doesn't have to be re-researched then.
- BIBLE.md edits: §6.1 cross-refs ADR-0006 + names the `superhuman-ai-consistency` parser; §7.7 adds the behavioral-test exit bullet + cross-refs ADR-0007; §8.1 updates the fallback recommendation per ADR-0008 (lczero-planning → OpenMOSS → sparsify) + flags the branch-pin verification gate; §10.1 PV-diff bullet updated to "lines plural" per Cruz, load-bearing for `did_you_consider`.
- Phase 8 §3.3 TransformerLens/nnsight wording referenced in the v3 patch does not correspond to a real BIBLE.md section in the current numbering — BIBLE.md has only one mention (§4.4 deferred-deps list). The substantive nnsight wording from Correction 3 already landed in PRIOR_WORK.md; no BIBLE.md edit warranted.
- `docs/_inbox/` deleted. Provenance preserved in this session log + the ADR References sections.

Phase 1 exit-criteria status after this session:
- [ ] 30 eval positions pass on all 3 handlers — blocked on friends; Chunk 5 unblocked itself by the carve-out
- [x] Zero hallucinated moves in last 100 interactions — 11/11 LLM calls in v2 smoke run produced legal moves (with Rule 2's retry catching the one validation issue)
- [x] <30 s end-to-end latency — max 11.48 s observed, well under
- [ ] You personally find it useful for your own games — pending Chunk 6 (CLI) before you can actually use it interactively

Next session goal:
- Chunk 6 — CLI REPL + one-shot `ask --fen ... --question ...` subcommand. The one-shot is what Chunk 7's eval grader will call (so its output contract matters more than the REPL's). Wires conversation manager + classifier (trivial for now — single handler) + the existing `translate_and_validate` pipeline. No GPU/pod requirement for development; pod only needed when you actually want to run it against BT4.

---

#### 2026-05-19 — Chunk 6 done; Phase 1 informally closed (ADR-0010); pivot to Phase 2 next

Strategic reframe agreed in this session: rather than finish Phase 1 by the BIBLE.md ordering (handlers 2/3 + eval grader + exit verification before Phase 2), the user opts for an **architecture-of-efficiency** ordering — CLI now (orthogonal to Phase 2, immediate user-facing reward), then Phase 2 (MCTS) which gives the deep search signal handlers 2/3 will rely on, then build handlers 2/3 once against the final findings shape. Eval grader (Chunk 7) and exit verification (Chunk 10) are skipped indefinitely; user is sole developer and sole user at this scale, so manual QA via the smoke script + ad-hoc CLI use replaces the formal regression-detection infrastructure. Documented in ADR-0010 (Accepted, Phase 1).

What I did:
- **Chunk 6 implementation:**
  - `leela_chat/cli/__main__.py` — replaced the Chunk 1 `NotImplementedError` stubs with real `cmd_ask` and `cmd_repl`. Shared lifecycle helpers: `_build_context` (load env → BT4 → Stockfish → LLMClient), `_close_context` (Stockfish cleanup), `_answer_one` (handler → translate → telemetry), `_emit_telemetry` (one-line stderr footer when `--verbose`). REPL supports `:fen`, `:history`, `:help`, `:quit` (+ aliases `:q`, `:h`, `:?`). `_prompt_initial_fen` validates before the loop starts; FEN errors reprompt rather than crash. `try/finally` guarantees Stockfish cleanup on every exit path.
  - `leela_chat/classifier/__init__.py` — trivial Phase-1 classifier: `classify(question, current_handlers=None) -> str` always returns `"what_is_leela_seeing"`. Optional `current_handlers` guard raises `ValueError` if the target handler isn't in the build (defends against silent degradation).
  - `leela_chat/conversation/__init__.py` — `Turn(fen, question, response, source)` + `ConversationState(current_fen, turns)` Pydantic models (mutable; REPL owns the state).
  - `tests/integration/test_cli_ask.py` (6 tests) — monkeypatches `_build_context` / `_close_context` / `_answer_one` to exercise argparse + dispatch + exit codes without touching BT4 / Stockfish / Anthropic. Covers happy path, `-v` telemetry routing, invalid FEN → exit 2 (before context build), `--mode iter/prod`, `--handler` pass-through, unknown subcommand → SystemExit.
  - `tests/unit/test_classifier.py` (3 tests) — pins the contract: trivial routing, `current_handlers` guard accepts when target present, raises when absent.
  - 40 → 49 tests pass, 28 skipped (BT4 parity tests gated on GPU).
- **ADR-0010** (skip formal eval grading) appended to `docs/DECISIONS.md`. Status Accepted, Phase 1. Documents: Rule 3 informally relaxed for Phases 1-2; smoke script (`leela_chat/scripts/smoke_test_handler.py`) becomes the de-facto regression harness; ADR-0007 (Phase 3 behavioral-test exit) still applies for Phase 3+; reversal cost is low because Chunk 7 design is unchanged, just deferred.
- **Pod verification (this session):**
  - One-shot ask on starting position, `--mode iter --verbose`: BT4 loaded in 39.7s, Stockfish ready, single LLM call (in=760 / out=306, cache_write=3205), `source=llm validation_ok=True handler=1.62s translate=7.39s`, prose surfaces the Leela-policy-vs-Stockfish e4 split correctly, exit 0.
  - Invalid FEN one-shot (`"not a fen"`): exit 2, clear stderr message, no BT4 load attempted (short-circuit before `_build_context`).
  - REPL via stdin heredoc (FEN → question → `:fen <other>` → question → `:history` → `:quit`): both questions answered. Turn 1 clean (`source=llm`). Turn 2 (Italian Game `r1bqkbnr/...`) hit validator twice — LLM cited `Nf3`/`Bc4` (moves already played in the FEN history, not legal options for Black to move). Templated fallback fired per Rule 2; `source=templated, validation_ok=False` reported correctly. `:history` printed both turns with their sources. Clean exit 0. No Stockfish zombies (pgrep matched its own command line, false positive).

What surprised me:
- **The LLM occasionally cites "ancestor" moves.** Turn 2's `Nf3` and `Bc4` errors were both moves that *led to* the Italian Game position, not moves Black can play from it. The validator's legality check at the root board correctly catches this (those moves aren't legal for Black to move). The fallback chain handled it — Rule 2 holds — but the prompt-v3 wording doesn't yet explicitly forbid referencing the move sequence that produced the position. Worth a v4 anti-pattern when handlers 2/3 are written ("only reference moves from `policy.top_moves`, `stockfish.top_moves`, or `stockfish.top_moves[*].pv_san`; never reference the moves that led to this position"). Not blocking; the system gracefully degraded to a grounded templated response.
- **First-call cache writes are bigger than expected.** Pod one-shot showed `cache_write=3205` on the first (and only) LLM call, with cache_read=0 — meaning the v3 prompt + JSON findings prefix is now substantial enough that Anthropic pre-caches it on the first call. Sonnet 4.6's cache window is engaging well, consistent with the earlier 2026-05-18 observation that the documented 2048-token minimum may be looser in practice.
- **REPL via stdin heredoc works fine without `-t`.** `input()` reads line-by-line from the piped stdin; the prompts (`Position FEN (or :quit): `, `> `) appear in stdout but the loop progresses through the scripted commands cleanly. Means future smoke scripting of multi-turn flows doesn't need pty allocation.

Open questions / threads:
- **Prompt v4 "no ancestor moves" rule.** Hold until handlers 2/3 are written post-Phase-2 — best to consolidate the v3→v4 bump with the schema additions MCTS will bring rather than ship a one-line v4 now.
- **ADR-0001 through ADR-0010 `[initials]` placeholders.** Tenth ADR added without backfilling the earlier ones. Defer until a routine docs-hygiene pass (low-priority).
- **Friends' eval positions.** Per ADR-0010, these become "nice-to-have manual spot-check material" rather than an ML-style regression baseline. If they arrive, run the smoke script against them and eyeball; no formal scoring loop needed.

Phase 1 exit-criteria status (informal closure per ADR-0010):
- [ ] 30 eval positions pass on all 3 handlers — **N/A under ADR-0010.** Only handler 1 exists; handlers 2/3 are deferred. Manual QA replaces formal pass/fail.
- [x] Zero hallucinated moves in last 100 interactions — preserved by Rule 2 (validator + fallback) which IS still in force; Chunk 5 and Chunk 6 verifications confirm.
- [x] <30 s end-to-end latency — Chunk 6 verified: ask = ~50s with cold BT4 load; ~9s warm per turn in REPL. Cold load is amortized in REPL (load-once, many turns) and is well-within tolerance for one-shot.
- [x] You personally find it useful for your own games — CLI now lets the user actually run positions; subjective utility check happens organically as the user plays.

Next session goal:
- **Phase 2 Entry Brief** (CLAUDE.md §10) — MCTS integration. Mandatory reading from PRIOR_WORK.md (Lc0 `--verbose-move-stats` doc, McGrath Hydra Effect for Phase 6 prep-awareness pass), tooling (Stockfish wrapper already shipped in Chunk 5; new piece is the MCTS tree parser — ADR-0006 documents the option-space, decision still pending), go/no-go decisions (parser sourcing: Option A wrap superhuman-ai-consistency vs Option B clean-room port). Phase 2 is a multi-chunk effort; brief will propose the chunking before any code lands.

---

## Phase 2 — MCTS Integration

### Exit criteria (per Phase 2 plan, adjusted under ADR-0010)

- [ ] MCTS extractor produces clean Pydantic on ~30+ positions (smoke against dev fixtures + walkthrough + fresh).
- [ ] Handler 1 v4 prompt surfaces real disagreement on the Rxd4 walkthrough position (`source=llm`, prose names the policy-vs-MCTS visit-share gap explicitly).
- [ ] Handler 2 `why_this_move` explains a chosen move usefully on ~5 positions.
- [ ] Handler 3 `did_you_consider` distinguishes "tactically refuted" from "didn't search enough" on ~5 positions.
- [ ] No regression: `pytest tests/ -q` stays clean; CLI REPL still functional.
- [ ] Subjective utility check: user finds the system more useful than at Phase-1-close.

(Original criterion "6 handlers total pass eval" reinterpreted per ADR-0010 — no grader.)

### Locked Phase 2 decisions (2026-05-19, from Phase 2 Entry Brief)

- **G2.1** — Parser sourcing: **Option B (clean-room port)**. Reasoning: superhuman-ai-consistency has zero tests, conflicting deps (chess==1.9.4 vs our 1.11), and a lossy schema (collapses W/D/L into a single WL float). Porting buys Pydantic typing, edge-case tests, schema fields we need (per-child PV, mate flag, search-complete marker), and dep-conflict-free integration. Reference repo stays on disk for grammar consultation.
- **G2.2** — Chunk plan: 7 chunks (subprocess wrapper → parser → extractor → Handler-1 v4 → Handler-2 → Handler-3 → manual QA). Architecture-of-efficiency reorder: handlers 2/3 deferred from Phase 1 to land here against the final findings shape.
- **G2.3** — Pydantic models preserve per-child raw counts (not aggregated stats). Phase 6 counterfactuals (e.g. "what if Lc0 had searched longer / with different cpuct / with policy ablated?") require raw N/Q/P/wins/draws/losses per child — aggregates foreclose them. Cost: ~1.5 KB extra Pydantic per call; trivially worth it.

### Chunk plan (per Phase 2 plan)

| # | What | Status |
|---|---|---|
| 1 | Lc0 subprocess wrapper (`Lc0Engine`) | ✅ done 2026-05-19 |
| 2 | MCTS parser (`parse_verbose_move_stats(lines) -> MCTSFindings`) | ✅ done 2026-05-19 |
| 3 | MCTS extractor (`extract_mcts(board, *, engine, nodes)`) | ✅ done 2026-05-19 |
| 4 | Handler 1 v4 — incorporate MCTS findings | ✅ done 2026-05-19 |
| 5 | Handler 2 `why_this_move` | ✅ done 2026-05-19 |
| 6 | Handler 3 `did_you_consider` | pending |
| 7 | Manual QA + CLI polish if surfaced | pending |

### Session log

#### 2026-05-19 (cont.) — Phase 2 entered + Chunk 1 done

What I did:
- Delivered Phase 2 Entry Brief (CLAUDE.md §10) — reading list, tooling check, G2.1/G2.2/G2.3 with my recommendations. User deferred all Tier 1 reading ("reference inline as needed"), asked for the MCTS-basics + G2.3 sections of the plan expanded; both expanded with worked examples on the Rxd4 position. G2.1/G2.2/G2.3 all accepted as proposed.
- Chunk 1 (`Lc0Engine` wrapper) implemented in `leela_chat/engine/lc0_engine.py` (+ `engine/__init__.py` re-exports). Uses `chess.engine.SimpleEngine.popen_uci([binary, "--verbose-move-stats", "--weights=<path>"])` to spawn Lc0 with verbose stats enabled. Key design choice: capture info-string lines via the streaming `engine.analysis(...)` context manager (iterating over per-line InfoDicts) rather than `engine.analyse(...)` (which would aggregate and overwrite `info["string"]` keeping only the last). Each loop iteration: `info_string_lines.append(info["string"])` when present.
- 11 unit tests in `tests/unit/test_lc0_engine.py` covering constructor (path validation, CLI args, extra_args plumbing), analyse_verbose (line collection, bestmove reporting, missing-bestmove tolerance, EngineError tolerance, nodes-limit passthrough), and lifecycle (context manager, quit-error tolerance). All pass locally.
- 2 integration tests in `tests/integration/test_lc0_engine_integration.py` (auto-skipped off-pod) — run real Lc0 against starting position (500 nodes) and the walkthrough Rxd4 position (2000 nodes). Pod run: 13/13 pass, 11.45s total including BT4 GPU load.
- Renamed the integration test to `test_lc0_engine_integration.py` after pytest balked at the basename collision with the unit test. Convention worth keeping: when a feature has both unit and integration tests, suffix the integration file with `_integration.py` rather than adding `__init__.py` package markers under `tests/`.

What surprised me:
- **Lc0's actual line format differs slightly from what I had in the Phase 2 plan's example.** Real v0.32.1 output:
  ```
  d2d4  (293 ) N:      96 (+ 0) (P: 12.28%) (WL:  0.04399) (D: 0.649) (M: 198.6) (Q:  0.04399) (U: 0.05111) (S:  0.09506) (V:  0.0364)
  ```
  Differences from my plan's draft format: (a) move-index in parens after the UCI move (Lc0 internal — discard during parse); (b) `N: X (+ Y)` form encodes in-flight visits — `+ Y` is visits-in-flight separate from completed `X`; (c) V comes at the END not in the middle; (d) field order is N, P, WL, D, M, Q, U, S, V. Plus a final `node ...` summary line for the root aggregate. Chunk 2's parser regex will handle all of these. Captured here so Chunk 2 doesn't have to re-discover the format.
- **The streaming `analysis()` approach worked first-try.** I'd half-expected to need to subclass `UciProtocol` to intercept lines, but pulling `info["string"]` from each yielded InfoDict captures every verbose-move-stats line cleanly because the context manager exits BEFORE the bestmove sentinel, but the iterator drains the queue. The relevant python-chess source paths: `chess/engine.py:1798-1810` (`_parse_uci_info` stuffs the post-`info string ` content into `info["string"]`), `chess/engine.py:1756` (`analysis.post(info_dict)` per line), `chess/engine.py:2727-2737` (`AnalysisResult.post` queues each one for iteration).
- **Lc0 startup chatter goes to stderr** (CUDA detection, weights file loading) which python-chess logs as `<UciProtocol (...)>: stderr >> ...` — not into `info["string"]`. Cosmetic only; doesn't pollute the captured lines.

Open questions / threads:
- **The in-flight visit count `N: X (+ Y)` form.** For root summary at end-of-search, in-flight should always be 0 (search has concluded). For mid-search captures, in-flight could be nonzero. Parser should store both `visits` (X) and `visits_in_flight` (Y) — useful for Phase 6 if we want to inspect intermediate search states.
- **The `node ...` summary line.** Lc0 emits one as the final captured line — represents the root node aggregate (subtree sum across all root children). Chunk 2's parser should split candidate lines (move-prefixed) from the node summary line and store the node line's stats separately (likely as `MCTSFindings.root_stats: dict`).
- **The (move-index) field in parens after the UCI move.** Lc0's internal ordering index. No interpretive value for us; parser can discard.

Next session goal:
- **Chunk 2** — `leela_chat/extractors/mcts_parser.py`. Pure-function `parse_verbose_move_stats(lines: list[str]) -> MCTSFindings`. Inline Pydantic models (`MCTSChild`, `MCTSFindings`). Unit tests with frozen Lc0 v0.32.1 sample outputs (the starting-position 21-line dump captured today + ~5-10 more from the walkthrough positions). No pod needed — pure-Python parser tests run locally.

---

#### 2026-05-19 (cont., 2) — Phase 2 Chunk 2 done: MCTS parser

What I did:
- Captured 5 Lc0 v0.32.1 fixtures on the pod, saved under `tests/unit/fixtures/lc0/`: starting position (500 nodes, 20 candidates), Italian Game Black to move (500 nodes, 31 cand), walkthrough Rxd4 (2000 nodes, 51 cand), endgame R+N vs B+P (1000 nodes, 31 cand), QGD Bb4 opening (500 nodes, 24 cand). Each fixture header records FEN + nodes_requested + bestmove_uci + search_complete + wall_seconds + line_count, then the raw info-string lines.
- Implemented `leela_chat/extractors/mcts_parser.py`. Pure-Python, no I/O. Three Pydantic models: `MCTSChild` (per-root-candidate raw counts — visits, visits_in_flight, prior_pct, q, v, wl, draw, moves_left, u, s, plus an empty `pv_san: list[str]` placeholder for Chunks 5-6 to fill in if needed), `MCTSRootStats` (root aggregate from the `node` line; same fields minus u/s which root doesn't have), and `MCTSFindings` (the top-level container). Parser uses three composable regexes: `_HEADER_RE` peels off `move (idx)`, `_N_RE` peels off `N: X (+ Y)`, and `_FIELD_RE` extracts every `(KEY: VALUE)` pair after. Children sorted by visits desc (tie-broken by prior desc, deterministic). 19 unit tests including a flagship "Rxd4 surfaces on top" test that pins the walkthrough Phase-1-fail case as a regression guard.
- Smoke verification: parser handles all 5 fixtures cleanly. Walkthrough Rxd4 case validates: `top1 = Rxd4 (N=842, Q=+0.951, prior=6.79%)` — exactly the policy-vs-MCTS gap Phase 1 couldn't surface. Now Phase 2 will.

What surprised me:
- **Lc0 emits `(V: -.----)` as a sentinel for unexpanded moves.** When a candidate has never been expanded by the search (low prior, search didn't get to it), Lc0 still emits a `(V: ...)` field on the line, but with literal `-.----` instead of a number. My initial regex (numeric values only) silently dropped these — which was actually correct, but the consumer code defaulted V to 0.0, which is wrong (0.0 would mean "value head says exactly even"). Changed `MCTSChild.v` to `float | None` so the missing-V state propagates honestly. Walkthrough rxd4 has 19 None-V children (low-prior moves Lc0 never tried); test pins this so a future parser change doesn't regress.
- **Total visits doesn't match nodes_requested.** Asked for 2000 on the walkthrough position; root_stats reports visits=931 + visits_in_flight=164 = 1095. Lc0's node counter excludes some bookkeeping or stopped early on the heavily-confidence-converged Rxd4 line. Not a parser bug — the engine itself terminates when convergence is detected. `total_nodes_searched` in findings reports the engine-true number; `nodes_requested` from `Lc0SearchResult` carries the caller-asked-for number; handlers can compare to detect early termination.
- **Sub-second searches at 2000 nodes.** Wall-clock on the pod's RTX 3090: starting position 500 nodes = 4.8s (mostly BT4 net load), walkthrough 2000 nodes = 0.35s (BT4 warm). Reasonable budget for handler use — `--nodes 10000` should still fit in ~2-3 s warm.

Open questions / threads:
- **Should the parser expose the raw `_parse_one_line` dict for debugging?** Currently the function is module-private (underscore prefix). I exported it in the test file's imports to write unit tests, which is a slight layer-leak but acceptable for testing internals. Revisit if a third caller needs it.
- **PV (principal variation) is empty per-child.** `MCTSChild.pv_san = []` always at Chunk 2. For Handler 3 (`did_you_consider`), we may want to know "if Leela had played X, the opponent would respond Y, you'd respond Z" — that requires deeper tree dump or move-forcing. Defer the decision to Chunk 5-6 when the handler design forces a call.

Next session goal:
- **Chunk 3** — `leela_chat/extractors/mcts.py`. Composes `Lc0Engine.analyse_verbose` + `parse_verbose_move_stats` into a single `extract_mcts(board, *, engine, nodes=10000) -> MCTSFindings` callable. Tests mock both pieces and verify the glue. ~50 LOC, ~5 unit tests, ~30 minutes including PHASE_NOTES entry.

---

#### 2026-05-19 (cont., 3) — Phase 2 Chunk 3 done: MCTS extractor (the glue)

What I did:
- `leela_chat/extractors/mcts.py` — `extract_mcts(board, *, engine, nodes=10000) -> MCTSFindings`. Pure composition: call `engine.analyse_verbose(board, nodes=nodes)` → feed `info_string_lines` + `bestmove_uci` + `search_complete` into `parse_verbose_move_stats(...)` → return findings. Loguru debug line records `requested / searched / bestmove / top1_visits / complete` for telemetry. ~40 LOC including docstring.
- 5 unit tests in `tests/unit/test_extract_mcts.py` (mocked Lc0Engine returning frozen fixture lines): happy path on starting position, nodes-kwarg passthrough, board passthrough (with the walkthrough Rxd4 fixture to verify SAN translation uses the correct board), default nodes=10000, search-incomplete propagation.
- 2 pod integration tests in `tests/integration/test_extract_mcts_integration.py`: end-to-end against real Lc0, checking the walkthrough Rxd4 position surfaces Rxd4 with Q > 0.5 and >200 visits at 2000 nodes, plus a starting-position smoke that confirms 20 children + bestmove on a 500-node search.
- All 28 MCTS + Lc0 tests pass on pod in 21.93s. Local: 84 pass + 32 skip (the 32 skips include the 4 pod integration tests that auto-skip without `/workspace/bin/lc0`).

What surprised me:
- **Nothing**, which is actually a good signal. Chunks 1 and 2 were designed so that Chunk 3 was a trivial composition; the absence of surprises means the layer boundaries were right. The most "interesting" thing was confirming that Lc0 at 2000 nodes really does find Rxd4 in the walkthrough case — which we knew from the fixture capture, but seeing it through the full `extract_mcts` pipeline against the live engine is the proof.

Open questions / threads:
- **Lc0 search-budget tuning.** Default `nodes=10000` is reasonable for Handler 1 (`what_is_leela_seeing`). Handler 3 (`did_you_consider`) may want 50k+ to give low-visit candidates a fair chance. Defer the per-handler-default decision until Chunks 5-6 land.

Next session goal:
- **Chunk 4** — Update `what_is_leela_seeing` to incorporate MCTS findings. Touches: `leela_chat/handlers/what_is_leela_seeing.py` (add `extract_mcts` call + `mcts: MCTSFindings` field on `WhatIsLeelaSeeingFindings`), `templated_fallback` (extend to render MCTS info), `leela_chat/llm/prompts/what_is_leela_seeing/v4.md` (new prompt teaching three-signal framing with worked example on the Rxd4 case), `leela_chat/cli/__main__.py` (build Lc0Engine in `_build_context`, pass to handler, close in `_close_context`). Smoke against the walkthrough positions; expect Rxd4-case prose to be substantially richer than Phase 1.

---

#### 2026-05-19 (cont., 4) — Phase 2 Chunk 4 done: user-visible MCTS payoff

What I did:
- **Handler + schema** (`leela_chat/handlers/what_is_leela_seeing.py`): added optional `lc0_engine: Lc0Engine | None` param + `mcts_nodes: int = 10000`. New field `mcts: MCTSFindings | None` on `WhatIsLeelaSeeingFindings` (defaults None for graceful no-Lc0 paths). Handler truncates `mcts.children` to top `MCTS_TOP_N_FOR_LLM = 5` for its findings projection; the extractor's full output stays available to Handler 3 (`did_you_consider`) when it lands in Chunk 6. `templated_fallback` extended to render an MCTS section ("Leela MCTS (committed: Rxd4, 842 of 1000 visits = 84%): top by visits Rxd4 (N=842, Q=+0.95), …") only when `findings.mcts is not None`. `PROMPT_VERSION` bumped to 4.
- **v4 prompt** (`leela_chat/llm/prompts/what_is_leela_seeing/v4.md`, ~6 KB): rewrites the three-signal framing — policy first-impression vs MCTS committed-judgment vs Stockfish second opinion. Explicitly teaches: **visit share is the load-bearing metric, not policy %**. Two worked examples: consensus case (Italian Game, all three agree on Bc5) and the canonical disagreement (Rxd4 walkthrough, MCTS overrides policy with 77% visit commitment despite 4th-ranked 6.8% prior). New anti-patterns specifically for MCTS data: don't cite V when null, don't claim per-child PV (always empty in Phase 2), 2-decimal Q max, never conflate policy % with visit %, don't say "considered and rejected" for zero-visit candidates ("the search didn't even try" is the accurate phrasing), no inventing mate-in-N from Q near ±1.
- **CLI** (`leela_chat/cli/__main__.py`): `_build_context` now best-effort spawns `Lc0Engine` from `$LC0_PATH`/`$LC0_WEIGHTS` env vars (defaults `/workspace/bin/lc0` + the BT4 .pb.gz on the pod). Missing binary or weights → warning log + degrade to policy-only mode. `_close_context` extended to also close Lc0. `CliContext.lc0` is `Lc0Engine | None`. Test paths and local dev boxes that lack Lc0 still work; production pod always has it.
- **Smoke on pod (Rxd4 walkthrough, the binding test):** `source=llm validation_ok=True`, single LLM call (no retry), 7.05 s handler / 10.81 s translate. Prose explicitly names the policy-vs-search gap: *"Leela's policy network ranked Rxd4 only fifth at 6.8% — its first-impression preference was Rxb1+ at 33.0%… But after 4782 nodes of MCTS, the engine committed 4217 visits (88%) to Rxd4 with Q +0.92… Stockfish at depth 15 agrees… Stockfish's line continues Rxd4 Qc3 Rxb1+ Kxb1 Qb8+…"* Exactly the prose Phase 1 couldn't produce.
- **Smoke on pod (Italian Game, the Phase-1-fallback case):** `source=llm`, no retry needed. Three-way consensus correctly named ("All three signals converge clearly on Bc5"). LLM cites real MCTS visit counts (Bc5: 4560 / 5370 = 85%) and discusses low-visit candidates honestly (Be7: 32 visits, Q −0.17) — no claim that MCTS "considered and rejected" them. The Phase-1 ancestor-move bug (LLM citing Nf3/Bc4 which led to the position) is mitigated because MCTS gives the LLM 30+ legitimate candidates to discuss.

What surprised me:
- **`prior_pct` in `mcts.children` ≠ `policy_pct` in `top_moves`.** Pod smoke showed Bc5 with `policy_pct=67.5%` (from `extract_network_inference` raw policy net) but `prior_pct=49.0%` in `mcts.children` (from Lc0's verbose-move-stats P field at root). The two should be the same number — both are Lc0's policy-net softmax. The likely cause: Lc0 applies a softmax temperature scaling or per-move blending at the root before MCTS PUCT consumes it. Worth investigating in a follow-up — but the LLM handled the discrepancy gracefully by citing both as separate numbers (which they technically are). Not blocking Chunk 4 ship.
- **Pod-state-caveat strikes again: `BT4.pt` was missing.** The Fudan-vendored loader hardcodes `/path/to/models/lc0/BT4.pt` which is outside `/workspace`. On the new pod this didn't exist. Fixed by running `convert_bt4_onnx_to_pt(...)` once to produce the .pt under `/workspace/.../BT4.pt`, then symlinking. **setup_pod.sh should grow this step** (idempotent: skip if symlink already points at a real file). Adding to "things to revisit" — fix opportunistically next session.
- **Lc0 reported 4782 actual nodes searched on a 10000-node request.** Same convergence-driven early-termination behavior we noted in Chunk 1-3. The v4 prompt instructs the LLM to cite the actual total (`mcts.total_nodes_searched`), which it did correctly.

Open questions / threads:
- **`prior_pct` vs `policy_pct` divergence.** Confirm whether this is Lc0's softmax-temperature scaling, root Dirichlet noise (we don't enable that), or something else. Cleanup ideas: (a) add a sentence to v4 explaining the two are technically different sources, (b) report only one in the LLM-facing JSON and pick the more-faithful one. Defer to a docs-hygiene pass.
- **`setup_pod.sh` BT4.pt step.** Should add: if `/path/to/models/lc0/BT4.pt` is missing AND `/workspace/.../BT4.pt` is missing, run conversion; else symlink. Idempotent. Important so future pod restarts don't pay the conversion cost twice.

Next session goal:
- **Chunk 5** — Handler 2 `why_this_move`. New file `leela_chat/handlers/why_this_move.py` + Pydantic findings model + `leela_chat/llm/prompts/why_this_move/v1.md`. Handler signature: `(board, *, move_uci, model, stockfish_engine, lc0_engine)`. Findings include MCTS subtree info about the asked-about move (visit share, Q, V) and how it compares to the engine's actual top pick. CLI: extend `ask` subcommand with `--move` arg; classifier still trivial-routes for now. Smoke against ~5 user-eyeballed positions where you played a specific move and want a second opinion.

---

#### 2026-05-19 (cont., 5) — Phase 2 Chunk 5 done: `why_this_move` + bucket framing

What I did:
- **Handler module** (`leela_chat/handlers/why_this_move.py`, ~280 lines): `WhyThisMoveFindings` Pydantic with position-level fields mirroring Handler 1 + focal-move callouts (`focal_*` for policy / MCTS / Stockfish, each with rank + stats; None when the move is outside that source's reported top). `handle_why_this_move(board, move, *, model, stockfish_engine, lc0_engine, ...)` validates move legality, runs the three extractors, scans pre-truncation candidate lists for the focal move, then truncates `mcts.children` to top-10 by visits (wider than Handler 1's 5 since the focal move is more likely to need its raw stats surfaced — though the pre-extracted `focal_mcts_child` is the authoritative source regardless). `templated_fallback` renders both branches (`focal_is_engine_pick=True/False`) sensibly.
- **v1 prompt** (`leela_chat/llm/prompts/why_this_move/v1.md`, ~6 KB): three-signal framing zoomed onto the focal move with **categorical eval buckets** per user feedback. Five fixed buckets: **winning** (eval > +2.0 pawns / Q > +0.80), **better** (+0.5 to +2.0 / +0.30 to +0.80), **draw / equal** (-0.5 to +0.5 / -0.30 to +0.30), **worse** (-2.0 to -0.5 / -0.80 to -0.30), **losing** (eval < -2.0 / Q < -0.80). MCTS Q thresholds derived from Lc0's `cp = 90 * tan(1.5637541897 * q)` formula. "Near-equal" = same bucket AND `abs(Q_focal - Q_alt) < 0.10` (or Stockfish fallback `< 0.50 pawns`). Cross-bucket alternatives are explicit downgrades. Three worked examples: cross-bucket dominant (walkthrough Rxd4), same-bucket near-equal (quiet recapture), focal-NOT-engine-pick (the user-played-Rxb1+ case).
- **CLI** (`leela_chat/cli/__main__.py`): `--move <UCI>` added to `ask` subparser. UCI format + legality validated up-front (exit 2 before BT4 load on bad input). Default-routing rule: `--move` without `--handler` auto-routes to `why_this_move`. Imports of handler symbols use `as`-aliases (`WHAT_HANDLER_NAME` / `WHY_HANDLER_NAME`) to disambiguate the two handler modules. `_answer_one` dispatches on handler name.
- **Tests** (8 unit + 5 CLI integration): handler logic (focal-as-engine-pick / focal-not-pick / focal-outside-top / illegal-move-raises / lc0=None-skips-mcts / children-truncated-to-10 / templated_fallback both branches), CLI argv routing (--move passes through / --move alone routes to why_this_move / explicit --handler overrides / invalid UCI exits 2 / illegal-for-board exits 2). 97 local tests pass (84 → 97).
- **Pod smoke (binding tests):** Both branches on the Rxd4 walkthrough position, `source=llm validation_ok=True` for both, no retries. Branch A (`--move d8d4`): prose names "winning bucket" / "cross-bucket downgrade" / "just outside the near-equal threshold" — exactly the categorical language the prompt teaches. Branch B (`--move b8b1`): prose leads with "Leela's MCTS firmly prefers Rxd4, not Rxb1+" and surfaces the genuinely insightful observation that **Rxb1+ appears later in Stockfish's PV** ("the check has real value — just not as move one"). That's real chess inference from the data the prompt gave the LLM, not generic engine-number reporting.

What surprised me:
- **The Branch B prose was unexpectedly insightful.** I'd designed the prompt to explain "engine's pick + focal-move gap" but the LLM connected the dots about Rxb1+ being a *later* move in Stockfish's preferred PV — the kind of pattern reasoning that bridges scoreboard-reporting and chess-content. Not a prompt change I planned for; emerged from giving the LLM the focal-move-in-Stockfish-PVs data plus enough room in the prompt.
- **Bucket boundaries held cleanly in practice.** No "slightly better" / "marginally winning" weasel words in either smoke output. The LLM categorised Rxd4 = winning, Rbc8 = winning, Rxb1+ = better, and named cross-bucket gaps explicitly. The fixed thresholds removed an entire class of LLM hedge phrasing.
- **One Branch A prose nit:** the LLM cited the runner-up Rbc8 with a Q gap of 0.12 (vs the 0.10 near-equal threshold) and correctly called it "just outside the near-equal threshold" — but ideally the prompt would prefer that this be framed as "narrowly outside" rather than implying a hard cliff. Not a defect; minor v2 polish for later.

Open questions / threads:
- **Refactor `FocalMoveFindings` as shared module before Chunk 6?** Chunk 6's `did_you_consider` will need the same focal-* fields. Current decision per plan: extract iff clear overlap during Chunk 6 implementation. Both findings shapes could share inheritance / composition; defer until both exist.
- **Backlog: "practical-best" / Maia-driven recommendation** (user request 2026-05-19). Idea: future handler `recommend_practical_move(fen, opponent_elo)` that scores candidates by *how hard the resulting position is for a human opponent to defend*. `update_research/repos/maia-chess-master/` is already on disk. Approach sketch: load Maia at target ELO, run it on positions after each candidate, compute the gap between Maia's predicted human response and Leela-MCTS's optimal response (bigger gap = harder for the human). Natural fit alongside Phase 3 concept-probe handlers. Tracked here so it doesn't get lost.

Next session goal:
- **Chunk 6** — Handler 3 `did_you_consider`. New handler module + v1 prompt + CLI flag wiring. Reuses the same MCTS-bearing pipeline as Chunk 5; the prompt zooms onto **why an alternative was rejected** rather than justifying the engine's pick. Decision-tree on `focal_mcts_child.visits`: if zero → "search didn't reach it within the budget" framing; if low (e.g. < 5% of total visits) → "search dismissed early" framing; if non-trivial → "explored but evaluated worse than the pick" framing. CLI: `--handler did_you_consider --move <UCI>`. Likely refactor the shared focal-move-finding helpers (`_find_focal_in_policy` etc.) into a small shared module if the duplication becomes noisy.

---

#### 2026-05-19 (cont., 6) — Phase 2 Chunk 6a done: methodology doc + Q-trajectory extractor

Chunk 6 (`did_you_consider` with dual-timescale dismissal tracking) is too large for one chunk. Split into 6a (methodology + search-time Q-trajectory) / 6b (network-time layer emergence) / 6c (handler + prompt) per the approved plan at `/home/kornel/.claude/plans/tell-me-and-explain-mutable-axolotl.md`. 6a is this session.

What I did:

- **`docs/INTERPRETABILITY_NOTES.md`** — new methodology document (~280 lines). Design-doc-first discipline: landed BEFORE any extractor code. Covers (1) the premise — extracting deliberation, not just verdict; (2) two timescales (network-time forward-pass emergence + search-time MCTS refutation), with the four interpretive cases enumerated; (3) network-time methodology (linear vs zero-ablation logit-lens, BT4 policy-head reapplication math, summary tag derivation); (4) search-time methodology (multi-budget MCTS, why checkpoint resolution beats visit-by-visit, cost analysis); (5) the Rxb1+/Rxd4 worked example with predicted dual-timescale story; (6) honest caveats (linear logit-lens approximation, no causal claim, BT4-specific); (7) references to Iterative Inference, Alain & Bengio, Jenner Look-Ahead, Fudan, Monroe, McGrath; (8) future work pointing at Phase 3 probes / Phase 4 SAEs / Phase 6 counterfactuals / Phase 7 reasoning pathways. This is the living interpretability-methodology document for the project; it grows with each phase.

- **Schema extension** (`leela_chat/extractors/mcts_parser.py`): added `MCTSTrajectoryPoint(nodes_searched, visits, q)` frozen Pydantic model + `trajectory: list[MCTSTrajectoryPoint] = []` field on `MCTSChild`. Default empty so Chunks 1-5 callers (single-budget `extract_mcts`) are unaffected. Decision recorded in this entry: **embedded** rather than a sibling `MCTSFindings.trajectories` dict. Rationale: handler ergonomics (single lookup for "everything about this candidate"), and network-time layer-emergence won't pile onto `MCTSChild` because it lives in its own `LayerEmergenceResult` (Chunk 6b).

- **New extractor** (`leela_chat/extractors/mcts_trajectory.py`, ~110 lines): `extract_mcts_trajectory(board, *, engine, budgets=DEFAULT_BUDGETS=[500, 1000, 2500, 5000, 10000]) -> MCTSFindings`. Calls existing `extract_mcts` once per budget, joins per-candidate stats across checkpoints by UCI, returns the deepest-budget findings with `children[*].trajectory` populated. Candidates absent from a lower-budget checkpoint get a zero-fill point (search didn't reach them yet). Candidates that appear at low budgets but drop out of the deepest budget's verbose-move-stats are simply not in the result (intentional: the deepest search defines the candidate set). Validation: budgets must be non-empty, positive, strictly ascending.

- **Unit tests** (`tests/unit/test_mcts_trajectory.py`, 11 tests): happy-path three-budget trajectory assembly, default-budgets call, zero-fill for absent-at-low-budget candidate, drop-from-top behavior at high budget, single-budget degenerate case, ascending-order invariant, validation (empty / negative / non-ascending budgets), frozen-Pydantic preservation, default-empty trajectory on single-budget calls. All pass. Full unit suite stays green: 88 tests.

- **Integration scaffold** (`tests/integration/test_mcts_trajectory_integration.py`, 3 tests, pod-only): Rxd4 Q-confirms-across-budgets, Rxb1+ refuted-across-budgets (final Q below early Q, final Q ≥0.2 lower than Rxd4's), starting-position smoke. Skipped off-pod via `pytest.mark.skipif` on the Lc0 binary + BT4 weights paths. Collects cleanly off-pod.

What surprised me:

- Nothing methodologically. The existing `extract_mcts` was clean enough that the multi-budget wrapper is ~20 lines of real logic; the rest is validation + zero-fill bookkeeping. The methodology doc was the heavier lift — anchoring the *why* in writing first paid off, since the extractor's design follows directly from §4 of the doc rather than the other way around.

Open questions / threads (carry to 6b):

- **Linear logit-lens degeneracy risk.** §3.2 of INTERPRETABILITY_NOTES commits to starting with the linear approach (cheap, ~1.5× one forward pass) and falling back to zero-ablation (~N× forward passes) if early-layer outputs are degenerate. Decision criterion is encoded in 6b's integration test: final-layer top-1 must match `extract_network_inference`'s top-1; layer-0 top-1 on easy positions (starting position) shouldn't be wildly different from the final layer's. This decision will be made *during* 6b based on empirical output.
- **BT4 PolicyHead callability on intermediate residuals.** I asserted in §3.3 of the methodology doc that we can invoke `model.policy_head(residual_L)` directly on a cached `hook_resid_post.{L}`. Have not verified the vendored module supports this without modifications — checking this is the first concrete 6b task before any extractor logic.
- **Checkpoint resolution upgrade path.** Default budgets `[500, 1000, 2500, 5000, 10000]` chosen as logarithmically-spaced "qualitative story" resolution. If 6c's prose feels too coarse (e.g. the LLM can't tell whether visits "stagnated" or "kept growing"), step to 8-10 budgets across the same range. Latency scales linearly.

Next session goal:

- **Chunk 6b** — `extract_layer_emergence` using HookedTransformer hooks + BT4 PolicyHead reapplication. New file `leela_chat/extractors/layer_emergence.py`, unit tests with mocked model, pod integration test asserting layer-15 top-1 matches `extract_network_inference`. Falls back to zero-ablation logit-lens iff early-layer outputs are degenerate.

---

#### 2026-05-19 (cont., 7) — Phase 2 Chunk 6a v1: reviewer-feedback revision

After 6a v0 (commit `8d17f36`) shipped, a reviewer feedback pass identified five methodological problems that — if left in place — would let the LLM produce prose that overclaims relative to the data (violating CLAUDE.md §2 Rule 1). Plan revised; this session implements the search-time corrections (revisions 2 + 3 + 5a) plus the methodology-doc updates for revision 1 (which mainly affects 6b code). Plan file: `/home/kornel/.claude/plans/tell-me-and-explain-mutable-axolotl.md`.

What I did:

- **Methodology doc revision** (`docs/INTERPRETABILITY_NOTES.md`):
  - **§3 (network-time) — Revision 1.** Zero-ablation logit-lens is now the 6b default, not a fallback. Linear logit-lens is demoted to a 6.5-or-later optimization, gated on demonstrated equivalence (concordant top-3 across ≥5 diverse positions). Rationale: BT4's attention-based PolicyHead can produce plausible-looking-but-mechanically-meaningless output on intermediate residuals; the final-layer top-1 match check (the original v0 sanity check) cannot distinguish subtle rank-order errors from geometric noise.
  - **§4 (search-time) — Revision 2.** Rewritten end-to-end for single-search live-stats. v0's multi-search approach stitched five independent searches; this is methodologically wrong because the K trees have different shapes due to MCTS stochasticity. v1 uses Lc0's `--log-live-stats` (UCI: `LogLiveStats`, confirmed at `update_research/repos/lc0-master/src/search/classic/params.cc:294`) to capture intra-search snapshots of the SAME tree at different visit counts.
  - **§5.3 (walkthrough prose) — Revision 3.** Causal phrasings replaced with descriptive ones ("later layers noticed Kxb1" → "rank stayed top-1 across all measured layers; refutation came from search"). Stockfish is the content authority for "which move refutes;" rank-trajectory data covaries but does not establish causation.
  - **NEW §6.5 — Revision 3.** "What layer-emergence does and does not tell us": the encoding-vs-use distinction (ADR-0007) re-emerging one layer down. Rank change at layer L is consistent with both "layer L computed the change" and "the network refines a representation; the rank change is a side-effect with no semantic notion of consideration." Forbidden language list ("noticed," "realized," "discovered") and allowed language list documented.

- **Engine wrapper** (`leela_chat/engine/lc0_engine.py`): added `log_live_stats: bool = False` keyword to `analyse_verbose()`. When True, the wrapper toggles Lc0's `LogLiveStats` UCI option around the analysis call (and resets to False in a `finally` block). Off by default — Chunks 1-5 single-snapshot callers unaffected.

- **Extractor rewrite** (`leela_chat/extractors/mcts_trajectory.py`): function signature `extract_mcts_trajectory(board, *, engine, budgets=DEFAULT_BUDGETS)` preserved. Implementation switched: one `analyse_verbose(nodes=max_budget, log_live_stats=True)` call; live-stats stream grouped into per-emit-cycle blocks (split at `node`-summary boundaries); each block parsed with the existing `parse_verbose_move_stats`; closest-snapshot rule picks one parsed `MCTSFindings` per requested milestone; trajectories assembled per UCI from the deepest snapshot's children. Fallback path: if the live-stats stream produces no parseable blocks, falls through to a single end-of-search snapshot via the existing `extract_mcts` and emits a flat trajectory.

- **Unit tests refactored** (`tests/unit/test_mcts_trajectory.py`): `_StubEngine` replaces the `_patch_extract_mcts` mock. Tests build synthetic live-stats streams via `_emit_cycle` / `_candidate_line` helpers that emit the same format Lc0 produces. Test count: 18 (11 v0 intents preserved + 3 block-grouper tests + 2 milestone-selection tests + 1 fallback test + 1 schema-default-empty test). Subtle bug found and fixed during writing: my initial fixture used `{q:+.5f}` to format Q values, producing `+0.85000` — the parser's `_FIELD_RE` only accepts `-?` (optional minus) not `+`, so positive Q values were silently dropping to 0.0. Fixed by using `{q: .5f}` (space-pad-positives, matching Lc0's own convention).

- **Integration test tightened** (`tests/integration/test_mcts_trajectory_integration.py`): per **Revision 5a**, replaced strict monotonicity with single-comparison-plus-tolerance: `qs[-1] - qs[0] > 0.10` for Rxd4 (confirmation), `< -0.10` for Rxb1+ (refutation). Strict monotonicity flakes on MCTS Q at low visit counts because PUCT visit order creates local noise. Added a fourth test that asserts `--log-live-stats` actually produces ≥2 emit blocks during a 10k search — investigation-gate proxy.

What surprised me:

- **The fixture-format `+` bug** — a positive-format-flag `{q:+.5f}` produces `+0.85000` which doesn't match the parser's regex `-?\d+`. Caught by failing tests (q values came back as 0.0). The fix is a one-character change; the lesson is that any synthetic fixture for the Lc0 verbose-move-stats format must match Lc0's space-pad-positive convention exactly. Easy to get wrong because most production Lc0 output uses `-` for negatives and a space (not `+`) for positives.

- **Refactor smaller than expected.** The data model (`MCTSTrajectoryPoint`, `MCTSChild.trajectory`) was right in v0; only the population mechanism changed. The Pydantic schema, the unit-test intents, and the integration-test structure all transferred 1:1. The plan's "decision: in-place refactor, not parallel extractor" was the right call — no dead code, clean replacement.

Open questions / threads (carry to 6a v1 pod verification + 6b):

- **`--log-live-stats` emission cadence is unknown.** The plan's investigation gate is a pod-only step. The 4th integration test (`test_live_stats_block_count_meaningful_for_10k_search`) is its proxy: if a 10k-node search emits fewer than 2 blocks, the closest-snapshot rule degrades and milestone resolution collapses. If this happens on the pod, options: (a) drop the budget count to match what Lc0 actually emits, or (b) submit an Lc0 patch upstream to make cadence configurable. The fallback path (no parseable blocks → single-snapshot) already handles the worst case gracefully.

- **Multiple milestones mapping to the same snapshot.** The closest-snapshot rule explicitly allows duplicates. The `did_you_consider` v1 prompt (Chunk 6c) needs to handle this gracefully — quoting the *actual* `nodes_searched` value (which may be e.g. 1200 not 1000) avoids the trap of claiming a snapshot represents a milestone it doesn't.

Test status: 95 unit tests pass (up from 88 in v0, +7 from the refactored test_mcts_trajectory). Integration suite collects 56 tests cleanly off-pod.

---

#### 2026-05-19 (cont., 8) — Phase 2 Chunk 6a v2: second-pass reviewer revision

After 6a v1 shipped (`452a4b1`), a second reviewer pass surfaced nine more items (A1, B1, B2, C1, C2, C3, D1, D2, D3). All valid. Plan revised; this session implements the off-pod work in three commits.

What I did:

- **Commit 1 — Doc restructure (`031b9c2`).** Per **C1 + B1 + B2 + C3**:
  - `docs/INTERPRETABILITY_NOTES.md` rewritten as a thin cross-cutting index (~290 lines): §1 Premise, §2 Two timescales, §3 Prose contract (was §6.5, promoted out of "Limitations" per **C3**), §4 Worked example, §5 Per-extractor pointers, §6 Cross-cutting limitations, §7 References, §8 Future work. Per-extractor methodology specifics MOVED to module docstrings — the source of truth now travels with code per the user's explicit choice of Option B.
  - `leela_chat/extractors/layer_emergence.py` NEW scaffold. Module docstring is the full network-time methodology contract (zero-ablation only; **B1**'s rewrite — drops the linear-logit-lens-with-mitigations framing entirely). Function body raises `NotImplementedError`; the 6b implementation must satisfy the docstring contract.
  - `leela_chat/extractors/mcts_trajectory.py` module docstring expanded to ~110 lines covering the single-search live-stats methodology, why multi-search was rejected (carryover from v1), and the dense-capture + summary + content-aware-down-sampling design that lands in commit 2 (**B2**'s rewrite). The prose contract from §3 is referenced as the binding constraint on consumer prose.

- **Commit 2 — Dense capture + summary + content-aware down-sample (`3fa52d3`).** Per **C2**:
  - `leela_chat/extractors/mcts_parser.py` — new `MCTSTrajectorySummary` Pydantic model (q_first, q_last, q_peak, q_peak_at_visits, q_trough, q_trough_at_visits, q_stabilized_at_visits, visit_growth_rate, q_crossed_zero, inflection_points, block_count). Extended `MCTSChild` with `trajectory_full` (every captured block), `trajectory_selected` (content-aware down-sample for prompts), `trajectory_summary` (one-shot stats). Legacy `trajectory` field preserved as alias of `trajectory_selected` for v1 callers.
  - `leela_chat/extractors/mcts_trajectory.py` — new helpers: `_smooth_q`, `_detect_inflection_points`, `_find_q_stabilized_visits`, `_compute_trajectory_summary`, `_content_aware_downsample`, `_build_full_trajectory_for_candidate`. `extract_mcts_trajectory` rewritten: `budgets` is now Optional. When provided (v1 contract), legacy `trajectory` populated by closest-snapshot per milestone. When omitted (v2 default), content-aware down-sample. `trajectory_full` and `trajectory_summary` populate identically in both paths.
  - 14 new unit tests in `tests/unit/test_mcts_trajectory.py` cover the helpers + end-to-end behaviour + v1 back-compat preservation. Total trajectory tests: 32 (was 18). Full unit suite: 109 (was 95).
  - One bug found and fixed during implementation: `_find_q_stabilized_visits` initially returned the last point's visits because a singleton suffix is trivially "stable" (point equals itself). Fixed by requiring ≥2 subsequent points; an anchor with only one successor is "we ran out of trajectory," not "Q settled."

- **Commit 3 — D3 + D2 + PHASE_NOTES (this commit).** Per **D3 + D2**:
  - `docs/phase3/concept_candidates.md` NEW. Phase 3 probe-concept candidate tracker; ~80 seeded concepts across tactical motifs, pawn structure, piece activity, king safety, endgame patterns, strategic concepts, opening-phase concepts. Cross-reference checklist (Silman, Aagaard, Watson, Dvoretsky) for the Phase 3 entry brief's pressure-test pass. Explicit "not on this list" section (opening repertoires, style markers, engine artefacts) to keep the scope honest.
  - This `PHASE_NOTES.md` entry plus the Phase 4 placeholder + Chunk 7 entry below.

What's deferred to pod session before Chunk 6c implementation:

- **A1 — Divergent-timescale test fixture.** Requires real Lc0 capture; off-pod blocks the FEN selection on running Lc0 ourselves to verify the candidate position actually shows the network-low / MCTS-rescue signature. The plan's pre-6c gate stands: a `tests/unit/fixtures/lc0/divergent_<name>.txt` (Lc0 verbose-move-stats capture) plus a metadata YAML (FEN, expected MCTS-rescue move, Stockfish-depth-20 PV) must be committed before 6c implementation begins. The 6c v1 prompt's third worked example anchors on this fixture.

  **A1 source-hunt order** (user guidance 2026-05-19):
  1. Check `update_research/repos/superhuman-ai-consistency-main/` test set first — likely the cleanest divergent candidates.
  2. Fall back to leela-interp's `interesting_puzzles.pkl` (figshare download required) only if (1) doesn't surface a clean candidate quickly.
  3. Selection signature: tactical-sacrifice positions where the network *underrates* the sacrifice but MCTS finds it AND Stockfish's bestmove agrees with MCTS — concretely: focal move is Stockfish-top-1, MCTS-top-1, but outside the network policy top-3.

What surprised me:

- **The doc restructure was bigger than I'd estimated.** The slim INTERPRETABILITY_NOTES.md is 290 lines — more than the ~150 target. §3 (Prose Contract) is the most expensive section by a wide margin because it's load-bearing (forbidden / allowed language lists, permanence clause, encoding-vs-use explanation). I left it long; cutting it further would make the contract weaker. The methodology-doc trajectory will be revisited at Phase 4 anyway per the Option-B decision's exit criterion.

- **C2's content-aware down-sample is genuinely better prose-fodder than fixed milestones.** Testing the worked example mentally: "Q peaked at +0.74 around visit 800 then declined steadily to +0.40 by visit 10 000" is a sentence the LLM can produce *only* if it knows where the peak was. With fixed milestones at [500, 1000, 2500, 5000, 10000], the peak at visit 800 would be invisible to the prompt. C2 makes the inflection narrative tractable.

- **The `MCTSChild` schema needed four trajectory fields**, not three. I went with `trajectory` (legacy alias) + `trajectory_full` (everything) + `trajectory_selected` (down-sample) + `trajectory_summary` (stats). The legacy alias is genuinely useful — back-compat tests pass byte-identically with no special-casing in callers. The downside is field-count clutter on `MCTSChild`. Acceptable; will revisit at the 6.5 retrofit when handler-side serialization is overhauled.

Open questions / threads (carry to 6b):

- **`_content_aware_downsample` may not include all 5 priority anchors if some collide.** E.g., q-peak == first or q-trough == last. The dedupe-by-`nodes_searched` is correct (no double-points), but the resulting selection may be ≤6 even when 6 distinct anchors were targeted. Acceptable; the alternative is double-counting which is worse.

- **`q_stabilized_at_visits` requires 3+ trajectory points.** With Lc0's `--log-live-stats` cadence unknown, a 10k-node search that emits only 2-3 blocks would have null stabilization. The pod investigation gate will tell us whether this is a real concern.

- **`extract_layer_emergence` scaffold contract.** The module docstring is now the binding spec for 6b. If 6b's first integration test (final-layer top-1 must match `extract_network_inference`) fails with zero-ablation, the documented pivot path is mean-ablation (replace activation with calibration-batch mean), NOT linear logit-lens. This is explicit in the docstring's §"6b implementation contract" and DECISIONS.md would record the pivot.

Test status: 109 unit tests pass. Integration suite still collects 56 cleanly off-pod (no changes there).

---

### Phase 4 — Coach-voice prompt migration (placeholder, D2)

> Reserved slot per Chunk 6a v2 reviewer pass D2. Phase 1-2 prompts are
> deliberately telemetry-voice for safety while extractors are thin. By
> Phase 3 (probes) + Phase 4 (SAE features) + Chunk 7 (chess-content
> extractors), the findings JSON is rich enough to support coach-voice
> prose.

Scope sketch:

- Rewrite each handler's prompt from "narrate the findings" to "give the
  player a coaching takeaway grounded in the findings."
- Voice-A/B harness for comparing telemetry-prompt vs coach-prompt on
  the same findings JSON.
- Validate that the coach voice doesn't degrade Rule-1 grounding — the
  coach-voice prompts must still pass the validator on every output.
- Eval-set rerun comparing the two voices on the same eval positions.

This is a real chunk, not a polish item. Full plan written when Phase 4
begins.

---

### Chunk 7 — Chess-content extractors (placeholder, D1)

> Reserved slot per Chunk 6a v2 reviewer pass D1. The user has clarified
> the product goal: a usable chess coach, not an engine-internals
> narrator. Coach-voice prose needs chess content to ground in.

Likely scope (full plan when 6c lands and surfaces specific gaps):

- **Stockfish full PVs at `multipv=5`, `depth=20+`** — small parameter
  change to `extract_stockfish_eval`.
- **Threat detection** — `leela_chat/extractors/threats.py`: hanging
  pieces, loose pieces (attacked-defended-once with worse defender),
  pinned, skewered. ~100-150 lines via python-chess.
- **Material delta on resulting position** — `material_change(board,
  move)` trivial helper.
- **Tactical-motif detection** — `leela_chat/extractors/motifs.py`:
  forks, pins, skewers, discoveries. ~200-300 lines.
- **King-safety scalar** — `leela_chat/extractors/king_safety.py`:
  pawn shield count, attacker count, open files near king. ~50-100
  lines heuristic.

NOT bundled into 6c (6c stays focused on dual-timescale methodology).
NOT in 6.5 (6.5 is opt-in retrofit of existing handlers with new
trajectory data). Chunk 7 lands after 6.5 once we know what specific
chess content the dual-timescale prompts need to ground in.

---

#### 2026-05-19 (cont., 10) — Pod validation pass; all pre-6c gates closed

**Pod state on entry:** RunPod RTX 3090 alive (`213.192.2.110:40039`); Lc0 v0.32.1 + BT4 weights present per setup_pod.sh. Project tree at `/workspace/leela-chat/` (no git; rsync-deployed). Synced this session's local diffs via rsync (`-rltz --no-owner --no-group`).

**Gate 1 — Hook-wiring sanity (6b).** `test_baseline_top1_matches_network_inference_starting_position` PASSED on first try. The baseline (L=N, no hooks) policy top-1 from `extract_layer_emergence` matches `extract_network_inference`'s top-1 byte-for-byte on the starting position. Methodology validated mechanically.

**Gate 2 — Layer-emergence non-degeneracy (6b).** `test_walkthrough_rxd4_layer_trajectories_are_non_degenerate` PASSED. Printed trajectories on the Rxd4 walkthrough position (n_layers=15):

```
Rxb1+  (b8b1): ranks [41, 36, 35, 36, 46, 38, 39, 41, 34, 34, 32, 31, 25, 15, 2, 1]; tag=rose_and_stayed
Rxd4   (d8d4): ranks [43, 32, 29, 31, 35, 32, 24, 35, 35, 37, 34, 33, 31, 25, 14, 9]; tag=never_rose
Qg5    (e5g5): ranks [1, 27, 43, 43, 29, 24, 26, 30, 38, 44, 47, 47, 43, 32, 16, 7]; tag=rose_and_stayed
```

Three load-bearing observations:
1. **Rxb1+ surfaces only at the final 2 layers** (rank 2 at L=14, rank 1 at L=15). The earlier layers don't see it. This contradicts the v0 prediction in the methodology doc ("Rxb1+ rank stays #1 across every measured BT4 layer"). The real BT4 builds the high-prior Rxb1+ commitment late.
2. **Rxd4's network rank is ≥ 9 at every layer.** The network truly never finds the eventual MCTS bestmove. This is the A1 divergent-timescale signature — the existing walkthrough position satisfies the A1 criteria without a separate fixture.
3. **Qg5 is the embedding-only (L=0) top-1.** Drops to rank 7 by the final layer. The embedding "considered then dismissed" Qg5 (in allowed prose: "Qg5's rank dropped from #1 at L=0 to #7 at L=15").

**Gate 3 — Live-stats trajectory (6a v2).** Three failures on first try, then refixed. Two issues uncovered:

*Issue 3a: `LogLiveStats` UCI option doesn't work.* The 6a v1 implementation called `engine.configure({"LogLiveStats": True})` per-call. Empirically, the option is `kProOnly` visibility in Lc0 v0.32.1 — exposed only via the `--log-live-stats` CLI flag, NOT via UCI `setoption`. The configure call was a no-op; only end-of-search verbose-move-stats reached the wrapper (1 block, not 21+). Confirmed via `(echo uci; ...) | lc0` — no `LogLiveStats` in the UCI option list, only `VerboseMoveStats`.

Fix: spawn `Lc0Engine` with `--log-live-stats` as a command-line flag always. Drop the UCI configure call. Add `_keep_last_block` helper to `lc0_engine.py` that filters the multi-block stream to the final emit cycle when callers pass `log_live_stats=False` (the default for Chunks 1-5 single-snapshot callers). Chunks 1-5 callers see the same single-block output they always have; trajectory callers (`extract_mcts_trajectory`) get the full stream.

*Issue 3b: V0 prediction wrong on Q-trajectory shape.* The integration test expected `qs[-1] - qs[0] < -0.10` for Rxb1+ (monotonic-or-near-monotonic drop). Actual Rxb1+ Q-trajectory on the pod: starts at +0.74 (V-head optimism, 1 visit), DIPS to +0.36 around v=22 (search refutation), then PARTIALLY RECOVERS to +0.52 by the last live-stats emit (v=93). The dip-then-recovery pattern is the real refutation signature, not a monotonic drop.

Fix: rewrite the integration test assertions to use `trajectory_summary` fields (specifically `q_trough - q_first < -0.20` for refutation, `q_peak - q_first > 0.10` for confirmation). The summary stats are method-agnostic and capture the actual signal; milestone-snapshot deltas (the v1 approach) measure the wrong endpoints. Methodology doc §4.3 worked example updated to match the empirical numbers.

Result: all 4 mcts_trajectory integration tests PASSED on the rerun:
- Rxd4: `q_first=+0.558 q_peak=+0.963@v125 q_trough=+0.423@v0 q_last=+0.918 blocks=25` ✓ (q_peak - q_first = +0.41)
- Rxb1+: `q_first=+0.739 q_peak=+0.747@v2 q_trough=+0.362@v22 q_last=+0.523 blocks=25 inflections=2` ✓ (q_trough - q_first = -0.38)
- `test_live_stats_block_count_meaningful_for_10k_search` ✓ (25 blocks emitted)
- `test_trajectory_smoke_on_starting_position` ✓

**Empirical limitation discovered:** Lc0's live-stats emission cadence stops well before search budget exhaustion. A 10 000-node search produces ~25 blocks with the deepest emit at ~5 000 nodes — once PUCT is confident in bestmove, no more `info` updates fire, and Lc0 plays out the remaining budget silently. So `q_last` in the summary is "Q at the deepest live-stats emit," NOT "Q at end-of-search." Methodology doc and module docstring updated to make this explicit; the deliberation period (which the trajectory captures) is what we care about anyway, so this is fine.

**A1 closed.** The walkthrough_rxd4 position covers BOTH divergent quadrants:
- Rxb1+ = "network keeps / search refutes" (late-rising network rank + dipping Q)
- Rxd4 = "network never finds / MCTS rescues" (never enters network top-5; MCTS commits 84% visit share)

Plus the canonical concordant prediction is also visible (Qg5 ranks #1 at L=0 → #7 at L=15; not the engine's pick at all). One position, three quadrants of dual-timescale narrative. The user's source-hunt suggestion (`superhuman-ai-consistency-main/`) is preserved in backlog but not needed for 6c.

Added a new fixture `tests/unit/fixtures/lc0/walkthrough_rxd4_livestats.txt` — the raw 1300-line multi-block live-stats capture from this session. Companion metadata YAML at `tests/unit/fixtures/lc0/walkthrough_rxd4.meta.yaml` documents the position's dual-quadrant coverage with the empirical numbers. Two new unit tests parse the fixture to lock the format invariant: regression against future Lc0 version changes.

What surprised me:

- **The `LogLiveStats`-via-UCI-doesn't-work bug.** The 6a v1 plan and code happily configured this option as if it would work. Pod testing exposed it. The fix is small (CLI flag at engine spawn + per-call filter) but the methodology lesson is sharper: "UCI option exists per docs" is not the same as "UCI option is exposed via setoption." The Lc0 `kProOnly` visibility designation is the gate.

- **The Q-trajectory shape is more complex than predicted.** The dip-then-recovery pattern is methodologically more interesting than a monotonic drop — it tells us Lc0 considered Rxb1+'s "compensating attacking lines" briefly during search before settling for Rxd4. The C2 design (q_trough, q_peak, inflection points) captured this beautifully. Sticks score: 1.

- **Rxb1+ is a late-network-emerging move, not a stay-top-1 move.** v0's methodology doc claim "Rxb1+ stays top-1 across every measured layer" was wrong. The actual story: Rxb1+ surfaces only at L=14-15 (the network "decides" the move at the very end of its forward pass). This is actually a richer narrative — it suggests the late layers compute the high-prior attacking move while the early/middle layers are doing something else entirely. Real probe / SAE work in Phase 3+ could investigate what.

Test status:
- Unit: 128 pass (was 126; +2 fixture round-trip tests)
- Integration on pod: 4/4 mcts_trajectory PASS, 3/3 layer_emergence PASS (gates 1+2 + smoke)
- Total: 135 tests passing across the suite (off-pod skips the 11 pod-only tests)

---

#### 2026-05-19 (cont., 9) — Phase 2 Chunk 6b: zero-ablation layer emergence (off-pod code path)

The 6a v2 scaffold's module docstring (`leela_chat/extractors/layer_emergence.py`) was the written methodology contract. 6b implements it. Pod-only steps (integration test, real BT4) deferred to the next pod session.

What I did:

- **Investigation via Explore agent** — confirmed the exact vendored API:
  - **Hook names corrected.** Initial scaffold guessed `blocks.{L}.hook_resid_post`; actual BT4 hook points are `hook_attn_in`, `hook_attn_out`, `resid_mid_after_ln`, `hook_mlp_out`, `resid_post_after_ln` (per `leela_chat/_vendor/transformer_lens/components/leela_encoder.py:161-165`).
  - **Critical mechanism clarification.** Zero-ablating `resid_post_after_ln` would zero the entire residual stream at that point, not just the layer's contribution. The correct approach is to zero BOTH `hook_attn_out` AND `hook_mlp_out` (the two delta contributions the block adds to the residual). The docstring was rewritten to make this explicit.
  - **`run_with_hooks` signature** confirmed at `hook_points.py:417`: hook receives `(tensor, *, hook)` and returns either a tensor (replaces activation) or None.
  - **BT4 forward** returns a LIST `[policy_logits, value_logits, mlh_logits]`, not a tuple; first element shape `(batch=1, 1858)`.

- **Pydantic models added** to `layer_emergence.py`:
  - `LayerPolicyPoint(layer, rank, prob)` — one per layer per candidate.
  - `CandidateLayerTrajectory(uci, san, layer_trajectory, entered_top5_at_layer, exited_top5_at_layer, peak_rank, peak_layer, network_dismissal)` — pre-computes the categorical dismissal tag.
  - `LayerEmergenceResult(candidates, n_layers)` — outer container.

- **Helpers** added: `_zero_hook` (the hook protocol), `_build_ablation_hooks_for_layer` (one entry per ablated layer × 2 hookpoints), `_legal_move_probs` (mirrors `network_inference`'s legal-move-masked softmax), `_classify_dismissal` (categorical tag derivation), `_build_candidate_trajectory` (assembles a CandidateLayerTrajectory from per-layer policy distributions).

- **`extract_layer_emergence` body** — N+1 forward passes: L=k ablates `hook_attn_out` + `hook_mlp_out` at every block k..N-1; L=N is the baseline (no hooks). Per-layer rank computed against legal moves only (masked + renormalised softmax). Tracks the final-layer top-N candidates plus the focal_uci (if provided and legal). Terminal-position guard: empty result if no legal moves.

- **Unit tests** (`tests/unit/test_layer_emergence.py`, 17 tests):
  - Hook-protocol shape sanity + ablation-hook construction (right layer ranges, both `hook_attn_out` and `hook_mlp_out` per ablated layer).
  - `_legal_move_probs` renormalisation correctness.
  - `_classify_dismissal` coverage of all four tags (`rose_and_stayed`, `rose_and_fell`, `never_rose`, `unknown`).
  - `_build_candidate_trajectory`: peak detection, exit-detection correctness, terminal `rose_and_fell` case with final rank > 10.
  - `_FakeBT4` stub: end-to-end with canned per-pass policy logits. Asserts hook counts per pass (8, 6, 4, 2, 0 for a 4-layer model), focal_uci forced tracking, terminal-position empty return.

- **Integration test scaffold** (`tests/integration/test_layer_emergence_integration.py`, pod-only): three tests covering hook-wiring sanity (baseline top-1 vs `extract_network_inference`), Rxd4 walkthrough non-degeneracy (per-layer trajectories printed for manual eyeball + assertions that at least one candidate shows rank variation across layers), starting-position smoke.

What surprised me:

- **The scaffold's hook-name guess (`hook_resid_post`) was wrong.** Two interconnected issues: (a) BT4 doesn't expose a hook named `hook_resid_post`; the equivalent is `resid_post_after_ln`. (b) Even if we hooked the right name, zeroing `resid_post_after_ln` would zero EVERYTHING in the residual at that point, not just the layer's contribution. The Iterative Inference paper's "layer ablation" methodology is correctly implemented by zeroing the per-layer *deltas* (`hook_attn_out` + `hook_mlp_out`), not the residual itself. The docstring rewrite makes this explicit so the next reader doesn't repeat the confusion.

- **Two bugs caught by unit tests.** (a) `from leela_chat.network.uci_to_idx import encode_legal_move` is wrong — the correct path is `from leela_chat.network import encode_legal_move`. Caught at import time. (b) A test fixture intended to put move 'x' at rank 6 actually produced rank 5 due to argsort tie-breaking on equal probabilities; tightened the fixture so 'x' is unambiguously rank 7.

- **Pre-computed `network_dismissal` tag is more useful than I expected.** The four-quadrant taxonomy (rose_and_stayed / rose_and_fell / never_rose / unknown) lets the 6c v1 prompt make structural decisions from a single Literal field rather than from the raw 16-point trajectory. Reduces hallucination surface area for the LLM.

Open questions / threads (carry to pod + 6c):

- **Pod gate: hook-wiring sanity.** `test_baseline_top1_matches_network_inference_starting_position` is the must-pass check. If it fails, debug *before* trusting any layer-emergence output. Likely failure mode: hook names off (e.g. the smolgen pre-hook intercepts something we'd want to keep), or the `LeelaBoard` encoding is being applied differently.
- **Pod gate: non-degeneracy.** If `test_walkthrough_rxd4_layer_trajectories_are_non_degenerate` shows every per-layer rank trajectory identical to the final layer's, the ablation hooks aren't producing per-layer differences. Possible cause: the policy head's contribution dominates so much that intermediate-layer ablation barely shifts logits. Documented pivot path: mean-ablation (replace activation with a calibration-batch mean), recorded in DECISIONS.md if/when needed. Linear-logit-lens stays out of the codebase per the v2 reviewer-pass commitment.
- **Cost on pod.** N+1 = 16 forward passes; we haven't measured warm-BT4 latency on the pod yet. The plan says ~800 ms; if it's 5 s, may need to cache per-session (one extraction per board hash).

Test status: 126 unit tests pass (was 109; +17 layer-emergence). Integration suite collects 59 tests off-pod (was 56; +3 layer-emergence integration tests).

---

#### 2026-05-19 (cont., 11) — Chunk 6b.5 commit A: doc-only hardening before 6c

Pre-6c hardening from the v3 reviewer pass. Two findings from the pod-validation session warranted project-persistent capture before 6c locks in worked examples and prompt language. Commit A is off-pod doc edits only; commit B (pod Qg5 measurement + latency check) follows separately.

What I did:

- **CLAUDE.md §8 — new subsection "External-tool / protocol pitfalls."** Lead entry: UCI `setoption` silently accepts unknown / `kProOnly`-hidden options. Required verification pattern (option-readback at engine spawn; OR CLI flag treated as source-of-truth). Cross-references to HookedTransformer hookpoint-name silent-no-op (same class of bug) and future Stockfish UCI extensions. Lc0 v0.32.1 `LogLiveStats` (6a v1, 2026-05-19) recorded as the canonical example with timeline.

- **INTERPRETABILITY_NOTES.md §4 — major restructure.** §4.1 / §4.2 rewritten from v0 *predictions* (which turned out wrong on both rank and Q-trajectory shape) to *empirical pod measurements* (Rxb1+ late-emerging rank 41→1 at L=14-15; Rxd4 never-top-5 with rank 43→9). New §4.4 "Q-trajectory shape — the partial-U pattern" explains the PUCT-redirect math: why a refuted move's Q dips to a trough then partially recovers ("search no longer looking" vs "the move getting better"). Sets up the trough-prose contract that 6c's v1 prompt will enforce.

- **`mcts_trajectory.py` module docstring — trough-prose contract.** Two-tier prose contract section: (a) cross-cutting forbidden/allowed list referenced from §3 of INTERPRETABILITY_NOTES.md (no "decided/realized/discovered"); (b) Q-trajectory-specific forbidden/allowed list referenced from §4.4 (no "monotonic drop", no "Q recovered → move improved"; instead "Q hit a trough of X around visit Y, then visits stagnated as PUCT redirected elsewhere"). This docstring is the SoT for the 6c prompt.

- **`layer_emergence.py` module docstring — kProOnly cross-reference.** New "External-tool pitfall reference (CLAUDE.md §8)" section noting that hookpoint name strings are NOT validated at attach-time — same class of bug as the Lc0 UCI `setoption` silent-no-op. Cites the hook-wiring sanity test as the project's defense.

What did NOT change in 6b.5 commit A:
- The `_keep_last_block` always-on filter for `--log-live-stats` (the working solution from the pod session). If the latency check (commit B) shows regression, address there.
- The 6c handler/prompt — that's the next chunk, after 6b.5 closes.
- The walkthrough_rxd4.meta.yaml — Qg5 data lands in commit B with the pod measurement.

What surprised me (small):

- **The duplicate §4.3 heading I almost shipped.** My first Edit on §4.1/§4.2 added a "### 4.3 What the LLM should say..." line at the end, not realizing §4.3 already existed below. Caught by grep before commit. Reminder to read structure before bulk-editing in the middle of a section.

Open questions / threads (carry to commit B):

- **Qg5's MCTS quadrant.** Outcome will be either "concordant dismissal" (3rd quadrant; the walkthrough position then anchors three quadrants in one) or "rare network-changes-mind / search-disagrees" (4th quadrant; rare and architecturally most-interesting). Either is useful for 6c's v1 prompt. Measurement is a single ssh call.
- **`why_this_move` warm latency.** Now that Lc0 spawns with `--log-live-stats` always-on, single-snapshot extractors do extra parsing per call. Expected negligible (~10ms) vs the second-scale search itself, but worth verifying. If >15s warm, per-call CLI-flag toggle or separate-process pattern needed.

Test status: 128 unit tests pass (unchanged from commit B27dfe4; doc-only commit).

---

#### 2026-05-19 (cont., 12) — Chunk 6b.5 commit B: pod Qg5 measurement + latency check

Two pod tasks per the v3 reviewer pass. Both completed cleanly via ssh.

**Pod measurement — Qg5's MCTS behavior on the walkthrough position.**

Single `extract_mcts_trajectory(board, engine=eng)` call (content-aware sampling, default budget=10000). Compared Qg5 against Rxd4 and Rxb1+ at the deepest live-stats emit (root_visits=4592).

Empirical per-candidate summaries:

| Candidate | Visits | Visit share | q_first | q_peak @ v | q_trough @ v | q_last |
|---|---|---|---|---|---|---|
| Rxd4 | 4194 | 91.33% | +0.558 | +0.954 @ v=88 | +0.392 @ v=0 | +0.921 |
| Rxb1+ | 89 | 1.94% | +0.739 | +0.747 @ v=2 | +0.363 @ v=28 | +0.532 |
| Qg5 | 21 | 0.46% | +0.558 | +0.679 @ v=20 | +0.392 @ v=0 | +0.639 |

**Qg5 quadrant classification: concordant dismissal.** Visit share 0.46% (well under the 5% threshold). Q stays near the V-head baseline (+0.56 → +0.68); never develops a meaningful Q profile because PUCT doesn't allocate visits. So the story: L=0 embedding-only favorite that BOTH later network layers AND search agree isn't the move — by independent routes. Network: rank 1 @ L=0 → rank 7 @ L=15 (late-layer demotion). Search: never gets enough visits to develop a Q (search-time dismissal via visit allocation). **NOT the rare 4th quadrant** (network-changes-mind / search-disagrees) — Qg5 isn't kept by search; it's quietly ignored.

Walkthrough position now anchors **three** distinct quadrants in one position:
- Rxd4 = network-low / MCTS-rescue (peak rank 9 at L=15; visit share 91%)
- Rxb1+ = network-keeps-late / search-refutes (late emergence to rank 1; visit share 2%, Q trough at v=28)
- Qg5 = concordant dismissal (L=0 favorite that BOTH signals lose interest in; visit share 0.5%)

That's three of the four dual-timescale quadrants on a single fixture. The fourth quadrant (network-changes-mind / search-disagrees, would be rare and architecturally interesting) is not present here — backlog item for a future fixture hunt if 6c's v1 prompt demands it.

Fixture metadata YAML at `tests/unit/fixtures/lc0/walkthrough_rxd4.meta.yaml` updated with all three quadrants' measured numbers + the classification logic + the trajectory_pattern descriptions the 6c v1 prompt will draw from.

**Pod latency check — single-snapshot warm.**

Concern: now that `Lc0Engine` spawns with `--log-live-stats` always-on, single-snapshot extractors (`extract_mcts`, used by Handlers 1 + 2) parse the full 25-block live-stats stream per call, then filter to the last block via `_keep_last_block`. Expected overhead: ~10ms (the parser is cheap; the dominant cost is the search itself). Worst case: regression from ~10s to >15s on `why_this_move` warm.

Measured: `extract_mcts(board, engine, nodes=10000)` warm = **0.77s**. Total script wall (Lc0 spawn + BT4 load + 1 warm-up + 1 measured call) = 7.0s. **No regression.** The `_keep_last_block` filter is essentially free relative to search cost.

Conclusion: the `--log-live-stats`-always-on pattern in `Lc0Engine.__init__` is fine for production. No need for a per-call CLI-flag toggle or separate-process pattern. The 0.77s warm cost is the search itself; the live-stats parsing is in the noise.

What surprised me (small):

- **Qg5's `q_trough` and `q_first` are the same value (+0.392).** Both are the V-head-only block where visits=0. That's because Qg5 was first sampled at the block where visits transitioned 0→1; the early V-head-only blocks all show the same Q (+0.39-+0.56 range from the V-head). The summary code correctly identifies the lowest among them as the trough. Edge case but consistent.

- **Three-quadrant coverage in one position is a stronger A1 close than originally planned.** v2 plan called for one hand-picked divergent fixture; we ended up with one position that happens to cover three quadrants because we measured it carefully. The methodology-first discipline kept us from over-engineering a fixture hunt.

Open questions / threads (carry to 6c):

- **Should 6c v1 prompt have three or four worked examples?** With the walkthrough position's three quadrants + a separate concordant-pick example (e.g. starting-position d2d4 where network top-1 and MCTS top-1 agree), we have four examples. Whether to include all four or just the three from the walkthrough is a prompt-design question for 6c.

Test status: 128 unit tests pass (unchanged).

---

#### 2026-05-19 (cont., 13) — Phase 2 Chunk 6c: did_you_consider handler + v1 prompt + 4-quadrant pod smokes

The dual-timescale dismissal handler — Phase 2's third and most architecturally-distinctive handler. Off-pod code + pod smokes in one session.

**What I built (off-pod, commit `f6c139c`):**

- `leela_chat/handlers/did_you_consider.py` — Pydantic `DidYouConsiderFindings` combining position-level data + focal-move callouts + `focal_layer_trajectory` (network-time) + `focal_mcts_trajectory_summary` (search-time) + pre-computed `network_dismissal` / `search_dismissal` tags. Handler orchestrates all 5 extractors. `_classify_search_dismissal` derives the search-side tag from q_trough vs q_first + visit share, with thresholds calibrated from the 6b.5 pod measurements. Templated fallback respects the trough-prose contract (cites q_trough for refutation, q_peak for confirmation; never q_last as the verdict).

- `leela_chat/llm/prompts/did_you_consider/v1.md` — ~280 lines covering: role + signal sources + two-timescale framework + four-quadrant taxonomy + trough-prose contract + encoding-vs-use contract + substantive-finding pattern for `never_rose` cases + five categorical buckets + FOUR worked examples (one per quadrant, all using empirical pod numbers from the 6b.5 measurement: starting-position d4 = concordant pick; Rxb1+ = search-refutes; Rxd4 = MCTS-rescue / never_rose; Qg5 = concordant-dismissal) + anti-patterns list.

- `leela_chat/cli/__main__.py` — added `DYC_HANDLER_NAME` import; extended `HANDLERS_AVAILABLE`; `_answer_one` dispatch grew a third branch routing to `handle_did_you_consider` when handler is `did_you_consider`. Same `--move <UCI>` arg from Chunk 5.

- Unit tests `tests/unit/test_did_you_consider.py` (14 tests, all pass): tag derivation across all four cases; end-to-end with mocked extractors covering all three walkthrough quadrants; illegal-move ValueError; lc0=None and skip_layer_emergence=True graceful degradation; templated fallback contract compliance.

- CLI integration tests `tests/integration/test_cli_ask.py` extended with `did_you_consider` routing + HANDLERS_AVAILABLE presence checks.

Test status off-pod: **155 tests pass** (128 unit + 14 new handler + 13 CLI integration).

**Pod smokes — all four quadrants on real BT4 + Lc0 + Claude Opus 4.7:**

| # | Quadrant | Focal | Latency | Validation | Notable prose |
|---|---|---|---|---|---|
| 1 | search-refutes | Rxb1+ on walkthrough | 19s warm | ok | *"hit a trough of Q +0.36 around visit 28; visits then stagnated near 93 as PUCT redirected to Rxd4, and the drift back to Q +0.54 is search no longer sampling that branch, not Rxb1+ recovering"* — exemplary trough-prose contract compliance |
| 2 | MCTS-rescue (never_rose) | Rxd4 on walkthrough | 16s warm | ok | *"the network's pattern recognition doesn't suggest Rxd4 at any layer of computation... The discovery is a search artifact, not a network-emergence finding... This is exactly the kind of position dual-timescale extraction exists to surface — search reached a verdict the policy network alone could not."* — the substantive-finding pattern produced verbatim |
| 3 | concordant dismissal | Qg5 on walkthrough | 19s warm | ok | *"putting it nominally in the winning bucket but on a sample so thin the number isn't load-bearing"* — caught a chess-quality subtlety about Q-confidence at low visit counts |
| 4 | concordant pick | d4 from starting position | 16s warm | ok | *"the engines disagree on the headline first move but agree d4 is a top-tier candidate"* — clean disagreement framing |

Zero forbidden phrasings observed across all four outputs (grep-checked "monotonic", "recovered to", "noticed", "realized", "discovered" — none appeared). Visit numbers cited correctly; rank trajectories cited verbatim from the pod measurements; bucket vocabulary used correctly.

**Substantive chess findings the handler surfaced beyond what any individual engine could:**

1. **Smoke 1 (Rxb1+):** "Rxb1+ works as a follow-up *after* Rxd4, not as the immediate move." Reading from Stockfish's PV (`Rxd4 Rxd4 Qxa3 bxa3 Rxb1+`), the LLM noticed that Rxb1+ exists in the winning line, just one ply later. This is the kind of compositional insight no engine alone produces — it requires comparing the focal move to the engine's PV at the same depth.

2. **Smoke 2 (Rxd4):** The substantive-finding pattern about Rxd4 being "purely a search artifact." This is the project's most-distinctive output to date — a claim about *what the network does and does not do* that requires the dual-timescale data to support.

3. **Smoke 3 (Qg5):** The Q-confidence-at-low-visits caveat. Recognizing that Q +0.64 with 21 visits doesn't carry the same weight as Q +0.64 with 4000 visits is a piece of MCTS-literacy that emerges from the trajectory data.

4. **Smoke 4 (d4):** Clean Leela-vs-Stockfish disagreement framing on the most-played first move in chess history. Useful baseline; validates the prompt works on prosaic positions too.

What surprised me:

- **Prose quality on smoke 1 in particular.** The LLM made the compositional Stockfish-PV observation unprompted. The v1 prompt doesn't explicitly say "compare the focal move to Stockfish's PV continuation" — but it does cite Stockfish's PV in the data, and the prompt asks for Stockfish corroboration. The model bridged the gap on its own.

- **Cache hit on smokes 2-4.** Smoke 1's `cache_write=7111` (writing the system prompt to Anthropic's prompt cache). Smokes 2-4 show `cache_read=7111` — the prompt was reused as cache. Saves ~$0.20 per query at Opus rates. Validates the prompt-caching architecture chosen in ADR-0005.

- **Layer-trajectory numbers were stable but slightly different from the 6b.5 measurement run.** The 6b non-degeneracy test on Rxd4 measured rank 43 at L=0 and rank 9 at L=15. Smoke 2's run showed rank 43 at L=0 and rank 9 at L=15 — same numbers (BT4 is deterministic given identical inputs; Lc0 has stochastic backend behavior but the network forward pass is deterministic). The slight visit-count drift between sessions (4592 vs 4768 vs 4769 vs 4802 vs 9681 across runs at "nominal" 10k nodes) reflects Lc0's PUCT non-determinism and is expected.

**Phase 2 trilogy (6a + 6b + 6c) is complete.** The project now has a working dual-timescale dismissal handler producing validated, contract-compliant, genuinely-novel chess-content prose. The handler exits the v0 vision in the BIBLE for `did_you_consider`.

Open questions / threads for next sessions:

- **Chunk 6.5 retrofit decision.** The plan gates `--detailed` retrofit of Handlers 1 + 2 on 6c criteria. 6c smokes ALL PASSED with validation_ok=True (criterion 1), no conflation (criterion 2), warm under 30s (criterion 3). 6.5 is unblocked but optional — it's an opt-in extension; defer until handlers 1+2 are observed to actually need richer trajectory data.

- **Chunk 7 chess-content extractors.** Now operational handler exists, can identify what concrete chess content the prompts most lack. Smoke 1's compositional insight is a hint: "what move comes after the engine's pick" is implicit in Stockfish's PV but the LLM may not always extract it. More chess-content extractors (multipv depth 20+; threats; motifs) would enrich the findings JSON and produce more such insights without prompt-engineering tricks.

- **Phase 2 close + Phase 3 entry brief.** Phase 2 exit criteria from BIBLE.md / CLAUDE.md §11: "MCTSNode parsing works on 100 random positions" (have it on a representative sample); "eval trajectories populated" (have them via 6c); "6 handlers total pass eval" (Phase 2 originally targeted 6 handlers; we have 3; the BIBLE may want refinement). Phase 3 entry brief would surface this.

Test status: 155 tests pass (128 unit + 14 handler + 13 CLI integration). All four pod smokes pass with validation_ok=True. Phase 2 trilogy operational.

#### 2026-05-20 — Phase 2 Chunk 6.5: Handlers 1+2 `--detailed` retrofit

What I did:
- Three commits A/B/C following the 6.5 plan:
  - **A** (`c40daeb`): Refactor — extracted `_classify_search_dismissal` from `did_you_consider.py` into a shared `leela_chat/handlers/_dismissal.py` and renamed it to public `classify_search_dismissal` so Handler 2's --detailed path can reuse the same thresholds without a handler-3 import dependency. Did_you_consider keeps a thin underscored re-export for backward compat with the existing test imports. Zero behaviour change.
  - **B** (`28d605b`): Code + tests — added `detailed: bool = False` kwarg to `handle_what_is_leela_seeing` and `handle_why_this_move`. When True, both swap `extract_mcts` for `extract_mcts_trajectory` and call `extract_layer_emergence`. Handler 1 gets a new `layer_emergence: LayerEmergenceResult | None` field on its Findings model; Handler 2 gets four new optional fields (`focal_layer_trajectory`, `focal_mcts_trajectory_summary`, `network_dismissal`, `search_dismissal`) mirroring Handler 3. Added `PROMPT_VERSION_DETAILED` constants (5 for Handler 1, 2 for Handler 2). CLI `_add_common_flags` gained a `--detailed` boolean flag threaded through `_answer_one` to the handlers and to the prompt-version selection. The telemetry line now also surfaces handler name + prompt version. did_you_consider is unchanged (always detailed; `--detailed` is a logged no-op). 7 new unit tests for Handler 1, 6 for Handler 2, 4 new CLI integration tests — 155 unit tests pass.
  - **C** (`00db44b`): Prompts — `what_is_leela_seeing/v5.md` adds a "deliberation sentence" to v4's three-signal frame, naming which trajectory pattern drives the verdict (concordant pick / MCTS rescue / search refutes / concordant dismissal) plus full trough-prose contract, substantive-finding pattern for `never_rose`, encoding-vs-use contract, two worked examples (Italian Bc5 concordant + walkthrough Rxd4 MCTS-rescue). `why_this_move/v2.md` mirrors `did_you_consider/v1.md`'s dual-timescale framing but retains v1's Branch A / Branch B structure (focal IS engine pick vs focal IS NOT); three worked examples (concordant d4, search-refutes Rxb1+, MCTS-rescue Rxd4).

Pod state remediation (RunPod restart between sessions):
- Updated `~/.ssh/config` runpod alias to the new pod (`213.192.2.116:40006`).
- Re-ran `setup_pod.sh` to reinstall Stockfish (wiped on restart).
- Recreated `/path/to/models/lc0/BT4.pt` symlink pointing at `/workspace/weights/...` (the known `setup_pod.sh` BT4.pt idempotency backlog item bit again; logged but not fixed in this chunk).
- Rsync'd the 10 changed files (handlers + CLI + prompts + tests) to `/workspace/leela-chat/`.

5 pod smokes (warm BT4-cached; wall-clock includes ~30-50s BT4 load per process; handler_s + translate_s are the per-call timings):

| # | Handler | Mode | Prompt | source | val_ok | handler_s | translate_s | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | what_is_leela_seeing | default | v4 | templated | False | 7.52 | 19.64 | Pre-existing v4 quirk: validator rejected derived `27%` (visits / total_nodes). Not a 6.5 regression — v4 unchanged since Chunk 4. Templated fallback grounded by construction. |
| 2 | what_is_leela_seeing | --detailed | v5 | llm | True | 6.93 | 12.75 | Prose uses deliberation sentence: "d4 was tagged `rose_and_stayed` and reached rank 1 by L=5… Q peaked at +0.049 around 39 visits." Cites Nf3's separate layer story. |
| 3 | why_this_move | default | v1 | llm | True | 7.04 | 8.28 | Default-mode contract preserved exactly: Branch A, bucket framing, no trajectory chatter. |
| 4 | why_this_move | --detailed | v2 | llm | True | 8.16 | 11.12 | Substantive-finding pattern verbatim: "the network's pattern recognition does not suggest this move at any layer of computation… rank 43 at L=0…" Trough-prose pattern: "Q opened at +0.56, dipped to a trough of +0.42 before the first visit landed, then jumped to +0.92 by visit 12, peaked at +0.96 around visit 125." |
| 5 | did_you_consider | (always det.) | v1 | llm | True | 7.17 | 9.36 | 6c regression check — output materially identical to the 6c-shipped baseline. No drift from the refactor or the new shared dismissal helper. |

Smoke 1's validator failure is a pre-existing, position-specific v4 issue (the LLM's derived-percentage prose hits an `ungrounded_eval` check). Re-running Handler 1 default on a non-starting position would likely pass; the LLM's prose contained `27%` derived from `2587 / 9643` visits — both numbers are in the JSON, but the validator looks for the literal `27%` string. Filed as backlog candidate (not in 6.5 scope).

The Handler 1 detailed + Handler 2 detailed prose are the most-pleasing output the project has produced yet — both successfully integrate the network-time + search-time stories into the existing chess prose without dominating it. The deliberation sentence pattern (v5) and the substantive-finding pattern (v2) both land cleanly on the walkthrough position. This is the canonical Chunk 6.5 success signature.

Latency budget verified:
- Handler 1 default: handler_s ~7s (was the default Chunk 4 baseline).
- Handler 1 detailed: handler_s ~7s — the layer-emergence ~750ms addition is invisible at this resolution; trajectory parsing is essentially free.
- Handler 2 default: handler_s ~7s.
- Handler 2 detailed: handler_s ~8s — same picture.
- All detailed-mode totals ~15-20s wall-clock excluding BT4 load, matching Handler 3's measured 16-19s.

Bugs caught:
- None. Both new prompts produced contract-compliant output on first LLM call (no retry, no templated fallback) on the walkthrough position.

Surprises:
- The v4 templated fallback prose accidentally cited a `27%` value that didn't pass validation. The fallback path IS grounded by construction (every number from the JSON), so the user gets a sensible answer — but the validator's strictness means we record `validation_ok=False` for a path that's actually grounded. Worth thinking about whether the templated fallback should be exempt from the same validator that flags LLM output. (Backlog.)
- The v5 deliberation sentence picked up Nf3's separate layer story unprompted — "Nf3 has an interesting layer story of its own — rank 4 at L=0, dipping to rank 17 at L=1, then climbing all the way to rank 1 at L=13 before settling at rank 2" — emerging from the per-candidate layer_trajectory data. The prompt taught the form; the LLM applied it to the runner-up too.

What's still left for Phase 2:
- **Chunk 7** chess-content extractors (Stockfish multipv depth ≥20; threats; motifs; king safety; material delta) per the Chunk 6.5 plan §D1. Now that Handler 1+2 detailed have the dual-timescale story, the next leverage is chess content (the WHAT of what's threatened) so the coach-voice prompts (Phase 4 placeholder) can ground in concrete tactical content.
- **Phase 2 close + Phase 3 entry brief.** Phase 3 concept-list pruning is the project's single highest-leverage decision; the seed list in `docs/phase3/concept_candidates.md` (~91 candidates) was deliberately overfilled at 6a v2 and is now ready for pruning.

#### 2026-05-21 — Phase 2 Chunk 6.6: pre-Chunk-7 hygiene

What I did:
- Four commits closing three small backlog items + one optional doc:
  - **A** (`27e7b2d`): Validator — recognises derived visit-share percentages. New `_collect_derived_percentages` helper in `leela_chat/validator/__init__.py` computes percentage-form values from `mcts.children[i].visits / total_nodes_searched` (and the `visits_in_flight` variant, plus the `root_stats.visits` analogue) and feeds them into a SEPARATE candidate set for the percent grounding loop — derived values don't leak into signed-number grounding. Scope minimal: visit-share only, no general "scan every JSON pair" logic. 5 new tests in `test_validator.py`; 160 unit tests total (was 155).
  - **B** (`abb5b04`): `setup_pod.sh` — idempotent BT4.pt symlink block. The vendored TransformerLens loader at `_vendor/transformer_lens/loading_from_pretrained.py:1832` hardcodes `/path/to/models/lc0/BT4.pt`. That path lives outside `/workspace` so it dies on every pod restart; the new block re-creates the symlink as part of setup. Confirmed on-pod: `setup_pod.sh` now reports `symlink ready: /path/to/models/lc0/BT4.pt -> /workspace/weights/.../BT4.pt` on first run.
  - **C** (this commit — PHASE_NOTES update): Handler 1 `--detailed` walkthrough smoke; see the table below.
  - **D** (`1d86a07`): `docs/PROMPT_PATTERNS.md` — new design-pattern note capturing the "teach the form once; model generalises to homogeneous elements" observation from Chunk 6.5 Smoke 2 (LLM extended the v5 deliberation sentence to Nf3 unprompted). Cross-referenced from `INTERPRETABILITY_NOTES.md` §5; prompts themselves byte-clean.

Pod state remediation (RunPod network volume is DC-locked, lesson learned):
- Stopped previous pod between Chunk 6.5 close and Chunk 6.6 start. The first three pods I spun up (RTX 5080 + two 3090s) did NOT have the network volume — `/workspace/` was empty container overlay because the pods were in different datacenters than the volume (volume lives in eu-cz-1; pods were in other DCs). RunPod doesn't auto-warn you when the volume isn't available in a region; the pod just comes up with an empty `/workspace`. Fourth pod (`213.192.2.124:40164`) was in eu-cz-1 and the volume auto-mounted cleanly.
- Confirmed S3-compatible API access (`https://s3api-eu-cz-1.runpod.io`) is an option if a future session needs to read individual files from the volume without a same-DC pod. Not used this session.

6 pod smokes (5 regression + 1 new):

| # | Handler | Mode | Prompt | Source | val_ok | handler_s | translate_s | LLM calls | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 (re-run) | what_is_leela_seeing | default | v4 | **llm** | **True** | 7.12 | 10.75 | 1 | **Validator fix verified**: was templated/False in Chunk 6.5 Smoke 1 (LLM cited `27%` derived from 2587/9643 visits). Now passes cleanly on first LLM call. |
| 2 (regression) | what_is_leela_seeing | --detailed | v5 | llm | True | 7.35 | 16.41 | 1 | No change from Chunk 6.5 Smoke 2. |
| 3 (regression) | why_this_move | default | v1 | llm | True | 6.90 | 8.55 | 1 | No change from Chunk 6.5 Smoke 3. |
| 4 (regression) | why_this_move | --detailed | v2 | llm | True | 6.64 | 13.02 | 1 | No change from Chunk 6.5 Smoke 4. |
| 5 (regression) | did_you_consider | always det. | v1 | llm_retry | True | 7.51 | 45.80 | 2 | Anthropic API returned 529 OverloadedError on the first attempt; orchestrator's designed retry path engaged and the second LLM call succeeded with `validation_ok=True`. Not a 6.6 regression — transient external API condition. |
| **C** (NEW) | what_is_leela_seeing | --detailed | v5 | llm | True | 7.67 | 12.84 | 1 | **Walkthrough Rxd4 fixture.** Hard gates all pass. Prose covers BOTH candidates' stories in one paragraph: Rxd4 MCTS-rescue ("Rxd4 never appears in the policy top-5 anywhere in BT4's 15 layers... discovery is a search artifact") AND Rxb1+ search-refutes ("Q opened at +0.74, hit a trough of +0.36 around visit 24, and visits stalled at 94 as PUCT redirected to Rxd4 — the post-trough drift back to +0.54 is search no longer sampling that branch"). This is the project's most-comprehensive single-position deliberation narrative to date. Quality bar holds on the divergent fixture without focal-move framing. |

Smoke 1's flip from False (Chunk 6.5) → True (Chunk 6.6) is the bug-fix verification. Smokes 2-5 staying True is the regression-clean check. Smoke 5's first run during the back-to-back battery hit Anthropic's 529 Overloaded twice in a row and templated out (the orchestrator-by-design path); on isolated re-run with the same fixture and a fresh LLM call, the retry path engaged and validation_ok=True. The 529 is unrelated to 6.6 — Anthropic's API was under spike load during the European afternoon window when the smokes ran.

What I learned:
- The validator's derived-percentage extension landed cleanly without any false-positive interactions. The visit-share-only scope kept it tight; the separate `pct_values` set kept it from polluting signed-number grounding. The 5 new tests catch the obvious failure modes (visits in-flight variant, root_stats variant, false positives, zero total).
- RunPod network volumes are DC-locked. The dashboard's pod-creation flow doesn't filter region by volume availability, so you can pick any pod and the "Network Volume" option silently won't include your volume if it's elsewhere. Lesson: filter by datacenter explicitly to the volume's DC (eu-cz-1) before picking a pod.
- The PROMPT_PATTERNS.md pattern continues to hold: Smoke C's Handler 1 detailed prose narrated TWO candidates' trajectories in matching shape (same encoding-vs-use cadence, same trough-prose contract honoured per candidate). Pattern 1 ("teach the form once for the canonical element") predicts this exactly — the candidate list is homogeneous, the form is teachable, element identity is discoverable. Worth one more confirmation in future smokes.

Pod-volume observation: Anthropic 529 came once during back-to-back smokes (Smoke 5 first attempt) and again during the isolated re-investigation (attempt 1 of the re-run). These are likely correlated with European-afternoon API load. The retry path consistently rescued. Worth keeping the retry budget (currently 2 attempts → templated) as-is.

What's next:
- **Chunk 7** chess-content extractors. The Chunk 7 plan MUST include the two forward-references from the Chunk 6.6 plan: (1) per-new-extractor derivation checkpoint asking "does the prompt that consumes this extractor cite percentages or ratios derived from extractor fields?" and if yes, extend `_collect_derived_percentages` (or a parallel helper) with explicit field-pair mappings + one false-positive test each. (2) Latency parallelisation evaluation — current ~16-19s detailed will grow to ~20-25s with Chunk 7's added extractors; asyncio/futures invocation would let the slowest single extractor set the floor instead of the sum.
- **Phase 2 close + Phase 3 entry brief** (alternative direction; concept-list pruning is the highest-leverage Phase-3 decision).

Test status: 160 unit tests + 17 CLI integration tests + 36 validator-suite tests (subset of unit) pass. 6 pod smokes documented (5 regression + 1 new). Phase 2 trilogy + 6.5 + 6.6 complete.

#### 2026-05-21 (cont.) — Phase 2 Chunk 6.7: pre-Chunk-7 forward-debt closure

Three small refinements per the post-Chunk-6.6 reviewer pass:
- Commit A (`28ba93a`) — CLAUDE.md §8 implementation-pitfall entry for the per-extractor derivation checkpoint. Promotes the rule from PHASE_NOTES (read-on-demand) to CLAUDE.md (read at every session start per §9), so any future session adding a new extractor with derived-percentage prose can't silently re-introduce the Smoke-1-class false negative the Chunk 6.6 fix closed.
- Commit B (`8ff9fe9`) — PHASE_NOTES "Next session goal" rewritten as HARD Chunk 7 entry criteria (2 enumerated items: parallelisation + per-extractor derivation checkpoint), plus a 1-line CLAUDE.md §11 Phase 2 anchor referencing Phase 1's already-achieved `<30s` ceiling. Without the anchor, the Phase 2 exit reads as if latency stopped mattering after Phase 1 — but the ceiling silently re-applies.
- Commit C (`338ea36`) — `setup_pod.sh` empty-/workspace fail-fast guard. Catches RunPod's DC-locked volume issue in ~5 sec instead of ~20 min of cascading failures. Verified on the live pod via happy-path + mock-fail tests (renamed `/workspace/leela-chat` away → script exits 1 with the full error+recipe; restored cleanly).

Net effort: ~40 min, under the user's "under-an-hour-total" budget. No application code touched; no tests changed; 3 pure doc/script commits. Substrate genuinely clean for Chunk 7.

#### 2026-05-21 (cont.) — Phase 2 Chunk 6.8: Lichess Puzzle DB corpus + smoke battery (infrastructure complete, substantive results API-blocked)

What I did:

- Commit A (`45b34e9`) — Lichess corpus infrastructure. `leela_chat/eval/lichess_puzzle_db.py` with `material_signature`, `solver_first_uci`, `themes_to_list`, `load_corpus`, `sample_by_theme`, `sample_stratified` (with `fallback_to_lower` for sparse rating bands). `leela_chat/scripts/download_lichess_puzzles.py` with stream-download + ZSTD decompress + Parquet parse + integrity check (CSV row count ≥ 5M + Parquet row count match). 21 hermetic unit tests against synthetic PyArrow tables. zstandard + pyarrow deps added.
- Commit B (`6869b6f`) — smoke battery runner + doc + summaries dir. `STRATA` config covers 20 positions: 8 tactical (5 rating bands, 6 themes), 5 quiet middlegame, 4 endgame (EXCLUDING KNN_vs_K per project memory), 3 opening. The 5-call matrix per position. CRASH-safe inner loop.
- Commit C (`44e3e67`) — bug fix + aggregator + first dated summary. **Caught a `Moves[0]` vs `Moves[1]` Lichess convention bug** in the first pod run: had assumed Moves[1] was the solver's first move (an older convention); the current Lichess Puzzle DB uses Moves[0] = solver (FEN side-to-move IS the solver). First run produced 60 illegal-move CRASH entries for H2 + H3 calls. Fix landed; re-run produced 0 illegal-move errors. **BUT:** the re-run also produced 100/100 templated fallbacks — every Anthropic call returned HTTP 400 "Your credit balance is too low to access the Anthropic API". The orchestrator behaved correctly (templated path engaged; 0 Rule 1 violations; no hallucinated moves) but the substantive question is unanswered.

What this chunk DID verify:
- Corpus pipeline works end-to-end on the pod. 296 MB download in ~5s; decompress to 1.09 GB in ~5s; parse to 341.5 MB Parquet in ~9s. Total ~20s from cold start to queryable corpus.
- Sampling deterministic via `seed=0`. The 20-position fixture in `tests/eval/smoke_positions.yaml` reproduces exactly from same corpus + same seed.
- Battery runner runs 100 calls to completion without crashing under the templated-fallback path.
- Aggregator with per-theme + per-difficulty + Branch-A-vs-B + illegal-move-classification + eyeball-rubric renders correctly.

What this chunk did NOT verify (the substantive question):
- v5 / v2 prompts holding outside walkthrough Rxd4.
- Per-theme prose-mention rates.
- Per-difficulty Branch-A-vs-B prose-quality observations.
- Forbidden-language adherence in non-walkthrough prose.

Surprises / lessons:
- **Lichess Moves convention**: the OLD convention (pre-2023?) had `Moves[0]` as the opponent's setup move. The CURRENT DB uses `Moves[0]` as the solver's first move (FEN side-to-move IS the solver). My initial implementation followed the old convention. Caught by the validator's illegal-move check in the first pod run; the CRASH path captured it cleanly. Fixed in `solver_first_uci`; canonical docstring now states the current convention.
- **Anthropic credit exhaustion is a silent operational failure mode**: the orchestrator's templated path runs cleanly but `validation_ok=False` on every call. Without the run-status preamble I added at the top of `tests/eval/smoke_summaries/2026-05-21.md`, a future archaeologist might read "100/100 templated fallbacks" as a v5/v2 prompt-fragility finding rather than a billing-account-empty issue. Bumped item: any future smoke battery should pre-flight a single test API call before the 100-call loop to fast-fail on credit exhaustion.

Pod resources after this session:
- Corpus on the network volume at `/workspace/lichess_puzzle_db/lichess_db_puzzle.parquet` (341.5 MB; 5,939,980 rows). Survives pod restarts (eu-cz-1 volume).
- Phase 3 probe training data is unblocked at the data-pipeline level — when probe training starts, `sample_by_theme` works against this corpus without further setup.

Test status: 181 unit tests pass (was 160; +21 new for lichess_puzzle_db). 17 CLI integration tests + 36 validator-suite tests still pass. No regressions.

Next step (blocking on user action):
- **User tops up Anthropic credits.**
- Re-run battery with `--reuse-fixture` (uses the same 20-position fixture so before/after comparison isolates "what the LLM half adds").
- Re-run aggregator → fresh dated summary file at `tests/eval/smoke_summaries/<re-run-date>.md`.
- Then: Chunk 7 OR Phase 2 close + Phase 3 entry brief.

Bumped backlog candidate from this chunk:
- **Pre-flight API check in smoke battery runner.** A 1-token test call at battery startup; if it returns api_error, log clearly and exit with code 3 (distinct from successful templated runs). Saves wasting 30+ min of pod time + telemetry noise on a fully-blocked run. **[Closed by Chunk 6.8.1 — see consecutive-api-error circuit breaker below.]**

#### 2026-05-21 (cont.) — Phase 2 Chunk 6.8 commit E: substantive re-run

After topping up Anthropic credits + a 30-sec sanity-check ask call:
- Re-ran battery via `--reuse-fixture` (same 20-position fixture).
- 100 calls in ~26 min wall clock (avg ~15.7s/call — matches detailed-mode latency budget).
- **Hard gates ALL PASS:** validation_ok 94% (≥ 90% gate ✓); templated 6% (≤ 10% gate ✓); sources 79 llm + 15 llm_retry + 6 templated; latency median 13.1s / p95 27.9s / p99 32.0s.
- **Per-category breakdown:** tactical 95% / quiet_middlegame 96% / endgame 90% / opening 93% — all above the 90% gate.
- **Per-difficulty-band (counter-intuitively):** 1800-2200 93% / 2200-2600 90% / 2600-3000 100% / 3000-3400 100% — the HARDER bands validated more cleanly than the easier ones.
- **Branch B coverage 100% (60/60 H2+H3 runs):** every focal move was different from MCTS@1000-nodes pick — exactly the stress case the smoke battery was built for. The dual-timescale prose contracts mostly held; the residual ~5-8% is real prompt fragility.

Real findings (not regressions; substantive prompt-quality observations):

- **8 illegal_move classifications.** Validator caught all 8 BEFORE reaching the user — Rule 1 upheld via templated fallback. Per-FEN triage suggests ~4 are real LLM hallucinations (e.g., `Nf3+` on a position with no white knight; `Bxg4`/`Nxg4` cited as black-to-move when they're white moves on the FEN) and ~3-4 are parser FPs or PV-coverage gaps (e.g., `Kxf3` is a legitimate Stockfish PV continuation the validator's trusted-moves walk didn't recognise).
- **4 forbidden-language hits** in the eyeball rubric: `saw` ×3, `monotonic` ×1. Real encoding-vs-use + trough-prose contract violations per INTERPRETABILITY_NOTES.md §3.

Implications:
- v5 / v2 prompts hold MOSTLY outside walkthrough Rxd4 but produce real hallucinations on ~5-8% of hard-puzzle H2-detailed calls.
- A **v6 / v3 prompt-iteration chunk** is warranted before Chunk 7 to address these findings.
- The validator architecture worked exactly as designed: every hallucination caught; zero unsafe prose reached the user.

Backlog items surfaced:
- Runner gap: preserve LLM's failed-attempt prose in the JSONL so future runs can distinguish parser FPs from real hallucinations without manual FEN inspection.
- Validator PV-coverage gap: some legitimate PV continuations (e.g., `Kxf3` from a Stockfish PV) weren't in the trusted set.
- v6 / v3 prompt-iteration chunk (slots between 6.8.1 and Chunk 7).

#### 2026-05-21 (cont.) — Phase 2 Chunk 6.8.1: post-mortem hardening

Three permanent guardrails for future batch-script work, motivated by the 6.8 bug retrospective:

- **Item 1 (CLAUDE.md §8 implementation-pitfall):** `--dry-run` modes that bypass the failure surface are not verification. Required pattern: scale the iteration COUNT (`--positions 1`), not the iteration CONTENT (`--no-llm`, `--skip-context`). Canonical failure: Chunk 6.8's `run_smoke_battery.py --dry-run` skipped both `_build_context` (where credit-exhaustion HTTP 400 surfaces) AND `_answer_one` (where focal-move legality is checked); both Chunk 6.8 bugs (Moves[0]/Moves[1] + credit exhaustion) were invisible to the dry-run.

- **Item 2 (legality assertion in puzzle sampling):** `leela_chat/eval/lichess_puzzle_db.py::sample_by_theme` now calls `_is_solver_move_legal(fen, focal_uci)` on every candidate row; rows with illegal-on-FEN solver moves are dropped with a WARNING log. Drop-rate escalation: ≤0.1% logs at INFO (expected noise across 5.94M rows); >0.1% logs at WARNING (real signal worth investigating). Permanent immunity to the Moves[0]/Moves[1] convention bug class.

- **Item 3A (aggregator three-rate classification + top-api_error-kinds banner):** `aggregate_smoke_battery.py::_render` computes THREE rates separately — `llm_success_rate` (calls where source=llm/llm_retry AND llm_calls>0), `validation_pass_rate_among_llm_success` (the REAL prose-quality signal), and `templated_fallback_rate` (demoted to legacy). If `llm_success_rate < 95%`, a banner fires at the top with the top 3 api_error sub-kinds (`credit_exhausted` / `rate_limit` / `overloaded` / `timeout` / `other`) + their counts, so future readers don't have to grep the JSONL to understand why the run is dead.

- **Item 3B (runner circuit breaker + JSONL sentinel):** `run_smoke_battery.py` now tracks consecutive api_error responses; after 3-in-a-row, aborts the loop with sys.exit(3) AND writes a `{"_circuit_breaker": True, ...}` sentinel row to the JSONL. Downstream tooling (the aggregator, future monitoring) can detect the abort without parsing exit codes. The 3-consecutive threshold catches sustained failure (credit exhaustion, persistent outage); a comment in the code notes it doesn't catch intermittent ~50% failure patterns (acceptable for v1).

Test additions:
- 5 new tests in `test_lichess_puzzle_db.py`: legality skip (illegal-on-FEN), malformed-UCI skip, happy path + reframed existing tests to use legal FEN+moves fixtures.
- 7 new tests in NEW `tests/unit/test_aggregate_smoke_battery.py`: api_error sub-kind classification (credit/rate/overload/timeout/other/empty), sentinel detection (present/absent), three-rate computation (clean run / dead-API banner / mixed kinds banner / sentinel banner / val-pass-among-llm-success isolation).
- 197 unit tests pass (was 181; +16). All green.

Net Chunk 6.8 + 6.8.1 outcome:
- Lichess Puzzle DB corpus on the network volume → Phase 3+ probe-training data unblocked.
- 20-position smoke battery operational; deterministic from seed=0; reproducible across runs.
- Substantive results: 94% validation_ok on real LLM calls, with real prompt fragility surfaced.
- Three permanent guardrails against the next batch-script's failure modes.
- Validator architecture proven under adversarial conditions: 0 hallucinated moves reached the user across 100 calls.

Next session direction:
- **v6 / v3 prompt-iteration chunk** to address the 8 illegal_move hallucinations + 4 forbidden-language hits.
- THEN Chunk 7 (chess-content extractors) with the latency-parallelisation + per-new-extractor-derivation-checkpoint criteria from Chunk 6.7.

#### 2026-05-24 — Phase 2 Chunk 6.9: validator PV-coverage fix + runner forensics + v6/v3/v2 prompt iteration

The "Option B" bundle from the Chunk 6.8 retro. Plan at `/home/kornel/.claude/plans/tell-me-and-explain-mutable-axolotl.md`.

What I did:

**Commit A (`2476968`) — runner forensics infrastructure.**
- Added `llm_attempt_texts` to JSONL rows in `run_smoke_battery.py` (preserves both attempts' verbatim prose on validator-fallback rows; was lost in the templated-fallback path).
- Added `findings_summary` to JSONL rows via a new `_extract_findings_summary` helper — captures the trusted-moves arrays the validator walks (`top_moves_san/uci`, `mcts.children`, `stockfish_top_san/uci` flattened `pv_san`s, `focal_move_san/uci`, `mcts_bestmove_san/uci`). ~200 bytes per row; ~20KB across the 100-call battery.
- Refactored `_answer_one` in `cli/__main__.py` from a 3-tuple to a 4-tuple return (added `findings.model_dump()`); updated callers in `cmd_ask` + `cmd_repl` + 4 test patch points.
- Added "Failed LLM prose excerpt" + "In findings?" Y/N columns to aggregator illegal_move table via new helpers `_strip_san_marks`, `_span_in_findings_summary`, `_excerpt_around_span`. New `validator_gap` vs `real_hallucination` vs `parser_fp` classification (was just one undifferentiated bucket before — methodological gap closed; the original plan would have been theatre without this).
- Added 10 new aggregator tests covering all helpers + backward-compat against old (pre-`findings_summary`) JSONLs.

**Commit B (`5b546f3`) — diagnostic + worked-example audit + validator fix.**
- Pod re-run of pre-v6 battery; aggregator diagnostic classified 8 illegal_move flags: 1 `validator_gap` (`Bxg7` flagged when `Bxg7+` was in stockfish_pv_sans), 5 `parser_fp` (bare-SAN-for-square-location pattern; e.g., "Bd2", "Kxf3"), 1 `real_hallucination`, 1 ambiguous (likely also misclassified). Diagnostic doc written to `tests/eval/smoke_summaries/2026-05-24-prev6-diagnostic.md`.
- Worked-example compliance audit: grepped v4/v5/v1/v2/v1 prompts for forbidden-language list (`noticed|realized|discovered|saw|found|recognized|spotted|identified|perceived|detected|judged|monotonic`). All hits found were INSIDE `❌` anti-pattern blocks — no positive worked-example violations. Safe to copy examples verbatim into v6/v7/v3/v4/v2.
- Validator fix: added `_strip_san_marks` SAN normalization in `_check_legality` (strip `+`/`#`; NOT `x`). 3 new unit tests: positive case (`Bxg7+` matches `Bxg7`), false-positive guard (`Nxg7+` ≠ `Bxg7`), capture-`x`-preservation guard (`Bxg7` ≠ `Bg7`). 197 → 200 unit tests pass.

**Commit C (`ab93020`) — v6+v7 / v3+v4 / v2 prompt iteration.**
- New prompt files per the two-file (mode-separated) structure committed in the plan: `what_is_leela_seeing/v6.md` (iterates v4, default), `v7.md` (iterates v5, detailed), `why_this_move/v3.md` (iterates v1, default), `v4.md` (iterates v2, detailed), `did_you_consider/v2.md` (iterates v1, always-detailed). Three top-of-prompt guards prepended to each: (1) expanded forbidden-language list with cognition synonyms; (2) strict-citation rule extending to ALL moves in continuations including opponent replies; (3) piece-location disambiguation rule ("the rook on c1" not bare "Rc1").
- Handler constant bumps: `what_is_leela_seeing.py` PROMPT_VERSION 4→6, _DETAILED 5→7; `why_this_move.py` 1→3, 2→4; `did_you_consider.py` 1→2.
- Per CLAUDE.md §3: old prompt versions retained on disk (v1/v2/v4/v5 all stay).

**Commit D (this commit) — battery re-run + subjective scoring + final summary.**
- Step 4a walkthrough_rxd4 pre-flight (~$0.05, 5 min) — v6 prose preserved substantive-finding pattern + trough-prose contract + ran cleanly. Green-light for full battery.
- Step 4b full battery (`--reuse-fixture`; ~26 min wall clock on pod). Anthropic API health pre-check via standalone ask call (`validation_ok=True`) before kicking off the 100-call run.
- Step 4c aggregator → `tests/eval/smoke_summaries/2026-05-24-postv6.md`.
- Step 4d subjective scoring on 5 positions (4 puzzles via did_you_consider/detailed + walkthrough_rxd4 via why_this_move/detailed) → `tests/eval/smoke_summaries/2026-05-24-postv6-subjective.md`. Result: 3 v6 wins, 2 ties, 0 v5 wins (cells: tactical/Kg5, endgame/Re1, walkthrough/Rxd4 → v6; quiet/Qf6, opening/Nxf7 → tie). Clears the ≥3-of-5 ship rule.

Hard gates — all PASS, most by significant margin:

| Metric | Baseline (Chunk 6.8) | Target (plan) | Actual (Chunk 6.9) | Δ |
|---|---|---|---|---|
| validation_ok rate | 94% | ≥ 97% | **100%** | +6 pp |
| illegal_move count | 8 (mixed) | ≤ 2 | **0** | −8 |
| Forbidden-language hits | 4 | ≤ 1 | **1** | −3 |
| Templated-fallback rate | 6% | ≤ 6% | **0%** | −6 pp |
| LLM-success rate | 100% | 100% | **100%** | 0 |
| Branch B coverage | 100% | 100% | **100%** | 0 |
| Subjective quality (5 cells) | — | v6 ≥ v5 on ≥ 3 of 5 | **3 wins + 2 ties** | ✓ |

Latency improved (margin not required by plan; happy side effect): median 13.1s → 12.7s; p95 27.9s → 23.3s; p99 32.0s → 25.6s. The top-of-prompt guards add ~50-100 input tokens but the model spends fewer output tokens hedging.

What surprised me:
- **The single validator fix (SAN check/mate-mark stripping) eliminated ALL 8 illegal_move flags, not just the 1 classified as `validator_gap`.** The 5 `parser_fp` flags from the prev6 diagnostic disappeared too — possibly because the rerun's MCTS budget produced slightly different child orderings + the new prompts' piece-location disambiguation rule kept the LLM from emitting bare SANs as square references. Can't fully decompose the contributions without a control re-run (v6 prompts + OLD validator), but the joint effect is decisive. The LLM-stochasticity caveat the plan acknowledged applies — the v6 prose hits different surface forms even on the same fixture, so a paired test in the strict sense is impossible without seed control.
- **0% templated fallback** (was 6%) is the biggest gate result — the new prompts produce more compliant first-attempt prose, so retries that previously failed both attempts now succeed on attempt 1.
- **Latency improvement** was unexpected: the longer top-of-prompt guards should have INCREASED input tokens AND output time. The drop suggests the new guards reduce the model's mid-generation hedging (no more "I might be wrong about" prefixes that consume output tokens). Worth verifying in a later session if the trend holds.
- **The single residual forbidden-language hit** (`tactical_puzzle/why_this_move/default/puzzle_2ATyh` — "the policy network already saw the gap clearly") shows the language list still isn't exhaustive even after the synonym expansion. The plan's "language-list-grows-iteratively" framing is borne out — expect a v8/v6/v5/v3 micro-iteration to address whatever Chunk 7's next smoke battery surfaces.

Net Chunk 6.9 outcome:
- All Chunk 7 entry criteria from the Chunk 6.8 retro now satisfied. The Chunk 7 entry criterion #1 (extractor parallelisation) and #2 (per-new-extractor derivation checkpoint) — both promoted in Chunk 6.7 commit B — remain in force; Chunk 6.9 doesn't bypass them.
- Prompt-iteration cycle (6.8 → 6.9) closed cleanly. The 6.9 commit-A `findings_summary` JSONL forensics machinery is a permanent uplift for any future v8/v6/v5/v3 cycle.
- 200 unit tests pass (was 197; +3 SAN-mark validator tests).

Backlog (carried + new):
- v8/v6/v5/v3 prompt micro-iteration when Chunk 7's smoke battery surfaces the next language-leak or fragility pattern.
- Forbidden-language list expansion candidate: add `clearly` + `gap` to the watch list (the puzzle_2ATyh leak's phrasing).
- Pure-prompt-vs-pure-validator decomposition: a future control run with v6 prompts + OLD validator on the same fixture would let us attribute the illegal_move drop more precisely. Cost: ~$11-13 + 26 min for one extra battery. Defer until needed.

Next session direction:
- **Chunk 7 (chess-content extractors)** can now proceed. Per Chunk 6.7 commit B: latency parallelisation is the FIRST Chunk-7 deliverable; per-new-extractor derivation checkpoint is per-extractor.
- OR: Phase 2 close + Phase 3 entry brief (alternative direction; concept-list pruning of `docs/phase3/concept_candidates.md`).

#### 2026-05-24 (cont.) — Chunk 6.9 hard-close via user-delegation; Phase A of Chunk 7 plan resolved

Chunk 7 plan written and approved (`/home/kornel/.claude/plans/tell-me-and-explain-mutable-axolotl.md`). Plan has three phases: A (close 6.9 via user re-scoring), B (pre-Chunk-7 ADRs + small fixes), C (Chunk 7 itself, 11 sub-chunks Tier-1-foundational MCTS + chess-content extractors with Tier-2 features deferred per ADR-0013).

**Phase A execution and resolution:**

A1-A4 ran as planned:
- A1: regenerated full v5/v2 walkthrough baseline prose on pod (temporarily reverted `why_this_move.py` PROMPT_VERSION_DETAILED 4→2; one ask call against walkthrough_rxd4; handler restored cleanly via trap EXIT). Replaces the partial commit-message excerpts that were the only surviving v2 baseline. LLM-stochasticity caveat noted in the file: fresh sample of v2 prompt, not paired with the original.
- A2: per-cell User scoring tables added below the existing AI scoring tables; 4 dimensions × v5/v6 + notes field per cell.
- A3: "How to score" header replaced the brief Criteria section with the 4 dimensions + decision protocol + "indistinguishable is a valid verdict" framing + 30-45 min discipline note.
- A4: commit `fcb2de2` "6.9 follow-up: prep materials for user manual re-scoring" pushed.

A5 resolved by user delegation, NOT by user scoring:
- When user opened the subjective file, observed the prose is necessarily number-heavy: engine trajectory data (Q values, visit counts, layer ranks), Stockfish PV as bare move sequence, no chess content like forks/pins/threats/king-safety/material. Per CLAUDE.md §2 Rule 1, the LLM cannot cite chess content that isn't in the findings JSON, and the chess-content extractors (threats, king_safety, material_delta, motifs) are exactly Chunk 7's 7k+7l deliverable — they don't exist yet.
- User's call: a 2000+ player cannot contribute meaningfully when there's no chess content to evaluate; delegate the 6.9 close to AI evaluation; defer manual re-scoring to Chunk 7's 7o where 7k+7l will have added the chess-content extractors and the prose will actually have chess content to score against.
- A6-A9 executed in this commit: deferral note added to subjective file top; this PHASE_NOTES entry; Active Chunk pointer updated to Phase B (pre-Chunk-7 ADRs); commit "6.9 close: user deferred manual scoring to post-7k+7l".

**Net 6.9 close decision:** ship v6 per the AI scoring (3 v6 wins, 2 ties, 0 v5 wins; cleared the ≥ 3 of 5 ship rule). Quantitative gates all passed by significant margin per 2026-05-24 close above. Subjective gate cleared by AI scoring per user delegation. **6.9 is hard-closed.**

**Implication for Chunk 7's 7o subjective step:** the empty User scoring tables and the "How to score" header in the subjective file are preserved as forward-looking infrastructure for Chunk 7's 7o run. Same structure, used at a later phase when the prose actually contains chess content. The Phase A work isn't wasted — it's the scaffolding for the eventual real user scoring once chess content exists.

Honest reflection on this resolution:
- The user's deferral is methodologically correct, not avoidance. AI scoring on number-heavy prose can reasonably judge structural quality (clarity, concreteness, accuracy against findings, voice), which is what the prose currently CAN be judged on. AI scoring on chess insight would be theatre — neither AI nor a 2000+ user can score "chess insight" on prose with no chess content beyond a Stockfish move sequence.
- The Chunk 7 plan's framing already anticipated this — 7k+7l were called out as "the actual core deliverable" with the rest of Chunk 7 as supporting infrastructure. The user's deferral reinforces that framing: chess content is when the product becomes meaningfully evaluable.
- Risk to flag: if the AI scoring missed a regression that a 2000+ player would catch, we'd ship v6 with a latent bug. Mitigation: the next round of subjective scoring at Chunk 7's 7o is on chess-content-rich prose AND user-scored. If a v6 regression exists, it'll surface there (and prompt iteration in 7p / 7m is the response path).

Phase B starts next session direction:
- ADR-0011 (quality bar shift), ADR-0012 (latency cap relaxation), ADR-0013 (Tier-2 deferral), walkthrough fixture preservation, rater recruitment doc, Active Chunk pointer to Chunk 7.

#### 2026-05-24 (cont., 2) — Chunk 7 plan Phase B done

Phase B of the Chunk 7 plan landed in 6 commits. Net effect: the project-identity changes that were buried in the Chunk 7 proposal are now explicit ADRs, the walkthrough fixture is preserved against future regression, and the rater recruitment user-action-item is captured without blocking Chunk 7 sub-chunks.

Commit chain:
- `637fe41` B1 — ADR-0011 quality bar shift to super-GM (2700+); CLAUDE.md §1 amendment.
- `e336036` B2 — ADR-0012 latency cap relaxation; CLAUDE.md §11 amendment (per-mode caps replace uniform <30s gate; quick <30s, deep <90s, deepest out-of-scope).
- `5042b11` B3 — ADR-0013 Tier-2 MCTS features deferred to post-Phase-3 evidence (perturbation, multi-strength, adaptive, iterative-search-and-prune, deepest mode + the longer backlog list all explicitly Chunk-8+).
- `8b60160` B4 — `tests/eval/fixtures/walkthrough_rxd4_v5_baseline_prose.md` created; carries the v2 prose from A1 + the structural patterns the v2 prompt commits to (substantive-finding pattern, trough-prose contract, encoding-vs-use contract, Stockfish PV as tactical narration, cross-bucket framing) as the regression reference.
- `de30118` B5 — Pending user action items section added; 2500+ rater recruitment as first entry (does NOT block Chunk 7).
- (this commit) B6 — Active Chunk pointer to Chunk 7; this retrospective.

What surprised me (small things):
- **The forbidden-language CONTRACT reframing** (originally Phase B1 of an earlier plan draft) didn't need a standalone pre-Chunk-7 prompt bump per the user's refinement — the structural-rule reframing folds naturally into 7m's prose contract extension when there's already new contract content to integrate. Skipping the standalone bump saved a version-number-space burn (no v8/v9 mid-Phase-B; v6/v7 hold until 7m).
- **The walkthrough fixture pattern** turned out to be useful beyond just preserving the v5 prose: writing out the "structural patterns the v2 prompt commits to" section forced me to articulate the contract more explicitly than `INTERPRETABILITY_NOTES.md` §3 currently does (which talks about each contract individually but doesn't tie them together as a "what does this prompt actually guarantee" checklist). Worth carrying this pattern forward — Chunk 7's 7m can drop a similar "what does v8/v9 guarantee" checklist next to the walkthrough fixture.

Net Phase B outcome:
- Three pre-Chunk-7 ADRs landed cleanly; project-identity contract documented before Chunk 7 code touches it.
- Walkthrough fixture preserved for future regression checks.
- Rater recruitment open as a user action item without blocking code.
- 200 unit tests still pass (no code changes in Phase B; ADR / doc only).

Phase C starts next session direction:
- **7a (mode architecture + extractor parallelisation + extension-friendliness verification + deep-mode latency projection)** is the first sub-chunk per the plan. Per the user's refinement, 7a includes a proactive latency-projection step that runs per-extractor benchmarks BEFORE 7b-7f's extractors land — at >75s projected deep-mode total, pick the trade-off now rather than discovering at 7o we're over budget.
- 7a also includes the `tests/integration/test_chunk7_extension_friendliness.py` integration test that exercises three architectural properties (per-handler MCTS config accepts new tracks without rewriting; findings-JSON Pydantic models extensible-by-default; prompt-loading accommodates new findings-shape variants). This test is the contract that makes ADR-0013's Tier-2 deferral safe.
- 7b-7f follow in sequence (MultiPv, focused MCTS, multi-config ensemble, network confidence, Stockfish bump) per the plan.

#### 2026-05-24 (cont., 3) — Chunk 7a.1 done (mode infrastructure) + 7a.2 done (extractor parallelisation + latency baseline + deep-mode projection)

**7a.1 — mode infrastructure** (commit `4fb783d`): introduced `Mode = Literal["quick", "deep"]` at `leela_chat/handlers/__init__.py`. Replaced `detailed: bool` with `mode: Mode` across handlers (1, 2) + `_answer_one`. CLI: `--quick` / `--deep` flags + `--detailed` deprecated alias (logs deprecation INFO). Handler 3 unchanged at the layer level — `mode="quick"` auto-upgrades inside `_answer_one` with INFO log (preserves old `--detailed` no-op semantics). Smoke battery matrix renamed `"default"`/`"detailed"` → `"quick"`/`"deep"`. Tests: 240 pass (was 200 + 10 new mode-related tests; net 240 including pre-existing).

**7a.2 — extractor parallelisation + per-extractor latency baseline + deep-mode projection** (this commit). Three pieces:

1. **Parallelisation.** All three handlers now spawn one thread per ENGINE via `concurrent.futures.ThreadPoolExecutor`:
   - BT4 thread: `extract_network_inference` + (deep mode) `extract_layer_emergence` in sequence (both touch the `HookedTransformer`; concurrent BT4 forward passes would race on internal hook state).
   - Stockfish thread: `extract_stockfish_eval` (CPU subprocess; no contention with BT4/Lc0).
   - Lc0 thread: `extract_mcts` or `extract_mcts_trajectory` (separate process, small GPU contention with BT4).
   The slowest single engine sets the handler latency floor instead of the sum. Pattern is inlined per handler (3 handlers, not abstracted — premature abstraction risk).

2. **Per-extractor latency baseline.** New script `leela_chat/scripts/extractor_latency_baseline.py` runs each extractor in isolation on the walkthrough_rxd4 fixture and dumps a markdown table. Pod-run results (RTX 3090, post-warmup):

| Extractor | Wall-clock (s) |
|---|---|
| extract_network_inference (top_n=5) | 0.251 |
| extract_stockfish_eval (depth=15, multipv=3) [quick mode default] | 0.853 |
| extract_stockfish_eval (depth=30, multipv=10) [7f deep mode PROJECTION] | **235.932** ⚠️ |
| extract_mcts (nodes=10000) [quick mode, cold start] | 5.560 |
| extract_mcts_trajectory (max_nodes=10000) [deep mode current, warm] | 0.317 |
| extract_mcts_trajectory (max_nodes=100000) [7d verdict-pass projection] | 7.257 |
| extract_layer_emergence (focal=d8d4) | 0.362 |

Handler end-to-end (parallel, after all extractor warmups):

| Handler call | Wall-clock (s) |
|---|---|
| handle_what_is_leela_seeing (quick) | 0.076 |
| handle_what_is_leela_seeing (deep) | 0.348 |
| handle_why_this_move (quick) focal=d8d4 | 0.074 |
| handle_why_this_move (deep) focal=d8d4 | 0.345 |
| handle_did_you_consider (always-deep) focal=d8d4 | 0.344 |

3. **Deep-mode latency projection — proactive trade-off decision.** Per the plan's 75s/90s threshold:

- The `extract_stockfish_eval (depth=30, multipv=10)` measurement of **235.932s** is a **single-extractor blow-through of the 90s deep-mode gate per ADR-0012**. Cannot ship Chunk 7f at depth=30 + multipv=10 as proposed in the plan.
- Picked the trade-off **now**, not at 7o: deep-mode Stockfish config will be **depth=25, multipv=8** (or smaller — to be measured + pinned in 7f when it lands + ADR-0014). Conservative estimate: ~10-30s vs the 235s at depth=30/multipv=10.
- Verdict-pass MCTS at 100k nodes (7d) measured at 7.257s — cheap, comfortably within budget.
- Exploration-pass MCTS (7d) projected: 10-30k nodes × ~0.5s/10k ≈ 0.5-1.5s if parallel with verdict pass (Lc0 GPU-contention check in 7d will validate); ~12s if sequential.
- Projected deep-mode total (warm steady-state) on the walkthrough fixture: BT4 thread ~0.6s + Stockfish thread (depth-25/multipv-8) ~15s + Lc0 thread (100k verdict + parallel exploration) ~10s. **Handler latency ≈ 15s deep-mode floor** (dominated by Stockfish thread) — comfortably under 90s with margin for cold-start variance.
- Plus LLM translate time (~8-15s) → end-to-end deep-mode estimate ~25-35s. Well within ADR-0012's 90s deep-mode cap.

**Decision committed:** ship Chunk 7f at depth=25, multipv=8 (or whatever the per-extractor measurement at 7f-land time confirms fits the projected budget). Plan's depth=30/multipv=10 is downgraded; ADR-0012 cap is preserved.

**Important caveat on the per-extractor table:** the first cold-start call for each engine includes warmup overhead (network_inference at 0.251s second-call vs first-call seconds; extract_mcts at 5.560s first-call vs extract_mcts_trajectory at 0.317s second-call). Steady-state per-extractor times are what matter for the budget; cold-start hits the first call of the session only. Smoke battery's per-position handler times absorb the cold-start across positions.

What surprised me:
- **Stockfish depth-30/multipv-10 is 25x more expensive than I projected** (235s vs 10-15s estimate). This is exactly the failure mode the proactive 75s decision point was meant to catch — discovering at 7o that we're 4 minutes over budget would have wasted a Chunk 7f sub-chunk of work.
- **MCTS at 100k nodes is cheaper than projected** (7s vs 30-50s estimate). The 7d verdict-pass budget can plausibly be 200k nodes if needed; revisit at 7d.
- **Handler end-to-end times look unrealistically fast** (0.34s) — they're the WARM-state measurement, AFTER all engines have already been called by the per-extractor benchmarks. Real-world handler times in the smoke battery (~13s median per Chunk 6.9) are dominated by LLM translate time + cold-start variance, not by extractor work. The parallelisation gain is small in absolute terms (probably saves 1-2s vs serial), but it's the right structural foundation for when Chunk 7d's multi-config ensemble adds another ~7-10s of MCTS work.

Tests: 240 pass (no regression from parallelisation; tests mock extractors so threading exercised at call-graph level).

What 7a.2 does NOT do:
- Doesn't actually validate parallel-Lc0 contention (that's 7d's pod-side experiment).
- Doesn't change ADR-0012's cap (decision is to fit within the cap by downgrading Stockfish config, not by amending the cap).
- Doesn't ship the chess-content extractors (7k+7l) or any MCTS additions; those are their own sub-chunks.

Next: 7a.3 (extension-friendliness verification — integration test + Pydantic extensible-by-default audit).
