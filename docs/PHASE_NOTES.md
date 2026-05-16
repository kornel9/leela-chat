# Phase Notes — Leela Chat

> A living log of what's happening in the current phase, what surprised you, and what to remember when the next phase starts. Update at the start and end of every work session.
>
> **What goes here:** current chunk, today's plan, things you learned today, anything that surprised you, bugs that took hours to find, questions you have, things to revisit. Anything actionable or instructive at the chunk-to-phase scale.
>
> **What does NOT go here:** strategic vision (→ BIBLE.md), architectural decisions (→ DECISIONS.md), code conventions (→ CLAUDE.md).

---

## Current Status

- **Phase:** 0 — Environment and Foundations
- **Active chunk:** Chunk 1 — lczerolens smoke test runs on this machine
- **Last session:** 2026-05-16
- **Next session goal:** Complete Chunk 1 smoke test (`lczerolens/maia-1100` load + forward pass + policy shape) and commit `notebooks/00_smoke_test.ipynb`.

---

## Phase 0 — Environment and Foundations

### Exit criteria (from BIBLE.md / CLAUDE.md §10)
- [x] lczerolens walkthrough notebook runs end-to-end on this machine — `notebooks/00_smoke_test.ipynb`, 2026-05-16
- [ ] Lc0 compiled with CUDA backend, `--verbose-move-stats` produces parseable output — **deferred to follow-on task (no Linux binary in any recent Lc0 release; ~45 min build from source). Not blocking Phase 1 because lczerolens + ONNX BT4 already works for forward passes.**
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

## Open questions for the project

_Things I notice but don't have time to answer right now. Capture them so they don't get lost._

- (none yet)

---

## Things to revisit

_Decisions or shortcuts I took that I should reconsider later._

- (none yet)

---

## Phase retrospectives

_At the end of each phase, write a short retrospective: what worked, what didn't, what I'd do differently, what I learned that should change CLAUDE.md or the bible._

### Phase 0 retrospective
_(empty until end of phase)_

