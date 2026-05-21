#!/usr/bin/env bash
# scripts/setup_pod.sh — idempotent setup for a freshly-restarted RunPod box.
#
# RunPod wipes everything outside the persisted volume (apt packages,
# /workspace/bin, /usr/local/* additions) on stop/start cycles. /workspace
# itself persists (the leela-chat project, the venv, the BT4 weights all
# survive). This script restores everything the project needs that lives
# OUTSIDE the persisted volume — and is safe to re-run.
#
# Run:
#   ssh runpod 'bash /workspace/leela-chat/scripts/setup_pod.sh'
# Or, after first rsync of a new commit:
#   ssh runpod 'cd /workspace/leela-chat && bash scripts/setup_pod.sh'
#
# What it does:
#   1. Installs apt packages (meson, ninja-build, stockfish) — wiped on restart.
#   2. Adds CUDA to PATH so nvcc / cuBLAS link correctly during Lc0 build.
#   3. Builds Lc0 (release/$LC0_VERSION) into /workspace/bin/lc0 if missing —
#      source goes to /workspace/src/lc0 so it persists; only the binary
#      copy step repeats on a wipe.
#   4. Smoke-checks that lc0, stockfish, BT4 ONNX, and the project venv
#      all work.
#
# NOTE: An earlier version of this script applied a runtime patch to a
# /workspace/Leela-SAEs/ checkout (the two BT4-loader fixes). That step is
# obsolete now that we ship the fixed code as a vendored package under
# leela_chat/_vendor/transformer_lens/. The portable patch artifact
# (scripts/patches/leela_saes_bt4_loader.patch) is still kept for filing
# upstream PRs.

set -euo pipefail

LC0_VERSION=v0.32.1
LC0_SRC=/workspace/src/lc0
LC0_BIN=/workspace/bin/lc0
BT4_ONNX=/workspace/weights/BT4-1024x15x32h-swa-6147500/BT4-1024x15x32h-swa-6147500.onnx
BT4_PT=/workspace/weights/BT4-1024x15x32h-swa-6147500/BT4.pt
# The vendored TransformerLens loader hardcodes this path (see
# leela_chat/network/__init__.py:36-77 and the upstream-bit-identical
# leela_chat/_vendor/transformer_lens/loading_from_pretrained.py:1832).
# The symlink target lives OUTSIDE /workspace so it dies on every
# pod restart — re-creating it here is the idempotent fix.
BT4_PT_SYMLINK_DIR=/path/to/models/lc0
BT4_PT_SYMLINK=$BT4_PT_SYMLINK_DIR/BT4.pt
VENV_PY=/workspace/leela-chat/.venv/bin/python

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# 0. Fail-fast guard: network volume must be mounted ------------------------
#
# If /workspace/leela-chat is missing, the network volume isn't attached to
# this pod. RunPod network volumes are datacenter-locked: the volume lives
# in eu-cz-1, and pods in other DCs see an empty /workspace without warning
# (the dashboard's pod-creation flow doesn't filter region by volume
# availability — the "Network Volume" option silently shows nothing when
# the volume is in a different DC). Without this guard, the symptom is
# 20+ minutes of cascading failures (stockfish + Lc0 + venv all missing,
# but BT4.pt-load fails ten minutes later); with the guard, ~5 sec to
# diagnose. Surfaced in PHASE_NOTES 2026-05-21 (Chunk 6.6 session log)
# after three out-of-DC pods spun up before finding the right one.

log "verifying network volume is mounted"
if [ ! -d /workspace/leela-chat ]; then
    log "ERROR: /workspace/leela-chat missing — the network volume is not"
    log "       attached to this pod. RunPod network volumes are DC-locked;"
    log "       this pod is likely in a different datacenter than your volume."
    log ""
    log "       Action: stop this pod and create a new one in the volume's"
    log "       datacenter (eu-cz-1 per the 2026-05-21 session). In the"
    log "       RunPod dashboard, filter pod-creation by DC to eu-cz-1"
    log "       FIRST, then pick a 3090/4090 instance — the dashboard"
    log "       does NOT filter region by volume availability, so picking"
    log "       a pod first and the volume second silently produces this"
    log "       state."
    log ""
    log "       Alternative: pull individual files via the S3-compatible API"
    log "       at https://s3api-eu-cz-1.runpod.io (slower; doesn't recover"
    log "       the venv cleanly across architectures)."
    exit 1
fi
log "  /workspace/leela-chat present"

# 1. apt packages -----------------------------------------------------------

log "checking apt packages"
export DEBIAN_FRONTEND=noninteractive
missing_pkgs=()
for pkg in meson ninja-build stockfish; do
  if ! dpkg -s "$pkg" >/dev/null 2>&1; then
    missing_pkgs+=("$pkg")
  fi
done
if [ ${#missing_pkgs[@]} -gt 0 ]; then
  log "installing: ${missing_pkgs[*]}"
  apt-get update -qq
  apt-get install -y -qq "${missing_pkgs[@]}"
else
  log "  all apt packages present"
fi

# 2. CUDA on PATH -----------------------------------------------------------

if [ -d /usr/local/cuda-12.8 ] && ! command -v nvcc >/dev/null 2>&1; then
  log "adding /usr/local/cuda-12.8/bin to PATH for this shell"
  export PATH=/usr/local/cuda-12.8/bin:$PATH
  if ! grep -q 'cuda-12.8/bin' /root/.bashrc 2>/dev/null; then
    log "  also adding to /root/.bashrc for future shells"
    echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> /root/.bashrc
  fi
fi

# 3. Lc0 build --------------------------------------------------------------

if [ -x "$LC0_BIN" ]; then
  log "lc0 binary present at $LC0_BIN, skipping build"
else
  log "building Lc0 $LC0_VERSION (this takes ~10 min)"
  mkdir -p /workspace/src /workspace/bin
  if [ ! -d "$LC0_SRC/.git" ]; then
    log "  cloning Lc0 into $LC0_SRC"
    rm -rf "$LC0_SRC"
    git clone --branch "$LC0_VERSION" --recurse-submodules --depth 1 \
      https://github.com/LeelaChessZero/lc0.git "$LC0_SRC"
  else
    log "  Lc0 source already at $LC0_SRC"
  fi
  cd "$LC0_SRC"
  ./build.sh -Dgtest=false
  cp build/release/lc0 "$LC0_BIN"
  log "  lc0 built and copied to $LC0_BIN"
fi

# 4. Smoke checks -----------------------------------------------------------

log "verifying lc0"
"$LC0_BIN" --help >/dev/null
log "  lc0 OK"

log "verifying stockfish"
echo quit | stockfish >/dev/null
log "  stockfish OK"

log "verifying BT4 ONNX"
if [ -f "$BT4_ONNX" ]; then
  log "  BT4 ONNX present ($(stat -c %s "$BT4_ONNX") bytes)"
else
  log "  MISSING: $BT4_ONNX"
  log "  Re-download via huggingface_hub if needed (ADR-0001 SHA-256)."
  exit 1
fi

log "ensuring BT4.pt symlink for vendored TransformerLens loader"
if [ ! -f "$BT4_PT" ]; then
  log "  WARN: $BT4_PT missing — run 'python -m leela_chat._vendor.bt4_weight_conversion' first"
else
  mkdir -p "$BT4_PT_SYMLINK_DIR"
  ln -sf "$BT4_PT" "$BT4_PT_SYMLINK"
  log "  symlink ready: $BT4_PT_SYMLINK -> $BT4_PT"
fi

log "verifying project venv"
if [ -x "$VENV_PY" ]; then
  # Confirm both inference paths are importable: HookedTransformer (BT4
  # primary path, via vendored TransformerLens) and lczerolens (legacy /
  # cross-check + Maia experiments).
  "$VENV_PY" -c "
from leela_chat import _vendor  # sys.path hack
from transformer_lens import HookedTransformer
import lczerolens
print(f'  HookedTransformer importable; lczerolens={lczerolens.__version__}')
"
else
  log "  no venv at $VENV_PY — run 'uv sync' in /workspace/leela-chat first"
  exit 1
fi

log "pod setup complete"
