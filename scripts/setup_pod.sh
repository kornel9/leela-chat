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
#   4. Applies in-house BT4-loader fixes to Fudan's Leela-SAEs checkout
#      (patches/leela_saes_bt4_loader.patch) if present and not yet applied.
#   5. Smoke-checks that lc0, stockfish, BT4 ONNX, and the project venv
#      all work.

set -euo pipefail

LC0_VERSION=v0.32.1
LC0_SRC=/workspace/src/lc0
LC0_BIN=/workspace/bin/lc0
BT4_ONNX=/workspace/weights/BT4-1024x15x32h-swa-6147500/BT4-1024x15x32h-swa-6147500.onnx
VENV_PY=/workspace/leela-chat/.venv/bin/python
SAES_DIR=/workspace/Leela-SAEs
SAES_PATCH=/workspace/leela-chat/scripts/patches/leela_saes_bt4_loader.patch
# Sentinel line introduced by the patch — we test for it to know if the
# patch is already applied (avoids re-applying and erroring out).
SAES_PATCH_SENTINEL='"attention_body.ln.bias": "attn_body/ln.bias"'

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

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

# 4. Apply Fudan Leela-SAEs BT4-loader patch --------------------------------
# Two upstream bugs (see scripts/patches/leela_saes_bt4_loader.patch) cause
# wrong WDL on every position. Without these fixes the HookedTransformer
# load path is unusable. If Leela-SAEs isn't on this pod (yet), skip silently.

if [ -d "$SAES_DIR" ] && [ -f "$SAES_PATCH" ]; then
  if grep -q "$SAES_PATCH_SENTINEL" "$SAES_DIR/examples/weight_conversion.py" 2>/dev/null; then
    log "Leela-SAEs BT4-loader patch already applied"
  else
    log "applying Leela-SAEs BT4-loader patch"
    (cd "$SAES_DIR" && patch -p0 < "$SAES_PATCH")
    log "  patched. Remember to rebuild BT4.pt via examples/weight_conversion.py."
  fi
fi

# 5. Smoke checks -----------------------------------------------------------

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

log "verifying project venv + lczerolens"
if [ -x "$VENV_PY" ]; then
  "$VENV_PY" -c "import lczerolens; print(f'  lczerolens={lczerolens.__version__}')"
else
  log "  no venv at $VENV_PY — run 'uv sync' in /workspace/leela-chat first"
  exit 1
fi

log "pod setup complete"
