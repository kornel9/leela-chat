# Vendored packages

## `transformer_lens/` — Fudan TransformerLens fork

**Upstream:** https://github.com/JacklE0niden/Leela-SAEs — `TransformerLens/transformer_lens/`

This is the Fudan / OpenMOSS group's fork of [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) (originally by Neel Nanda). They added BT4/Leela-specific components:

- `components/leela_embed.py` — `LeelaEmbed`, `BT4LeelaEmbed`
- `components/leela_encoder.py` — `EncoderLayer`, `MultiHeadAttention`, `SmolGen`, `Lc0LayerNorm`, `LC0MLP`
- `components/leela_features.py` — `LeelaBoard` (FEN/PGN → 112-plane tensor)
- `components/leela_policyhead.py` — `PolicyHead` (1858-element softmax over BT4 move space)
- `components/leela_valuehead.py` — `ValueHead` (WDL softmax)
- `components/leela_mlhhead.py` — `MLHHead` (moves-left scalar)

Plus integration hooks in `HookedTransformer.py`, `HookedTransformerConfig.py`, and `loading_from_pretrained.py` (the `lc0/BT4-1024x15x32h` model entry).

**Snapshot source:** `Leela-SAEs` main branch, `TransformerLens/transformer_lens/` (taken from upstream around 2026-05-05; the upstream repo doesn't expose a release tag, so the date stands in for a SHA).

**License:** MIT (see `transformer_lens/LICENSE` — preserved verbatim from upstream).

**What we did NOT vendor:** the rest of the Leela-SAEs repo (SAE training code, examples beyond the BT4 weight converter, docs, demos, tests, the wider Fudan project chrome). We use only `transformer_lens` and one file from `examples/` (see below).

### Local modifications applied to the vendored copy

Both fixes are direct edits to the vendored source (no separate `.patch` file overlay). The original `scripts/patches/leela_saes_bt4_loader.patch` file in the repo root remains as a portable artifact for upstreaming.

1. **`transformer_lens/HookedTransformer.py`** (around line 247) — passes `self.cfg.eps` as the sixth positional argument to `EncoderLayer(...)`. Without this, the encoder LayerNorms and SmolGen LayerNorms silently use `eps=1e-5` (the `Lc0LayerNorm` default) instead of the `eps=1e-3` specified in the BT4 config. Symptom: subtle numerical drift through all 15 encoder blocks even when weights load correctly.

2. **(applies to the separate vendored `bt4_weight_conversion.py`, not to `transformer_lens/`)** — see that file's header.

The `bt4_weight_conversion.py` fix was diagnosed by comparing intermediate activations against `onnx2torch`.

### How to update

If we ever pull a newer upstream snapshot:

1. Verify our two patches still apply (or re-derive equivalents). Compare against `scripts/patches/leela_saes_bt4_loader.patch`.
2. Rebuild `BT4.pt` (vendored converter writes it).
3. Re-run the 240-position parity test against `lc0` binary.
4. Update this file's "snapshot source" date.

The upstream's API surface is stable for our use (we only call `HookedTransformer.from_pretrained_no_processing` and `LeelaBoard.from_fen` + `lcz_features`), so newer snapshots should slot in cleanly modulo our patches.

### `bt4_weight_conversion.py`

**Upstream:** https://github.com/JacklE0niden/Leela-SAEs — `examples/weight_conversion.py`

Converts a BT4 ONNX file (raw lc0 weights export) into a PyTorch `state_dict` `.pt` that `HookedTransformer.from_pretrained_no_processing` can load. We vendor only this single file from `examples/`.

**Local modification:** added `attention_body.ln.bias` and `attention_body.ln2.bias` to the explicit weight-name mapping. The original code only mapped `.weight` for these two LayerNorms; the corresponding `.bias` parameters exist in the ONNX (magnitudes up to |1.74|) but were never copied into `CleanLC0Model`, so the produced `.pt` shipped with zero LN biases in the attention body. This produces visibly broken WDL on every position (e.g., startpos → W0.01/D0.02/L0.97 instead of W0.19/D0.64/L0.17).
