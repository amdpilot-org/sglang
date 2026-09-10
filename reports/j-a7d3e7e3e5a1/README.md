# Bounded speculative custom-mask padding

## Scope

Upstream issue 37553 reports unbounded `custom_mask` growth in EAGLE and DFlash.
Open PR 37120 bounds the returned mask size, but its growth path replaces the
existing mask with an all-`True` tensor. This change preserves the existing mask
prefix while allocating only the current required size and returning an exact
slice on shrink.

## GPU evidence

- Installed source: `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`
- Installed import: `/sgl-workspace/sglang/python/sglang/__init__.py`
- Candidate commit: `9d9c9ef670d37c593b64dc56ee6d2f7b3bf32de1`
- GPU: one AMD Instinct MI300X, `gfx942`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- Triton: `3.7.0`

The installed baseline used batch sizes `[1, 8, 1, 16, 1, 32, 1]`,
`draft_token_num=8`, and `paged_kernel_len=10`. After each growth step, a later
batch of size 1 received the stale peak-sized mask instead of the independently
derived `144` elements. KV indices, cumulative KV lengths, and query offsets
matched their independent references in every case.

The preserved PR candidate passed its four size-only tests, but an adversarial
64-element mask with alternating `True`/`False` values was replaced by an
all-`True` mask on the first growth call. For example, EAGLE batch 1 expected
112 `True` values and received 144; DFlash showed the same mismatch.

## Reproduction

```bash
export PYTHONPATH="$PWD/python"
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-a7d3e7e3e5a1/triton
/opt/venv/bin/python -m pytest -q test/registered/unit/spec/test_custom_mask_padding.py
```

The new test compares the full mask, KV indices, cumulative KV lengths, and
query offsets against independent references across the finite alternating
batch matrix.

## Boundaries

- No full model weights were downloaded or run.
- No environment replacement or node-wide state change was made.
- The container does not expose its local image ID; the operator-specified
  image ID was recorded without treating the hostname as image identity.
- Validation is unit-level on one MI300X, not an end-to-end serving run.
