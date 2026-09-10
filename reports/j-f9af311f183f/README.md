# MI300X causal-conv state-rotation investigation

## Scope

This report records a bounded gfx942 investigation of multi-step chunked
`causal_conv1d_update` dispatch versus one-token recurrence, including repeated
wraparound of the circular convolution state. It intentionally does not
duplicate the already-working upstream fix in sgl-project/sglang PR 38623.

## Environment

- GPU: one AMD Instinct MI300X, `gfx942`
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Image ID: `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- Interpreter: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- ROCm/HIP: `7.2.26015-fc0010cf6a`
- Installed source: `/sgl-workspace/sglang`, commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`
- Installed native module: `/opt/venv/lib/python3.10/site-packages/sgl_kernel`
- Mirror base: `/job/sglang`, commit `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Tested upstream candidate: `dac135a1957f72122b492c31655d808e124c3de8`

## Reproduction

1. Run the installed-source baseline:

   ```bash
   /opt/venv/bin/python - <<'PY'
   import importlib.util, torch
   p = "/sgl-workspace/sglang/test/registered/layers/mamba/test_causal_conv1d.py"
   s = importlib.util.spec_from_file_location("baseline_causal_conv1d", p)
   m = importlib.util.module_from_spec(s)
   s.loader.exec_module(m)
   m.test_causal_conv1d_update(
       dim=2048,
       width=4,
       seqlen=1,
       has_bias=True,
       silu_activation=True,
       itype=torch.bfloat16,
   )
   PY
   ```

2. Run the adversarial rotation harness against a checkout:

   ```bash
   TRITON_CACHE_DIR=/tmp/sglang-cache-j-f9af311f183f \
   SGLANG_SOURCE=/path/to/checkout/python \
   REPRO_OUTPUT=/tmp/results.json \
   /opt/venv/bin/python reports/j-f9af311f183f/repro_causal_conv.py
   ```

The harness uses an independently derived `F.conv1d` ring-buffer reference. It
compares:

- chunked output against the reference,
- one-token recurrent output against the reference,
- chunked output against recurrent output,
- final chunked state against the reference,
- final recurrent state against the reference.

It covers widths 2-4, state lengths 4 and 8, cursors near both ends of the ring,
chunk lengths 2-5, repeated rotations, and both `float32` and `bfloat16`.

## Numerical gates

The existing operation contract and tolerances are preserved:

- `float32`: `rtol=3e-4`, `atol=1e-3`
- `bfloat16`: `rtol=1e-2`, `atol=5e-2`
- state storage: exact equality

## Results

### Installed-source baseline

The existing width-4, one-token update test passed in 6.053329 seconds. This
baseline is recorded in `baseline-first.json` and is not proof for later
checkout changes.

### Mirror base

The mirror base passed chunked-versus-recurrent equivalence, but both forms
failed the independent circular reference in 7 of 8 output cases and all 8
state cases. This demonstrates that the fallback ignores `cache_seqlens` and
updates the state as a linear buffer.

Raw results are in `repro-mirror-results.json`.

### Upstream candidate

The exact upstream candidate commit
`dac135a1957f72122b492c31655d808e124c3de8` passed all 8 adversarial cases for
chunked output, recurrent output, chunked-versus-recurrent equivalence, and
final state. Its two focused GPU regressions also passed:

```text
2 passed, 3 warnings in 11.36s
```

Raw results are in `repro-candidate-results.json`.

### Unsupported width boundary

The mirror base silently accepted width 5 and produced a wrong result:

```text
accepted max_abs_diff 3.291905403137207 allclose False
```

The upstream candidate rejected it explicitly:

```text
ValueError: causal_conv1d only supports width between 2 and 4, got 5
```

## Conclusion

No mirror code change is made because upstream PR 38623 already fixes the
demonstrated mismatch. This report preserves the tested candidate commit and
the raw gfx942 evidence without duplicating that working fix.

## Boundaries and unfinished work

- No full model weights were downloaded.
- No environment replacement or node-wide state modification was performed.
- No artificial GPU burn, unbounded loop, sleep loop, or repeated busy work was used.
- No upstream issue, PR, or comment was posted or modified.
- The investigation is bounded to the relevant Triton fallback and does not claim coverage for unrelated kernels.
