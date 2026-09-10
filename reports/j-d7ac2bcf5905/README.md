# gfx942 mRoPE compiled-path reuse investigation

## Scope

This is a bounded follow-up to amdpilot-org/sglang issue 214. It does not repeat
the original multimodal mRoPE rank trigger. Mirror `main` already contains the
working fix from sgl-project/sglang PR 34446, and open sgl-project/sglang PR
35744 was not copied or duplicated.

The uncovered case tested here is compiled-path reuse for
`MRotaryEmbedding.forward_cuda` across a finite supported token-shape sequence.

## Environment

- GPU: one AMD Instinct MI300X, gfx942, capability `(9, 4)`.
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local
  image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- Interpreter: `/opt/venv/bin/python`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- Triton: `3.7.0`.
- Mirror base commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Tested mirror kernel:
  `/job/sglang/python/sglang/kernels/ops/attention/rotary_triton.py`.
- Native dispatch: Triton JIT kernel
  `_triton_mrope_forward_fused`.
- Installed wrapper used for the sparse-checkout GPU run:
  `/sgl-workspace/sglang/python/sglang/srt/layers/rotary_embedding/mrope.py`.

The persistent mirror checkout was kept sparse to bound clone traffic. The
mirror kernel was loaded directly and injected into the installed
`MRotaryEmbedding` wrapper for execution. The committed test uses the normal
mirror imports in a complete checkout.

## Installed-source baseline

`/job/baseline-first.json` records the preinstalled source baseline. It is not
proof for later checkout changes.

The documented direct fused QK path selected PDL on gfx942 because its guard
only checked capability major `>= 9`. Triton emitted
`griddepcontrol.launch_dependents`, and HSACO linking failed with:

```text
ld.lld: error: target emulation unknown: -m or at least one .o file required
```

The meaningful neighboring control disabled only that unsupported PDL constexpr
and ran the same 1D Triton kernel. For tokens `17`, Q heads `4`, KV heads `2`,
head dimension `64`, rotary dimension `32`, and bfloat16 inputs, the independent
reference errors were:

- Q: `0.0074920654296875`
- K: `0.006683826446533203`
- gate: `0.0`

Input data and surrounding sentinels were unchanged. The first supported GPU
execution took `0.002688815351575613` seconds; four synchronized follow-up
calls took `0.000268423929810524`, `8.215196430683136e-05`,
`6.738118827342987e-05`, and `5.891872569918632e-05` seconds.

## Reuse experiment

The supported sequence used token counts `(1, 17, 33, 64)`, Q heads `3`, KV
heads `1`, head dimension `256`, rotary dimension `64`, interleaved mRoPE
section `[11, 11, 10]`, and bfloat16. Positions, query, and key were slices of
fixed maximum-size buffers, so their base addresses stayed constant while the
token shape changed.

Each case used an independent per-axis `torch.gather` reference. The test also
checked:

- in-place aliasing: returned query and key are the input tensors;
- dtype: query, key, and cache remain bfloat16;
- static graph address: positions/query/key pointers stay fixed across shapes;
- sentinel-protected storage: guards and unused suffixes remain unchanged;
- unsupported rank: three-dimensional positions raise `AssertionError`.

The committed test passed:

```text
2 passed, 1 warning in 4.68s
```

The warning was pytest's existing assertion-rewrite notice, not a numerical or
GPU failure.

## Timing matrix

Timing used `time.perf_counter` around one synchronized `forward_cuda` call in
a fresh job-private Triton cache. This is a bounded observation, not a CI
performance gate.

| pass | tokens | elapsed seconds | Triton specializations |
|---|---:|---:|---:|
| first sequence | 1 | 1.2944606766104698 | 1 |
| first sequence | 17 | 0.0001596701331436634 | 1 |
| first sequence | 33 | 8.115032687783241e-05 | 1 |
| first sequence | 64 | 7.707905024290085e-05 | 1 |
| warm sequence | 1 | 7.396191358566284e-05 | 1 |
| warm sequence | 17 | 7.075164467096329e-05 | 1 |
| warm sequence | 33 | 7.321778684854507e-05 | 1 |
| warm sequence | 64 | 6.703287363052368e-05 | 1 |

The single cold call includes Triton compilation. Every later shape and the
entire warm pass reuse the same one specialization.

## Reproduction

In a complete mirror checkout:

```bash
cd /job/sglang
export PYTHONPATH=/job/sglang/python
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-d7ac2bcf5905/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-cache-j-d7ac2bcf5905/inductor
export TMPDIR=/tmp/sglang-cache-j-d7ac2bcf5905/tmp
/opt/venv/bin/python -m pytest test/registered/rotary/test_mrope_triton_reuse.py -vv --no-header
```

No model weights were downloaded, no toolchain was replaced, no node-wide state
was modified, and no upstream issue, PR, or comment was posted or changed.

## Limitations

- The direct fused QK kernel's PDL variant is unsupported by this gfx942
  toolchain and was not forced through the committed production test.
- Timing is single-run wall-clock data on one assigned MI300X; it is not a
  steady-state benchmark or a performance assertion.
- The sparse-checkout validation injected the mirror kernel into the installed
  wrapper. A complete checkout remains the authoritative execution for CI.
