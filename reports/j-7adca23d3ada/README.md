# Vision Triton window and sink verification on MI300X

## Scope

This report records a bounded, single-GPU investigation of sglang issue 37983.
It deliberately excludes model and image loading. All numerical cases use tiny
synthetic GPU tensors and an independent reference implementation.

The tested upstream candidate is sglang pull request 38142 at head commit
`7284c6da0d08a5bfa0d241d08c8ceb8e8738e55e`. The persistent mirror `main` base
for this report is `0084030179bfba86bfeb6d43f7997d4076329d2c`. Because the
upstream candidate already works and remains open, this delivery does not
duplicate its code changes; it records the MI300X evidence and reproduction
artifacts only.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Local image ID: `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- GPU: one AMD Instinct MI300X, `gfx942`, capability `(9, 4)`, serial `692440004359`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`
- Triton: `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`
- Torch HIP library: `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so`
- AMD HIP library: `/opt/rocm-7.2.0/lib/libamdhip64.so`

The image ID is the operator-provided local identity. No container runtime
tool or socket was available inside the job to inspect it independently.

## Installed-source baseline

Command:

```bash
/opt/venv/bin/python reports/j-7adca23d3ada/installed_baseline.py
```

The installed source was commit
`8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4` with a dirty worktree. Its
`context_attention_fwd` signature had no `window_size` or `sinks` arguments.
The direct window/sink request failed with:

```text
TypeError: context_attention_fwd() got an unexpected keyword argument 'window_size'
```

The meaningful neighboring control was causal full attention on varlen lengths
`[17, 29]`, two heads, head dimension 64, bfloat16. It matched an independent
per-sequence PyTorch SDPA reference with maximum absolute error
`0.011729955673217773` and mean absolute error `0.0007168952724896371`. The
first synchronized GPU execution took `1924.070225097239 ms`; three subsequent
CUDA-event samples took `0.5341529846191406`, `0.06571199744939804`, and
`0.05272100120782852 ms`.

This is an installed-source baseline only and is not proof for later checkout
changes. Raw values are in `baseline-first.json`.

## Mirror main control

Command:

```bash
export PYTHONPATH=/job/sglang/python
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-7adca23d3ada/triton
export SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-7adca23d3ada/sglang-jit
/opt/venv/bin/python reports/j-7adca23d3ada/candidate_verification.py
```

On mirror `main`, `VisionTritonAttention` accepted `window_size` and `s_aux`,
but `context_attention_fwd` silently ignored them. Full attention passed the
independent reference. Local-window, sink-only, and combined window/sink cases
failed:

| Case | Max abs error | Mean abs error | Result |
| --- | ---: | ---: | --- |
| Full attention | 0.004708528518676758 | 0.00040089277899824083 | pass |
| Local window `(5, 7)` | 2.129146099090576 | 0.28670060634613037 | fail |
| Per-head sinks | 0.08786016702651978 | 0.005223601590842009 | fail |
| Window and sinks | 2.0873348712921143 | 0.2600824534893036 | fail |

The comparison gate was `atol=0.02, rtol=0.02`. Raw values are in
`mirror-main-control.json`.

## Upstream candidate result

The exact upstream PR head was checked out as
`candidate/pr-38142-7284c6`. Its existing GPU test suite was run with:

```bash
export PYTHONPATH=/job/sglang/python
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-7adca23d3ada/triton
export SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-7adca23d3ada/sglang-jit
/opt/venv/bin/python -m pytest -q test/registered/attention/test_triton_attention_window_sink.py -x
```

Result:

```text
13 passed, 1 skipped, 3 warnings, 7 subtests passed in 42.21s
```

The skipped test was the optional Hopper FA3 cross-check; its guard requires
NVIDIA sm90 and therefore does not apply to this ROCm `gfx942` run.

An independent float32 reference used an explicit boolean local-window mask and
a virtual per-head sink logit in the softmax denominator. It checked
`VisionTritonAttention` directly on varlen lengths `[37, 53]`, four heads, head
dimension 64, bfloat16:

| Case | Max abs error | Mean abs error | Result |
| --- | ---: | ---: | --- |
| Full attention | 0.004708528518676758 | 0.00040089277899824083 | pass |
| Local window `(5, 7)` | 0.009917736053466797 | 0.0006941012106835842 | pass |
| Per-head sinks | 0.004498720169067383 | 0.00039595639100298285 | pass |
| Window and sinks | 0.006193876266479492 | 0.0006127057131379843 | pass |

The comparison gate remained `atol=0.02, rtol=0.02`. The first synchronized
candidate GPU execution took `1613.0682271905243 ms`; three CUDA-event samples
took `0.13447000086307526`, `0.07373099774122238`, and
`0.06274399906396866 ms`. Raw values are in
`upstream-pr-38142-results.json`.

## Limitations

- This is a tiny-tensor semantic and bounded timing check, not a production
  throughput benchmark; cold timings include Triton JIT and launch overhead.
- Only one assigned MI300X (`gfx942`) GPU was used.
- The optional FA3 cross-check is Hopper-specific and was skipped on ROCm.
- The candidate was tested at its exact upstream head, not rebased onto the
  newer mirror `main`; a future integration may need conflict review.
- The Triton path is JIT-compiled and loads no prebuilt `.so` through
  `context_attention_fwd`. Importing SGLang can load the environment's AITER
  native module, but it is not used by this vision Triton path.
- No model weights or images were downloaded or loaded.
