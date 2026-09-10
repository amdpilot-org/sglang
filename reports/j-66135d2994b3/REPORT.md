# Reduced DSpark draft/target handoff and commit-block study

## Scope

This is a bounded, one-GPU reduced-block study on `amdpilot-org/sglang` commit
`0084030179bfba86bfeb6d43f7997d4076329d2c`. It composes existing SGLang DSpark
primitives with locally generated synthetic weights and states. It does not
load GLM weights, instantiate a model-specific GLM/DSpark constructor, use
distributed execution, or make a model-integration claim. No
distributed-performance claim is made.

The model-specific constructor boundary is explicit: the study stops at the
primitive handoffs—target hidden-state selection, greedy accept/finalization,
commit-inject layout, and KV projection. A production constructor would still
need model-specific wiring, a real checkpoint, cache-pool integration, and
end-to-end validation.

Upstream context was read only: SGLang issue `sgl-project/sglang#30734` and the
linked PRs `#31047`, `#31260`, `#31457`, `#31451`, `#32186`, and `#32374`.
No upstream issue, PR, or comment was posted or changed. Current mirror `main`
already contains the relevant DSpark primitives and tests, so this work does not
duplicate or alter an upstream fix.

## Environment

- GPU: one AMD Instinct MI300X, `gfx942`, device ID `0x74a1`, serial
  `692440004359`, GUID `39656`.
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local
  image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- Python: `/opt/venv/bin/python`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`; HIP runtime reports
  `7.2.26015-fc0010cf6a`.
- Checkout source: `/job/sglang/python/sglang/__init__.py`.
- Installed native Python modules:
  `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`,
  `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`, and
  `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`.
- Native libraries:
  `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so`,
  `/opt/rocm/lib/libamdhip64.so.7`, and `/opt/rocm/lib/librccl.so.1`.

## First GPU baseline

The first GPU execution used the preinstalled source, not the later delivery
checkout. It is labeled installed-source evidence only and is not proof for
checkout changes.

- Installed SGLang commit: `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`.
- Installed Python source:
  `/sgl-workspace/sglang/python/sglang/__init__.py`.
- Installed `sgl_kernel`:
  `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`.
- Test:
  `/sgl-workspace/sglang/test/registered/kernels/ops/speculative/test_boundary_kv_fix_kernels.py`.
- Command:
  `/opt/venv/bin/python -m pytest -q /sgl-workspace/sglang/test/registered/kernels/ops/speculative/test_boundary_kv_fix_kernels.py --disable-warnings`.
- Result: `1 passed`, `4 subtests passed`, `4 warnings`, exit code `0`.
- Timing: Python subprocess wall clock `35.798567 s`; pytest reported
  `32.16 s`.
- Reference comparison: the existing test compares boundary hidden-state/KV
  handoff kernels against pure-Torch references for four cases.
- Initial timing-tool error: `/usr/bin/time` was absent
  (`/bin/bash: line 4: /usr/bin/time: No such file or directory`). No GPU work
  ran in that failed attempt; the same pytest command was rerun with Python
  wall-clock timing.

The complete first-baseline artifact is `/job/baseline-first.json`.

The current delivery checkout was then validated with its existing independent
reference test:

```text
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q \
  test/registered/spec/dspark/test_dspark_kernel_parity.py --disable-warnings
```

Result: `1 passed`, `22 subtests passed`, `3 warnings`, exit code `0`, with
approximately `28 s` wall time (`23.12 s` reported by pytest).

## Reduced pipeline

The measured block is the commit bottleneck: commit-inject layout plus
per-linear or stacked KV projection. The upstream handoff is exercised first:

1. `select_draft_hidden_without_anchor` selects draft hidden rows from target
   hidden rows.
2. `AcceptGreedy.execute` verifies synthetic draft candidates against target
   argmax logits and applies the verify-length cap.
3. `FinalizeAcceptLens.execute` produces commit lengths and new sequence lengths.
4. `BuildCommitInjectLayout.torch` or `.triton` produces SWA locations and
   positions.
5. `CommitKvProj.torch` or `.triton` projects target hidden states through
   synthetic per-stage KV weights.

All four configurations are existing supported class methods on CUDA inputs:

- `torch_layout_torch_projection`
- `triton_layout_torch_projection`
- `torch_layout_triton_projection`
- `triton_layout_triton_projection`

The workload matrix has four cases, below the six-case limit:

| Case | Batch | Gamma | Stride | Target hidden | Target rows | Draft rows | KV head dim | Stages | Pool length | Synthetic weights |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `small_block` | 8 | 4 | 5 | 1024 | 40 | 32 | 128 | 2 | 512 | 524,288 B |
| `medium_block` | 32 | 4 | 5 | 2048 | 160 | 128 | 576 | 3 | 1024 | 7,077,888 B |
| `wide_batch` | 64 | 6 | 7 | 1024 | 448 | 384 | 128 | 2 | 2048 | 524,288 B |
| `large_token_block` | 128 | 4 | 5 | 512 | 640 | 512 | 128 | 2 | 4096 | 262,144 B |

Dtypes are fixed across configurations:

- Target hidden and projection weights: `torch.bfloat16`.
- Target logits: `torch.float32`.
- Candidates: `torch.int64`.
- Verify lengths and commit lengths: `torch.int32`.
- Prefix lengths, request/token mappings, positions, and SWA mappings:
  `torch.int64`.

## Accuracy gates

Every case/configuration pair uses the same generated inputs and independent
references. A failed gate rejects that configuration and prevents timing
acceptance.

- Draft hidden selection: exact dtype, shape, and value equality.
- Greedy accept length, bonus, and cap trim: exact equality against an
  independent prefix-product/argmax reference.
- Finalized commit lengths, new sequence lengths, and cap trim: exact equality.
- Commit-inject SWA locations and positions: exact equality against an
  independent indexing/masking reference.
- KV projection: finite `allclose` with `rtol=2e-2`, `atol=2e-3` against
  independent float32 linear references.

All 16 case/configuration pairs passed every gate. No numerical regression was
accepted.

## Timing and uncertainty

Timing uses CUDA events around real commit-block calls. Each sample is the mean
of 10 measured forwards after 3 warmup forwards. Each case/configuration pair
has 5 interleaved repeats on the same shared MI300X. The total bounded workload
is 1,040 commit-block forwards for timing plus 16 correctness calls. There are
no unbounded loops, sleep loops, or artificial GPU burn calls.

Median milliseconds per commit block are shown with interquartile range (IQR):

| Case | Torch layout + Torch proj | Triton layout + Torch proj | Torch layout + Triton proj | Triton layout + Triton proj |
|---|---:|---:|---:|---:|
| `small_block` | 0.163517 (IQR 0.029963) | 0.082126 (IQR 0.006699) | 0.165165 (IQR 0.012463) | 0.082639 (IQR 0.003727) |
| `medium_block` | 0.188259 (IQR 0.010825) | 0.101687 (IQR 0.005619) | 0.175329 (IQR 0.003554) | 0.096755 (IQR 0.006188) |
| `wide_batch` | 0.164913 (IQR 0.011458) | 0.080666 (IQR 0.001943) | 0.164672 (IQR 0.003057) | 0.083493 (IQR 0.003065) |
| `large_token_block` | 0.166729 (IQR 0.012543) | 0.081071 (IQR 0.004581) | 0.164772 (IQR 0.007401) | 0.082318 (IQR 0.003264) |

Raw per-sample medians, IQRs, minima, maxima, dimensions, dtypes, gates, paths,
and limits are in `results.json`.

In this reduced synthetic matrix, the Triton commit-layout path has a lower
median block time than the Torch layout path in all four cases. Switching only
the projection path did not improve median time at these small dimensions. This
is a shared-hardware observation with visible IQR spread, not a
workload-independent speedup claim.

## Resource and wall limits

- Synthetic weights: maximum case total `7,077,888 B` (`6.75 MiB`), below 4GB.
- Peak CUDA allocation: maximum `119,181,312 B` (`113.66 MiB`), below 48GB.
- Final CUDA allocation after the run: `96,468,992 B` (`92.0 MiB`).
- Workload cases: 4, at most 6 allowed.
- Kernel/dispatch configurations: 4, at most 4 allowed.
- Study wall limit: 7,200 seconds.
- Script-reported study elapsed time: `5.000759 s`; the full command including
  Python imports took approximately `19 s`.
- GPUs used: 1.
- Distributed execution: none; `world_size=1`.

## Reproduction

From the repository root:

```text
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m py_compile \
  reports/j-66135d2994b3/reduced_pipeline.py

PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  reports/j-66135d2994b3/reduced_pipeline.py \
  --output reports/j-66135d2994b3/results.json
```

The script exits nonzero if any numerical gate fails or a resource limit is
exceeded. It does not download weights or modify node-wide state.

## Left undone

- No GLM checkpoint, full model constructor, cache-pool integration, or
  end-to-end generation test was run.
- No multi-GPU or tensor-parallel claim is made.
- No upstream PR branch was checked out; current mirror `main` already contained
  the relevant primitives and was used as the preserved tested commit.
- The timing result is limited to the four synthetic reduced-block cases and one
  shared MI300X; it should not be generalized to other shapes, quantization
  schemes, models, or hardware.
