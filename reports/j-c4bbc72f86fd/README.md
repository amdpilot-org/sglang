# Bounded ROCm stochastic verifier study

## Scope

This is a report-only, reduced-block study for the speculative verifier/sampling
path referenced by `sgl-project/sglang` issue 30344. It does not change
production code and does not claim a full-model speedup.

I reviewed the current issue description, comments, and related changes. The
relevant neighboring work includes `sgl-project/sglang` PRs 34410, 34286,
32281, 31466, and 35413; none of them duplicate this reduced-block benchmark.

The mirror already has an open, working implementation in
`amdpilot-org/sglang` PR 157. I tested that commit as a candidate and did not
duplicate or modify it:

- PR: https://github.com/amdpilot-org/sglang/pull/157
- Tested commit: `82f29ac821b5bc2d8f811b4728c18cab495122dd`
- Result: 4/4 PR tests passed on gfx942

## Environment

- GPU: one assigned AMD Instinct MI300X, gfx942
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- Installed source: `/sgl-workspace/sglang` at `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`
- Installed native module: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`
- Candidate source: `/job/sglang`
- Candidate native module: `/tmp/sglang-cache-j-c4bbc72f86fd/pr157-run/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`

## Baseline and controls

The first GPU execution was the installed speculative top-k test:

```bash
/opt/venv/bin/python -m pytest -q \
  /sgl-workspace/sglang/test/registered/kernels/ops/speculative/test_spec_topk1.py -s
```

It passed 5 tests in 39.97 seconds. Raw details are in `baseline-first.json`.

The installed stochastic tree op is not registered by the installed native
module, although its Python wrapper exists:

```text
AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute
'tree_speculative_sampling_target_only'
```

The supported neighboring control, `test_eagle_utils.py`, passed 1/1 in 6.86
seconds. Details are in `installed-controls.json`.

## Benchmark

`bench_stochastic_verifier.py` uses the PR 157 native
`tree_speculative_sampling_target_only` kernel with:

- Synthetic binary draft trees
- Synthetic one-hot target probabilities
- Fixed uniform samples (`0.5`) and final samples (`0.25`)
- `float32` probabilities and uniform samples
- `int64` tree/candidate tensors
- `int32` verifier outputs
- Thresholds `threshold_single=1.0` and `threshold_acc=1.0`
- Independent CPU reference implemented in the script

The kernel uses one 1024-thread block per batch row. All tested vocab sizes are
divisible by four, so the existing dispatch selects `VEC_SIZE=4`. The study
uses two existing dispatch configurations:

- `deterministic=true`
- `deterministic=false`

There are four workload cases, within the six-case limit:

| Case | Batch | Tree tokens | Vocab | Spec steps | Deterministic | Mean ms | Std ms |
|---|---:|---:|---:|---:|---|---:|---:|
| small | 2 | 8 | 1024 | 4 | true | 0.04823 | 0.00098 |
| medium | 16 | 16 | 4096 | 5 | true | 0.04708 | 0.00086 |
| large | 64 | 32 | 8192 | 6 | true | 0.10143 | 0.00135 |
| medium_nondeterministic | 16 | 16 | 4096 | 5 | false | 0.06333 | 0.00081 |

Every case matched the independent reference exactly for:

- `predicts`
- `accept_index`
- `accept_token_num`
- mutated `draft_probs`

Timing uses CUDA/HIP events around bounded loops. Each case performs 5 warm-up
calls, then 3 repeats of 20 timed calls. The reported uncertainty is the
standard deviation across those three repeat means on the same assigned GPU.

The largest case uses 134,301,696 live tensor bytes and peaks at 134,312,448
allocated bytes, well below the 48 GB live-allocation limit. No model weights
are downloaded or generated.

## Reproduction

Build PR 157:

```bash
cd /job/sglang/python/sglang/kernels/aot
MAX_JOBS=8 TORCH_CUDA_ARCH_LIST=gfx942 \
  /opt/venv/bin/python setup_rocm.py build_ext \
  --build-temp /tmp/sglang-cache-j-c4bbc72f86fd/pr157-build \
  --build-lib /tmp/sglang-cache-j-c4bbc72f86fd/pr157-lib
```

Run its tests:

```bash
PYTHONPATH=/tmp/sglang-cache-j-c4bbc72f86fd/pr157-run \
PYTORCH_ROCM_ARCH=gfx942 \
/opt/venv/bin/python -m pytest -q -p no:cacheprovider \
  python/sglang/kernels/aot/tests/speculative/test_speculative_sampling.py
```

Run the benchmark:

```bash
PYTHONPATH=/tmp/sglang-cache-j-c4bbc72f86fd/pr157-run \
PYTORCH_ROCM_ARCH=gfx942 \
/opt/venv/bin/python \
  reports/j-c4bbc72f86fd/bench_stochastic_verifier.py \
  --output reports/j-c4bbc72f86fd/stochastic-verifier-results.json \
  --warmup-calls 5 --timed-calls 20 --timing-repeats 3
```

Raw results are in `stochastic-verifier-results.json`.
