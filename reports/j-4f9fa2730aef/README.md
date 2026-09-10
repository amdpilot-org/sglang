# gfx942 `fast_topk_v2` overflow investigation

## Result

- Current mirror `main` (`0084030179bfba86bfeb6d43f7997d4076329d2c`) does **not**
  compile `topk.cu` for ROCm. Its actual gfx942 `fast_topk_v2` path is the native
  `topk.hip` cooperative implementation introduced by upstream PR 37591. That path
  was exact across 4095, 4096, 4097, and 4607 resolved candidates.
- The exact upstream PR 37893 candidate commit
  `f36a7588361ec92351e40e3fb59d56f560ad792b` was also testable on gfx942 because
  its base still compiled `topk.cu` through the ROCm hipifier. It was exact across
  6141, 6143, 6144, 6145, 6147, and 6186 coarse candidates.
- The candidate's parent `978cc228caabaaaab19fa5ca88f50d6170cb4326` reproduced the
  silent overflow. At 6145 candidates, one of four rows returned 1,500 wrong indices;
  at 6186 candidates, all four rows were wrong, with 1,843–2,022 wrong indices per row.
- PR 37893 removed those continuous-score errors and all repeat-to-repeat changes on
  the same inputs. This validates that historical ROCm build of the candidate; it does
  **not** claim that the current-main native `topk.hip` control fixes CUDA `topk.cu`.
- No product code was changed. This is a bounded validation and evidence report.

## Upstream context

- Read-only issue: sgl-project/sglang issue 36807, still open.
- Read-only candidate: sgl-project/sglang PR 37893, still open, commit
  `f36a7588361ec92351e40e3fb59d56f560ad792b`, parent
  `978cc228caabaaaab19fa5ca88f50d6170cb4326`.
- Read-only related change: sgl-project/sglang PR 37591, merged as
  `6cee9285a3dd43e1f0270818aef7bf01a0568863`, replaced ROCm's hipified `topk.cu`
  with native `topk.hip` and `dsa_topk_coop.cuh`.
- No upstream issue, PR, comment, or review was posted or modified.

## Environment

- Image assigned by the operator:
  `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local image ID
  `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
  Docker CLI is not available inside the container, so the ID is recorded from the
  assignment rather than re-queried; hostname was not used as image identity.
- GPU: one AMD Instinct MI300X, gfx942, serial `692440003981`, GUID `61795`,
  unique ID `0x439e01ac3221d888`, device capability `(9, 4)`.
- Python: `/opt/venv/bin/python`, Python 3.10.12.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`,
  loaded from `/opt/venv/lib/python3.10/site-packages/torch`.
- Delivery source: `/job/j-4f9fa2730aef/sglang`.
- Current-main native module:
  `/tmp/sglang-cache-j-4f9fa2730aef/build-base/lib.linux-x86_64-cpython-310/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`.
- Candidate native module:
  `/tmp/sglang-cache-j-4f9fa2730aef/candidate-f36a7588/build-base/lib.linux-x86_64-cpython-310/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`.
- Baseline native module:
  `/tmp/sglang-cache-j-4f9fa2730aef/baseline-978cc228/build-base/lib.linux-x86_64-cpython-310/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`.
- All build and test caches stayed outside the delivery checkout under
  `/tmp/sglang-cache-j-4f9fa2730aef`. No model weights were downloaded.

## Path boundary

Current `main` has this exact ROCm source selection in
`python/sglang/kernels/aot/setup_rocm.py`:

```text
# Native HIP implementation of the same three ops exposed by topk.cu.
"csrc/elementwise/topk.hip",
```

`topk.cu` is absent from that source list. Therefore current-main CUDA `topk.cu`
is an unsupported boundary for this gfx942 build: it was not compiled and its
overflow behavior was not inferred from the native ROCm control.

The historical PR 37893 base is different. At commits `978cc228...` and
`f36a7588...`, `setup_rocm.py` still listed `csrc/elementwise/topk.cu`; the build
hipified it to `topk.hip` and compiled `csrc/elementwise/topk.o` for gfx942. The
candidate test is therefore a real historical ROCm build of the changed source, not
a substitute CUDA control.

The current native path uses `kHistBits = 12`, `kTieCap = 4096`, and a full-row
rescan when the tie buffer overflows. The historical radix path used
`SGL_TOPK_DYNAMIC_SMEM_BYTES=49152`, so its candidate capacity was
`49152 / (2 * sizeof(int)) = 6144`.

## Probe design

The probe is `probe_fast_topk_v2_gfx942.py`. It uses `k=2048`, batch 4, five calls
per case, and one case resident at a time.

- Continuous rows are distinct float32 values made by permuting a `linspace` row.
  This avoids accidental float32 ties while preserving a dense, continuous score
  distribution around the candidate boundary.
- The current native path uses widths and lengths that produce 4095, 4096, 4097,
  and 4607 resolved candidates.
- The historical radix path uses widths and lengths that produce 6141, 6143, 6144,
  6145, 6147, and 6186 fp16 coarse-bin candidates.
- Independent `torch.topk` supplies the oracle. For continuous rows, the probe
  checks exact selected-index sets, exact selected-value multisets, in-range and
  duplicate-free indices, no selected value below the k-th value, and no unselected
  value above it. It also records the k-th/(k+1)-th gap.
- The ties-only control uses 65,536 equal scores. Any 2,048 indices are valid. The
  probe requires all selected values to equal the k-th value, all selected indices
  to come from the tie group, and no selected value to be below the threshold. Raw
  differences from `torch.topk` and from repeat 0 are retained but are not errors.
- Peak allocated memory was 15,141,376 bytes for the current-main probe and
  13,596,672 bytes for each historical radix probe.

## Raw results

### Current main: native `topk.hip`

| Length | Resolved candidates | Max set errors | Max value errors | Max below k-th | Max unselected above k-th | Max repeat mismatch |
|---:|---:|---:|---:|---:|---:|---:|
| 65,535 | 4,095 | 0 | 0 | 0 | 0 | 0 |
| 65,536 | 4,095 | 0 | 0 | 0 | 0 | 0 |
| 65,537 | 4,096 | 0 | 0 | 0 | 0 | 0 |
| 65,552 | 4,096 | 0 | 0 | 0 | 0 | 0 |
| 65,553 | 4,097 | 0 | 0 | 0 | 0 | 0 |
| 73,728 | 4,607 | 0 | 0 | 0 | 0 | 0 |

All continuous rows had zero duplicate source values. The minimum k-th/(k+1)-th
gap was `1.7881393432617188e-06`, so these were genuinely distinct values rather
than hidden ties.

The ties-only control selected 2,048 indices from the 65,536-member tie group in
every row, had zero below-threshold selections, and correctly allowed multiple
valid outputs. Its raw repeat-to-repeat mismatch reached 1,854 indices; this is
expected nondeterministic tie resolution, not a correctness error.

### PR 37893 parent: hipified `topk.cu`

| Length | Coarse candidates | Max set errors | Max value errors | Max below k-th | Max unselected above k-th | Max repeat mismatch |
|---:|---:|---:|---:|---:|---:|---:|
| 65,520 | 6,141 | 0 | 0 | 0 | 0 | 0 |
| 65,540 | 6,143 | 0 | 0 | 0 | 0 | 0 |
| 65,550 | 6,144 | 0 | 0 | 0 | 0 | 0 |
| 65,558 | 6,145 | 1,500 | 474 | 1 | 1 | 1,500 |
| 65,580 | 6,147 | 1,819 | 2,000 | 3 | 3 | 1,819 |
| 66,000 | 6,186 | 2,022 | 1,950 | 16 | 16 | 1,778 |

At 6,145 candidates, the first repeat had per-row set errors `[0, 1500, 0, 0]`,
value errors `[0, 474, 0, 0]`, one below-threshold selection, and one unselected
above-threshold value. Later repeats changed the wrong sets. At 66,000 length, all
rows were wrong and repeat-to-repeat mismatches remained nonzero.

### PR 37893 exact commit: hipified `topk.cu`

| Length | Coarse candidates | Max set errors | Max value errors | Max below k-th | Max unselected above k-th | Max repeat mismatch |
|---:|---:|---:|---:|---:|---:|---:|
| 65,520 | 6,141 | 0 | 0 | 0 | 0 | 0 |
| 65,540 | 6,143 | 0 | 0 | 0 | 0 | 0 |
| 65,550 | 6,144 | 0 | 0 | 0 | 0 | 0 |
| 65,558 | 6,145 | 0 | 0 | 0 | 0 | 0 |
| 65,580 | 6,147 | 0 | 0 | 0 | 0 | 0 |
| 66,000 | 6,186 | 0 | 0 | 0 | 0 | 0 |

The candidate's ties-only control also selected 2,048 valid members from the tie
group, with zero below-threshold selections. Its raw repeat mismatch reached 512
indices, again valid under the multiple-output tie semantics.

## Unchanged numerical gates

- Current main:
  `/opt/venv/bin/python -m pytest python/sglang/kernels/aot/tests/test_topk.py -q --maxfail=1`
  produced **160 passed, 1 skipped** in 36.48 seconds.
- PR 37893 commit `f36a7588...`:
  `/opt/venv/bin/python -m pytest python/sglang/kernels/aot/tests/test_topk.py -q --maxfail=1`
  produced **123 passed** in 34.42 seconds.

## Reproduction

From the delivery checkout:

```bash
cd /job/j-4f9fa2730aef/sglang
export PYTHONPATH=/tmp/sglang-cache-j-4f9fa2730aef/build-base/lib.linux-x86_64-cpython-310
/opt/venv/bin/python reports/j-4f9fa2730aef/probe_fast_topk_v2_gfx942.py \
  --label current-main-rocm-topk.hip \
  --revision 0084030179bfba86bfeb6d43f7997d4076329d2c \
  --path rocm-coop \
  --output reports/j-4f9fa2730aef/results/current-main.json
```

For the historical candidate, use the candidate build directory in `PYTHONPATH`:

```bash
cd /job/j-4f9fa2730aef/sglang
export PYTHONPATH=/tmp/sglang-cache-j-4f9fa2730aef/candidate-f36a7588/build-base/lib.linux-x86_64-cpython-310
/opt/venv/bin/python reports/j-4f9fa2730aef/probe_fast_topk_v2_gfx942.py \
  --label upstream-pr-37893-f36a7588-topk.cu-hipified \
  --revision f36a7588361ec92351e40e3fb59d56f560ad792b \
  --path radix-cuda-port \
  --output reports/j-4f9fa2730aef/results/pr-37893-f36a7588.json
```

The parent run is identical except for its `--label`, `--revision`, output file,
and baseline `PYTHONPATH`.

Builds used:

```bash
cd <worktree>/python/sglang/kernels/aot
MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 \
TORCH_EXTENSIONS_DIR=/tmp/sglang-cache-j-4f9fa2730aef/<revision>/torch-extensions \
/opt/venv/bin/python setup_rocm.py build \
  --build-base /tmp/sglang-cache-j-4f9fa2730aef/<revision>/build-base \
  --build-temp /tmp/sglang-cache-j-4f9fa2730aef/<revision>/build-temp \
  --parallel 8
```

Observed build times were 52 seconds for current main, 49 seconds for the
candidate, and 48 seconds for the parent.

## Artifacts

- Probe: `reports/j-4f9fa2730aef/probe_fast_topk_v2_gfx942.py`
- Current-main raw JSON: `reports/j-4f9fa2730aef/results/current-main.json`
- Candidate raw JSON: `reports/j-4f9fa2730aef/results/pr-37893-f36a7588.json`
- Parent raw JSON: `reports/j-4f9fa2730aef/results/pr-37893-parent-978cc228.json`
- Current-main test log: `reports/j-4f9fa2730aef/results/current-main-test-topk.txt`
- Candidate test log: `reports/j-4f9fa2730aef/results/pr-37893-f36a7588-test-topk.txt`

## Limitations

- This job had one gfx942 GPU and did not run a CUDA GPU. No claim is made about
  CUDA `topk.cu` on current main from the native ROCm control.
- PR 37893 remains open and is not based on current main. Its validation is tied
  to the preserved commit `f36a7588...`; it would need a rebase review against the
  current source selection and PR 37591.
- Timing values are single-process event timings for allocation-bounded probes,
  not a performance benchmark.
