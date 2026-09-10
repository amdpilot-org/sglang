# gfx942 EP scatter dtype-path investigation: j-65a03906d3be

## Scope and result

This follow-up tests the direct `ep_scatter` payload contract for `bfloat16`
and `float8_e4m3fn` on one assigned AMD Instinct MI300X (`gfx942`). It does not
repeat the original race trigger from amdpilot-org/sglang issue 212.

All four bounded BF16/FP8 cases passed exact bit-level gates on mirror pull
request 241 commit `f61b86c7ccc92e8facb7823d00a9c1116394b353`. The documented
BF16 scale-ignored behavior passed, an FP8 scale dtype mismatch failed clearly
before dispatch, and one HIP graph capture/replay preserved all static tensor
addresses while producing the same exact result.

No kernel or production-test code is changed by this delivery. The existing
race fix in mirror pull request 241 was preserved and not duplicated.

## Prior evidence and uncovered scope

- Upstream context: sgl-project/sglang issue 31929.
- Upstream candidate: sgl-project/sglang pull request 31930, commit
  `3612e7a1ea60324e9425e7a3d41ed9f2414cad78`.
- Existing mirror fix: amdpilot-org/sglang pull request 241, commit
  `f61b86c7ccc92e8facb7823d00a9c1116394b353`, which changes the unsafe reload
  to a register reduction.
- Existing mirror follow-up: amdpilot-org/sglang pull request 338 already
  covers finite `float16` scatter-plus-inverse-gather round trips.

The uncovered case here is direct BF16/FP8 payload and scale representation,
including sentinel-protected storage, the BF16 scale-ignored contract, an
unsupported FP8 scale dtype mismatch, and static graph-address replay.

No upstream issue, pull request, or comment was posted or modified.

## Environment

- Required image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Operator-provided local image ID:
  `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- GPU: one AMD Instinct MI300X, compute major/minor `9/4`, 304 compute units.
- Python: `/opt/venv/bin/python` (3.10.12)
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- Torch native module:
  `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`
- Torch HIP library:
  `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so`
- Triton: `3.7.0`, `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`
- Job-private Triton cache:
  `/tmp/sglang-cache-j-65a03906d3be/triton-fixed`

The qualified Torch/ROCm stack was preserved. No model weights, framework
stack, or node-wide state were downloaded or changed.

## Installed-source first baseline

The first installed-source run used the preinstalled interpreter and imported
`/sgl-workspace/sglang/python/sglang/kernels/ops/moe/ep_moe_kernels.py` at
commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`.

The first successful GPU execution completed `4.672944662161171` seconds after
process start. The run dispatched real Triton kernels and generated AMDGCN
and HSACO artifacts under the job-private cache. The BF16, FP16, and FP8
ordered-index references failed because the installed source reproduced the
already-known `_fwd_kernel_ep_scatter_1` race. This installed-source result is
context only and is not proof for the later mirror checkout.

The first FP8 reference attempt also failed before dispatch with:

```text
RuntimeError: value cannot be converted to type at::Float8_e4m3fn without overflow
```

The reference was corrected to initialize FP8 sentinel storage through a
`uint8` view. The complete installed-source record is in
`reports/j-65a03906d3be/baseline-first.json`.

## Fixed-commit experiment

The persistent mirror checkout was detached at preserved PR 241 commit
`f61b86c7ccc92e8facb7823d00a9c1116394b353`. The delivery branch remains based
on mirror `main` at `0084030179bfba86bfeb6d43f7997d4076329d2c`.

Each timed case used:

- four local experts;
- padded counts `[128, 256, 0, 128]`;
- valid counts `[100, 200, 0, 128]`;
- 214 source tokens and top-2 routing;
- 512 padded destination rows;
- exact BF16 or FP8 payload bit patterns;
- exact FP32 scale bit patterns for FP8;
- sentinel guard storage before and after `m_indices`, `output_index`,
  `output_tensor`, and `output_tensor_scale`.

The independent reference checks:

- post-scatter cursors equal exclusive starts plus valid counts;
- `m_indices` exactly matches independently constructed expert IDs and `-1`
  padding;
- `output_index` is assigned, unique, and exactly covers each expert's valid
  destination set;
- each `output_index` maps back to the independently derived local expert;
- each scattered payload row exactly matches its source row at the bit level;
- each FP8 scale row exactly matches its source scale row at the bit level;
- all guard storage remains unchanged.

Atomic destination order is intentionally nondeterministic, so the reference
does not require a particular ordered `output_index` assignment.

| Case | Dtype | Hidden | Median ms |
|---|---|---:|---:|
| `bf16_hidden128` | BF16 | 128 | `0.09582100063562393` |
| `bf16_hidden1024` | BF16 | 1024 | `0.0931750015424324` |
| `fp8_hidden128` | FP8 E4M3FN | 128 | `0.09991099685430527` |
| `fp8_hidden1024` | FP8 E4M3FN | 1024 | `0.09995099902153015` |

Timing used three warmups followed by ten CUDA-event samples per case and
reports the median. The first fixed-commit GPU execution completed
`4.902326873037964` seconds after process start.

The BF16 control with a non-`None` scale passed and left the output scale
sentinel unchanged, matching the documented BF16 scale-ignored path. The FP8
control with `float32` input scale and `float16` output scale failed clearly
before dispatch:

```text
recv_x_scale.dtype: torch.float32, output_tensor_scale.dtype: torch.float16
```

The HIP graph control captured and replayed one BF16 call. The source,
`output_tensor`, `m_indices`, and `output_index` data pointers were identical
before and after replay, and every exact gate passed after replay.

The complete raw record, including all ten timing samples, native HSACO paths,
commands, and checks, is in
`reports/j-65a03906d3be/mirror-fixed-dtype-gfx942-results.json`.

## Reproduction

From the preserved PR 241 commit in this repository root:

```bash
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-65a03906d3be/triton-fixed
start_ns=$(date +%s%N)
PYTHONPATH="$PWD/python" /opt/venv/bin/python \
  reports/j-65a03906d3be/reproduce.py \
  --start-ns "$start_ns" \
  --output /tmp/ep-scatter-dtype-gfx942-results.json
```

The script performs four timed cases, one BF16 ignored-scale control, one
unsupported FP8 scale-dtype control, and one graph capture/replay control. It
uses no unbounded loop, sleep loop, synthetic GPU burn, or repeated work solely
to occupy the GPU.

## Honest limitations

- This is local kernel-contract evidence only, not distributed expert-parallel,
  DeepEP transport, or full-model proof.
- The tested commit is the open PR 241 fix, not merged mirror `main`; the
  delivery branch itself contains report and reproduction evidence only.
- No source/output aliasing variant is documented for `ep_scatter`, so distinct
  source and destination buffers were used and no unsupported aliasing was
  forced.
- No multi-GPU, long-running stress, illegal-access race scheduling, or model
  weights were used.
- The container has no Docker/CRI utility, so the operator-provided local image
  ID could not be independently resolved from container metadata.
