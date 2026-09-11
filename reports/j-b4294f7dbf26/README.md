# MI300X GDN target-verify state dtype investigation

## Scope

This is the execution-representation follow-up to mirror issue 216. It does not
repeat the original `mamba_next_track_idx is None` trigger: upstream issue 34786
is already addressed by upstream PRs 27998 and 30437. The uncovered case tested
here is the documented Mamba SSM state dtype path through the real
`fused_sigmoid_gating_delta_rule_update` TARGET_VERIFY kernel.

The state dtype map in `mamba2_state_dtype` documents `float32`, `bfloat16`,
and `float16`. Existing GDN verify coverage used a `float32` intermediate
state buffer. This change covers all three state representations, rejects an
undocumented `float64` state before dispatch, and preserves the allocated slot
pitch, sentinel-protected neighboring storage, and buffer address.

## Environment

- GPU: one AMD Instinct MI300X, `gfx942`, driver `6.19.14.31400000`.
- Interpreter: `/opt/venv/bin/python`, Python 3.10.12.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`.
- Installed-source baseline: sglang `0.5.18.dev20260826+g937af8538b`,
  commit `8eaffdf382`, module `/sgl-workspace/sglang/python/sglang`.
- Delivery checkout: commit `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Operator-specified image:
  `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local image ID
  `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
  No container runtime or image metadata API was available inside the job, so
  the image ID could not be independently re-read; hostname was not used as
  image identity.

## Installed-source baseline

The first successful GPU execution was recorded at `/job/baseline-first.json`.
It ran the existing GDN TARGET_VERIFY numerical test against the installed
source before cloning or editing:

```bash
TRITON_CACHE_DIR=/tmp/sglang-cache-j-b4294f7dbf26-baseline \
/opt/venv/bin/python -m pytest \
  /sgl-workspace/sglang/test/registered/kernels/ops/attention/test_fused_verify_triton_gdn.py \
  -q -s -p no:cacheprovider
```

Result: 12 passed in 11.64 seconds (13.145258344 seconds wall clock). The
unchanged state gates had max differences of `1.77e-03`, `2.19e-03`, and
`3.01e-03`, each with a 0.00% failure rate. This installed-source result is
not evidence for checkout changes.

## Checkout results

Run the focused GDN suite with:

```bash
export PYTHONPATH=/job/sglang/python
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-b4294f7dbf26/triton
export SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-b4294f7dbf26/jit
export AMD_COMGR_CACHE_DIR=/tmp/sglang-cache-j-b4294f7dbf26/comgr
/opt/venv/bin/python -m pytest \
  test/registered/kernels/ops/attention/test_fused_verify_triton_gdn.py \
  -q -s -p no:cacheprovider
```

Result: 18 passed. The original numerical gates were unchanged.

The new independent reference is a direct PyTorch recurrence. It consumes the
same dtype-specific tensors as the kernel, rather than reusing kernel output or
aliased scratch. Matched values are generated once as float32 masters and then
converted once to each state dtype. The sentinel-protected buffer has guard
slots before and after the selected write slots and an allocated step pitch of
8 while only 1 or 4 steps are active.

The bounded timing and raw result matrix is in `mirror-gpu-results.json`. It
covers `{float32, bfloat16, float16} × batch {1,4} × steps {1,4}`. After one
warmup per specialization, one CUDA event pair times one kernel invocation per
case. Observed times were approximately 0.147–0.168 ms. Every case preserved
guard slots, unwritten steps, and the buffer address. Maximum absolute
differences against the independent reference were:

- float32: `2.98e-07` to `4.77e-07`.
- bfloat16: `0.0` to `6.10e-05`.
- float16: `0.0` to `4.88e-04`.

The actual dispatch is Triton JIT through the HIP backend for `gfx942`. The
probe records the compiled kernel hashes and confirms `source`, `ttir`,
`ttgir`, `llir`, `amdgcn`, and `hsaco` artifacts. The wrapper now rejects
`float64` state storage with a clear `ValueError` instead of allowing an
undocumented dtype to reach the kernel.

Additional focused wrapper paths passed:

```bash
/opt/venv/bin/python -m pytest -q -s -p no:cacheprovider \
  test/registered/attention/test_gdn_noncontiguous_stride.py \
  test/registered/attention/unittests/gdn/test_gdn_replayssm_spec_fold.py \
  test/registered/kernels/test_kda_replayssm_fold.py \
  test/registered/kernels/ops/test_kimi_k3_prerequisite_ops.py::\
TestKimiK3PrerequisiteOps::test_replayssm_ring_fold
```

Result: 15 passed.

## Limitations

- This is a focused kernel-contract result, not a full-model E2E run.
- No model weights were downloaded.
- Only one MI300X was used; NVIDIA and multi-GPU behavior is not established.
- A broader `test_kimi_k3_prerequisite_ops.py` run had two unrelated MoE JIT
  failures under ROCm 7.2 (`__float2bfloat16_rn` and a 64-bit warp-mask
  assertion in `route_radix.cuh`). The GDN wrapper test in that file passed.
- `ruff` was unavailable in the qualified environment; `git diff --check`
  passed.
