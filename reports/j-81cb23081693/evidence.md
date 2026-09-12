# Independent review of candidate PR 3505

Reviewed exact candidate commit `c7a959a488f29c77f38b5d0771860f1fc33bce07`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.
The prepared checkout was already exactly at the recorded base; no discrepancy was found.

## Environment and loaded code

- GPU: AMD Instinct MI350X, `gfx950:sramecc+:xnack-`, 270566162432 bytes VRAM
- Torch: `2.11.0+rocm7.2`; HIP: `7.2.26015`
- Interpreter: `/tmp/amdpilot-repo-j-81cb23081693/venv/bin/python`
- SGLang module: `/job/repo/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`
- Torch module: `/opt/venv/lib/python3.12/site-packages/torch/__init__.py`
- DeepGEMM was disabled/unavailable on this AMD host.
- The candidate changes only Python and Python tests; no native C++ or FlyDSL source changed,
  so no native rebuild was applicable.

Raw command output is retained outside the checkout in
`/job/review-evidence-j-81cb23081693/` so it survived revision switching.

## Failing-before reproduction

On the untouched recorded base, the actual registered `aten::mm.dtype` path was exercised
with BF16 `[17,7168] @ [7168,256]` inputs and an FP32 output request:

```text
actual_dtype: torch.float32
non_bf16_values: 0 / 4352
max_error_fp64_reference: 0.966156005859375
max_error_after_bf16_round: 0.966156005859375
repeat_equal: true
subbatch_equal: true
near_scores: [100.0, 100.0]
near_argmax: 0
```

Thus the base returns a nominally FP32 tensor containing values already rounded to BF16.
The independent near-tie construction should produce scores `100.0` and `100.0625`; base
rounding collapses both to `100.0` and changes the selected expert.

Command:

```bash
SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0 PYTHONPATH=/job/repo/python \
  /tmp/amdpilot-repo-j-81cb23081693/venv/bin/python \
  /job/review-evidence-j-81cb23081693/probe.py
```

Exit code: 0. Full output: `base-probe.log` in the external evidence directory.

## Exact-candidate validation

The same independent probe at exact candidate commit produced:

```text
actual_dtype: torch.float32
non_bf16_values: 4351 / 4352
max_error_fp64_reference: 0.00030517578125
max_error_after_bf16_round: 0.966156005859375
repeat_equal: true
subbatch_equal: true
near_scores: [100.0, 100.0625]
near_argmax: 1
near_bf16_argmax: 0
```

This directly demonstrates an FP32 store rather than BF16 widening and restores the intended
near-tied routing result without changing repeat/sub-batch invariance.

The candidate regression suite passed:

```bash
SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0 PYTHONPATH=/job/repo/python \
  /tmp/amdpilot-repo-j-81cb23081693/venv/bin/python -m pytest -q \
  test/registered/unit/batch_invariant_ops/test_batch_invariant_ops.py
```

Result: `11 passed, 19 warnings, 54 subtests passed in 33.33s` (exit code 0).

Independent adversarial cases covered shapes `(M,K,N)` of `(1,63,63)`, `(4,64,64)`,
`(5,65,129)`, `(16,127,256)`, `(17,129,257)`, and `(129,257,64)`. All returned FP32,
retained values not representable in BF16, repeated bitwise, and had maximum absolute error
from `9.54e-7` to `1.14e-5` against CPU FP64 matmul rounded once to FP32. A separate FP32-bias
case had maximum error `5.72e-6` and retained non-BF16 values. The actual `MoEGate.forward`
deterministic branch was exercised at 1, 4, 5, 16, and 17 tokens; every output was FP32 and
every prefix was bitwise equal to the corresponding rows of the 17-token result.

## Source review

The candidate allocates the Triton output in the requested dtype, making the kernel's existing
FP32 store reachable. `_mm_dtype_compat` forwards `out_dtype` rather than widening a rounded
result. FP32 requests bypass the BF16-only DeepGEMM function and use the same persistent Triton
path for every token count. The deterministic DeepSeek gate requests FP32 through
`torch.mm(..., out_dtype=torch.float32)` on CUDA/HIP. These changes preserve the existing tile
configuration and reduction loop; only output allocation/store selection changes.

No counterexample was observed on the supported ROCm gfx950 kernel. CUDA execution and actual
DeepGEMM execution remain unverified because the assigned architecture is AMD and DeepGEMM is
not available. The DeepGEMM bypass was source-reviewed and mock-tested, but that is not a real
NVIDIA DeepGEMM execution. The tiny Llama serving fixture was intentionally not used because it
cannot qualify DeepSeek router numerical semantics.

Recommendation: accept the candidate. The implementation and regression are substantive, not
test-only hardening. Universal full resolution is not claimed solely because the CUDA hardware
path could not be executed in this review environment.
