# Independent review of amdpilot-org/sglang#637

Candidate reviewed exactly at `d1f6440c5ef02abc13ec640a2b860e1e85a7d750`
against base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **request changes**.

The candidate is a meaningful partial fix. It makes CUDA version comparison
numeric, rejects sm89 in `is_fa3_supported()`, adds the missing guard to
`flash_attn_with_kvcache`, and makes `ver=3` the only accepted value rather than
silently sending every value to the FA3 operator. Its 22 focused tests pass.

It does not fully close the original launch-time gap. `_validate_fa3_contract()`
calls `is_fa3_supported()` without a device, so PyTorch checks the current CUDA
device rather than the device that owns `q`, `k`, and `v`. In a multi-GPU
process whose current device is supported (for example sm90) while the input
tensors are on sm89, validation passes and `torch.ops.sgl_kernel.fwd.default`
is still reached for the unsupported sm89 tensors. The inverse can reject a
supported tensor merely because another, current device is unsupported.
`flash_attn_varlen_func` has the same problem. The independent instrumented
probe observed `is_fa3_supported(None)` and a backend call for a tensor modeled
on `cuda:1`; the candidate tests do not assert the tensor device is forwarded.

There is also a contract caveat: the issue explicitly proposed honoring
`ver=2` with FA2 or dropping the parameter. The candidate retains the public
parameter but rejects 2. This is safer than silently ignoring it and is useful
hardening, but it is not FA2 dispatch and it does not remove the advertised
selector.

## Evidence

- On the base, the actual source entrypoint sent `ver=2`, `3`, `4`, `None`, and
  `"3"` across the same instrumented `torch.ops.sgl_kernel.fwd.default`
  boundary. The base also reported every 8.x and 9.x capability tested,
  including `(8, 9)`, as supported with CUDA 12.3.
- At the candidate commit, its regression suite passed: `22 passed`.
- Independently, the candidate accepts `(8,0)`, `(8,6)`, and `(9,0)` and rejects
  `(8,7)`, `(8,9)`, `(9,1)`, and `(10,0)` in the source wrapper.
- Independently, `ver=3` reached the FA3 boundary; `2`, `4`, `None`, and `"3"`
  were rejected before it.
- CMake emits FA3 `sm_90a`, plus `sm_80` and `sm_86` only when
  `ENABLE_BELOW_SM90` is enabled. No sm89 FA3 target is present.
- The candidate contains no native source change; a native rebuild was not
  applicable.

## Environment limitations

The prepared interpreter is `/tmp/amdpilot-repo-j-faace530f82a/venv/bin/python`,
with Torch `2.11.0+rocm7.2` from `/opt/venv`. It exposes one AMD Instinct MI350X.
There is no NVIDIA GPU, `nvcc`, `cuobjdump`, or `nvidia-smi`. The installed
`sgl_kernel` resolves to the ROCm environment egg, while review tests explicitly
loaded the checked-out source at
`python/sglang/kernels/aot/python/sgl_kernel/flash_attn.py`. No locally built
CUDA `flash_ops` library exists. Therefore CUDA binary contents, NVIDIA numerical
correctness, actual sm89 execution, and cubin runtime behavior remain unverified.

Raw commands and outputs were retained outside the revision-switching checkout
under `/job/review-evidence-j-faace530f82a/`.
