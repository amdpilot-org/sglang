# Independent review of PR 2282

Reviewed `https://github.com/amdpilot-org/sglang/pull/2282` at exact commit
`e0a959e5ddc6d2718cf8d94e8730dc07ebb463f2` against upstream issue
`https://github.com/sgl-project/sglang/issues/33385` and mirror issue
`https://github.com/amdpilot-org/sglang/issues/2321`.

## Finding

Recommendation: **accept**, with the original distributed serving race explicitly
remaining unverified. The candidate is a justified source-level correction, not
test-only hardening: when the `cpu_tensor` backup backend is selected but decode
KV offload is disabled, it no longer calls the unsupported DeepSeek-V4 CPU-copy
stub. It releases device KV and routes the request through the existing PD
rebootstrap/recompute path. If offload is enabled but the pool still raises
`NotImplementedError`, the same rebootstrap path is used. Unrelated copy errors
continue to propagate and host-pool exhaustion remains a distinct abort signal.

The recorded base independently reproduced the original contract violation:
with decode mode and `disaggregation_decode_enable_offload_kvcache=False`,
`release_req()` called offload once and raised `NotImplementedError`. The exact
candidate passed its regression suite and independent adversarial cases. Its tip
also preserves the established normal decode queue call shape while forwarding
`is_rebootstrap=True` only for actual rebootstrap requests.

`fully_resolves_original` is reported as false because the reported tp8/dp8
DeepSeek-V4 MTP4 HiSparse KV-full serving race was not executable here. The
available single gfx950 GPU and absent DeepSeek-V4 weights/distributed topology
cannot establish end-to-end race timing, distributed rebootstrap transport, or
semantic output. This is a verification limitation, not evidence of a remaining
source counterexample.

## Evidence

- `raw/base-original-repro.txt`: failing-before on recorded base `358c163...`;
  imported `/job/repo/python/sglang`, confirmed DeepSeek-V4 inherits the base
  unsupported CPU-copy stub, made one offload call despite the false flag, and
  raised `NotImplementedError` (exit 1).
- `raw/candidate-regression.txt`: candidate-owned fallback and priority tests;
  20 passed, 3 subtests passed.
- `raw/candidate-focused.txt`: broader related candidate coverage; 48 passed,
  5 subtests passed.
- `raw/candidate-adversarial.txt`: independent assertions for disabled offload,
  enabled-but-unsupported CPU copy, unrelated exceptions, host-pool exhaustion,
  and unknown backends; exit 0.
- `raw/candidate-gpu.txt` and `raw/gpu-environment.txt`: 2 exact-value supported
  MHA backup/restore tests passed on one AMD Instinct MI350X (gfx950), Torch
  2.11.0+rocm7.2. This does not cover DeepSeek-V4 or distributed serving.

No C/C++/HIP/FlyDSL source changed in the candidate, so no native rebuild was
applicable. The validated Python imports came from the checked-out source tree,
not an installed SGLang wheel.
