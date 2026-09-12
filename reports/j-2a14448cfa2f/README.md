# Investigation of FP8 KV-cache decode overhead

Upstream issue: https://github.com/sgl-project/sglang/issues/30815

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2356

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Outcome

The KV-write portion of the report is reproduced in the prepared checkout, but
no implementation change is proposed here because upstream PR
https://github.com/sgl-project/sglang/pull/31652 is an active, current-main
candidate for precisely this defect. Its author confirmed continued work on
2026-09-12. Duplicating that patch would not be a narrow independent correction.

The prepared source still executes the reported eager sequence in
`MHATokenToKVPool.set_kv_buffer`: two optional in-place divides, two FP8 casts,
then `_set_kv_buffer_impl`/`store_cache`. The pool-level GPU fixture records one
relevant device kernel for BF16 and five for scaled FP8. On the assigned MI350X
gfx950, batch 1 with 8 KV heads of dimension 128 measured median 24.88 us for
BF16 and 46.72 us for scaled FP8 (1.88x). K and V output were bit-exact against
the independent `(source / scale).to(torch.float8_e4m3fnuz)` reference.

The profiler also records two device-to-device copies in each case because the
fixture clones caller inputs. Those are fixture operations and are excluded
from the one-versus-five cache-write kernel census; the complete event list is
retained in `reproduce_fp8_kv_write.log`.

## Related fix review

Upstream issue comment https://github.com/sgl-project/sglang/issues/30815#issuecomment-5010010078
points to PR #31652. The PR adds a CUDA-only fused quantize-and-store JIT kernel
and explicitly leaves ROCm on the eager path. Its current commit is
`9fc7b60e25bd91cb18d399cd05e8316452e9680a`; the PR remains open. This checkout
does not contain that implementation (`kvcache.py` exposes only `store_cache`,
and `memory_pool.py` retains `div_` plus `to`).

## Evidence and limitations

- `reproduce_fp8_kv_write.py`: deterministic pool-level reproducer using the
  actual checked-out `MHATokenToKVPool` implementation.
- `reproduce_fp8_kv_write.log`: raw device, timing, profiler-event, and numerical
  output.
- `focused_tests.log`: existing cache kernel and MHA pool tests, 366 passed plus
  9 subtests.
- GPU execution used exactly one assigned MI350X (`gfx950:sramecc+:xnack-`).
- The original report is specifically Hopper H200/SM90 with FA3. That hardware
  was unavailable, so no H200, FA3 attention-kernel, full-model ITL, or semantic
  accuracy claim is made.
- No model weights were needed for the isolated write-path reproduction. The
  tiny Llama serving fixture would validate transport and generic engine
  execution only, not Hopper FA3 behavior, so it was not substituted for the
  reported workload.
- The separate per-layer query-conversion and FA3 split-KV claims remain
  unverified here because FA3/Hopper is unavailable.
- No native source was changed or rebuilt.

