# Independent review of candidate PR 1666

- Upstream issue: https://github.com/sgl-project/sglang/issues/34205
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1703
- Candidate: https://github.com/amdpilot-org/sglang/pull/1666
- Exact candidate commit: `1216431b2d712b6c9a3a5e1eb3fcfa89963a8d41`
- Recorded and prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Recommendation: **accept**

## Finding

The candidate fully resolves the original request-level LoRA reference leak in
the implementation paths exercised by the issue's deterministic contract. On
the recorded base, the candidate regression observes that a direct abort
removes request state but awaits `LoRARegistry.release()` zero times. At the
exact candidate commit, the direct abort schedules one release for the acquired
LoRA ID.

The additional change in `_handle_abort_finish_reason()` is justified rather
than unrelated: after the producer-side direct-abort release is added, the old
500/503 consumer fallback could release the same reference again. The candidate
limits that fallback to the exact `ReqState` it still owns. Candidate tests and
an independent reused-RID case verify this exactly-once ownership boundary.

This is a source fix with regression hardening, not a test-only change or an
unverified claim. No remaining counterexample was found within the original
contract.

## Independent evidence

- Failing before: the exact candidate regression failed on the recorded base
  with `Expected release to have been awaited once. Awaited 0 times.`
- Candidate focused cases: direct LoRA abort, 503 producer/consumer
  exactly-once behavior, consumer-owned 500 fallback, and non-LoRA abort all
  passed (4 tests).
- Candidate module: the complete RID cleanup test module passed (30 tests and
  3 subtests).
- Independent adversarial cases passed (2 tests): a duplicate direct abort
  releases once; and an old 500-error consumer cannot remove or release a new
  state that reused the same RID.
- `git diff --check` passed for the exact base-to-candidate diff.

Raw logs, JUnit output, the independent test, and measured import paths are in
`reports/j-886eba825bc5/evidence/`.

## Architecture and environment limits

The interpreter imported SGLang from `/job/repo/python/sglang` and
`tokenizer_manager.py` from the reviewed checkout. PyTorch was
`2.11.0+rocm7.2`; the assigned device was visible as AMD Instinct MI355X with
`gfx950:sramecc+:xnack-`.

No GPU execution is claimed as issue evidence. The defect is CPU-side Python
async registry bookkeeping, and the candidate changes no native, compiler, or
kernel source. Therefore no native rebuild was required or performed. A live
server with real model weights and a real LoRA adapter was not run, so model
architecture, semantic accuracy, adapter-eviction throughput, and distributed
behavior remain outside this review. The deterministic tests validate the
actual TokenizerManager source and coroutine calls, not a full serving workload.

