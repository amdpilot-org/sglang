# Issue 32378 investigation

Outcome: **candidate_rejected**. No product source was changed.

The report launches both roles with `python -m sglang.compile_deep_gemm`. In the
prepared implementation this is a finite precompilation command, not a serving
entrypoint: it starts a temporary server, sends one warmup request, prints
`DeepGEMM Kernels compilation finished successfully.`, waits ten seconds, and
kills the child process tree. Consequently, observing both processes exit after
startup is expected. The two supplied `multiprocessing.resource_tracker`
warnings are shutdown diagnostics and do not identify the preceding failure.

The exact upstream candidate, sgl-project/sglang#32444, remains open. It globally
replaces `multiprocessing.resource_tracker.register` with a no-op at module
import. That is not a safe issue-specific correction: it suppresses tracking for
resources the process genuinely owns. Existing SGLang workarounds scope the
override only around attachment to shared memory owned by another process.

## Evidence

- `evidence/compile-deep-gemm-tiny.log`: the real checked-out entrypoint loaded
  and executed the qualified tiny Llama on gfx950. It reached GPU prefill and
  completed its warmup. The run then encountered an unrelated local port bind
  collision and was terminated (exit 143); it is retained as negative evidence,
  not presented as an issue reproduction.
- `evidence/mooncake-transfer-unit.log`: five deterministic Mooncake transfer
  batching tests and three subtests passed.
- `evidence/mooncake-unit.log`: a broader selection completed the same five unit
  cases and then hung on environment-dependent HiCache/Mooncake integration; it
  was stopped and is not counted as passing.
- `evidence/gpu-boundary.log`: an independent FP16 GPU matrix multiplication
  passed against a CPU FP32 reference on one AMD Instinct MI350X (`gfx950`).

## Remaining limitations

The job has one AMD GPU and no GLM-5.2-W4AFP8 weights, eight-GPU TP/EP topology,
second host, or matching four-HCA RDMA fabric. It therefore cannot reproduce or
clear the reported GLM architecture, quantization, speculative decoding,
TP8/EP8, two-node PD, Mooncake RDMA, or HiCache combination. The original log
also omits the exception or traceback that caused shutdown before the cleanup
warnings. A qualified diagnosis requires that earlier log content and the
reported deployment prerequisites.

Upstream issue: https://github.com/sgl-project/sglang/issues/32378

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2094
