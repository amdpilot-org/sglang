# Independent review of PR 2310

Candidate: https://github.com/amdpilot-org/sglang/pull/2310 at `6fdbee617e969a0f65c788f41934ffbd15392b90`

Upstream issue: https://github.com/sgl-project/sglang/issues/32507

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2350

Recommendation: **accept**. The exact candidate fully resolves the original contract for arbitrary positive local-expert count vectors in the tested native path.

At recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, a fresh gfx950 rebuild reproduced `RuntimeError: unexpected N` for the issue's N=16 vector. At the exact candidate, a separate fresh rebuild copied N=16 exactly. Independent boundary cases N=1, 15, 17, 31, 33, 511, 512, 513, 1024, 1025, and 4097 also matched deterministic CPU int32 references exactly. Empty input and malformed dtype, shape, contiguity, size, output dtype, and device cases were rejected.

The production `copy_list_to_gpu_no_ce` wrapper passes the complete list directly and contains no 512-element invariant. The candidate's chunked fallback therefore addresses the concrete N=513/N=1024 counterexamples identified by PR 2218, rather than merely adding N=16 or hardening tests.

The checked-in pytest could not collect because this ROCm checkout has no architecture-specific `common_ops` library. Review validation instead compiled and directly loaded the checked-out `copy.cu` via PyTorch's HIP extension flow; the log records two translated kernel launches, zero unsupported CUDA calls, and `hipcc --offload-arch=gfx950`. Raw logs and the probe are retained outside the revision checkout at `/job/review-evidence-j-539eef7324e8/`.

Limitations: the assigned environment is one AMD Instinct MI350X (gfx950) with ROCm 7.2. It lacks NVIDIA H800/CUDA, GLM-5.2 weights, and the two-node TP16/EP16 DeepEP deployment. Thus the native transfer contract is verified on gfx950, but NVIDIA compilation and the full distributed serving workload remain unverified.
