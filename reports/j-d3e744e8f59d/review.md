# Independent review of candidate f8502d20

Upstream issue: https://github.com/sgl-project/sglang/issues/34675

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1655

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1617

Recommendation: **accept**. The candidate fully resolves the original request-parsing and forwarding defect at the source boundary that can be deterministically exercised without a distributed model deployment.

## Evidence

The prepared checkout was clean on `amdpilot/j-d3e744e8f59d` at the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`; there was no difference from the requested failing-before revision. With `PYTHONPATH=/job/repo/python` and the prepared interpreter, constructing `ResponsesRequest` from a body containing `data_parallel_rank=4` discarded the injected field. Neither the deprecated field nor canonical `routed_dp_rank` existed after validation, and the contract assertion failed.

The exact candidate `f8502d20d3039acd58b32a60f901bb090802a7d1` is one commit directly atop that base. Its focused Responses API suites passed: 61 tests and 2 subtests. Independent cases additionally confirmed that a legacy body rank reaches `GenerateReqInput`, the canonical body field wins over the deprecated alias, an `X-Data-Parallel-Rank: 0` header overrides a nonzero body rank, body rank zero is preserved, and a non-integer header yields HTTP 400. Source inspection also confirms the effective rank is copied when a built-in-tool continuation creates a replacement `GenerateReqInput`.

The implementation follows the existing Completions and Chat Completions contract: both request-body spellings are accepted, `data_parallel_rank` migrates to `routed_dp_rank` without overriding an explicitly supplied canonical value, and the header has final precedence.

## Scope and limitations

The candidate changes no native C++, CUDA, HIP, or FlyDSL files, so no native rebuild was required or performed. SGLang imports resolved to `/job/repo/python`, not an installed wheel.

The host exposed one AMD Instinct MI350X (`gfx950:sramecc+:xnack-`) through Torch 2.11.0+rocm7.2 / HIP 7.2.26015. No GPU model execution was performed because it would not validate this CPU-side schema/adapter defect. No model weights, multi-rank DP server, distributed scheduler, or multi-node workload were available or claimed; the verified boundary is the real `ResponsesRequest` parsing and serving conversion into the `GenerateReqInput` handed toward the scheduler.

Complete revision-independent evidence, including the candidate diff, issue/PR metadata, base failure output, and independent adversarial script, remains under `/job/review-evidence-j-d3e744e8f59d` while the repository is back on the prepared review branch.
