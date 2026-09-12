# Independent review of NCCL RAS candidate

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/2789 at `93773a6d3a67ed79e50d198c20f9964be234832e`

Upstream issue: https://github.com/sgl-project/sglang/issues/32928

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2816

Candidate provenance issue: https://github.com/amdpilot-org/sglang/issues/2734

## Recommendation

Request changes. The candidate is a substantial partial implementation, but it does not yet fully resolve the original runtime contract.

## Blocking finding

`RasSocketClient` uses a fixed one-second read timeout for the complete job-wide `STATUS` response. NVIDIA's reference documentation says that gathering communicator data may take several seconds, particularly for large jobs or jobs experiencing problems. NVIDIA's reference client starts with a short connection/handshake timeout but then expands the receive timeout to cover the RAS collective timeout and extra allowance before requesting status.

The independent adversarial server used the wire commands accepted by NCCL (`SET FORMAT json`, then `STATUS`) and returned valid JSON after 1.2 seconds. The candidate returned `None`, which the detector publishes as `poll_success=0`. Thus a slow but functioning RAS service is falsely reported as the candidate's single-node crash signal. This is directly tied to the original detection contract and likely becomes more common during the failure conditions being diagnosed.

The candidate's socket test uses a mock whose response is immediate, so the passing 34-test suite does not cover this behavior.

## What did pass

- At the exact recorded base, importing `sglang.srt.distributed.nccl_ras` fails because the feature is absent.
- At the exact candidate commit, imports resolve to `/job/repo/python`, and all 34 candidate tests plus 2 subtests pass.
- Parser, detector, metric wiring, log-only fallback, world-rank election, and scheduler lifecycle logic receive useful CPU coverage.
- The changed Python files compile under the pinned interpreter.
- No native source changed, so there was no native library to rebuild.

## Environment and unverified paths

The prepared node has one AMD Instinct MI355X with Torch 2.11.0+rocm7.2 and RCCL reporting NCCL API compatibility version 2.27.7. `localhost:28028` refused connections. NVIDIA NCCL 2.28.7+ RAS, a second node/GPU, the approximately 5-second `unresponsive` transition, the approximately 60-second `considered_dead` transition, and TP=2 live serving metrics could not be exercised. The tiny Llama fixture would not supply the missing NCCL RAS architecture and was not used as substitute proof.

Raw commands and outputs are retained in `reports/j-b47ffc385754/raw/`; the structured claim record is in `result.json`.
