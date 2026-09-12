# Independent review of amdpilot-org/sglang PR 1101

Upstream issue: https://github.com/sgl-project/sglang/issues/37022

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1135

Candidate: https://github.com/amdpilot-org/sglang/pull/1101 at `0aeacd1b2bf8d0a0920c0a2beb6db2393d88311c`

## Recommendation

Request changes. The candidate changes only tests and reports; its production source is identical to the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`. Its nine-test suite also passes when the candidate test file is run against that base, proving that the candidate is test-only hardening of an already-present, opt-in recovery mechanism.

That mechanism is useful but does not fully resolve the original report. `SGLANG_ENABLE_FAILED_SESSION_PROBE` defaults to false, so the reported sticky `failed_sessions` behavior remains under default configuration. The original 5-prefill/3-decode GLM5.2 NVIDIA/XCCL RoCEv2 workload was unavailable, and the issue contains no minimal reproducer, so the underlying reason for the recurring Mooncake transfer failures was not reproduced or fixed.

An independent concurrency case also exposes a remaining state-machine bug. `_run_one_probe_pass` snapshots a failed session, probes without holding `session_lock`, and then unconditionally clears the session and its failure count on success. If a new transfer failure for the same session is recorded while that probe is in flight, the successful stale probe erases the newer failure. This counterexample fails identically on the base and candidate.

PR 1101 should describe itself as test-only coverage for an existing partial, opt-in mitigation and use an outcome consistent with that limited evidence. It should not label the candidate as verified against the original production issue.

## Evidence

- Candidate regression on exact candidate: 9 passed and 3 subtests passed.
- Candidate regression file executed against the recorded base: the same 9 passed and 3 subtests passed.
- Production source comparison: no differences under `python/sglang` between base and candidate.
- Independent adversarial case on candidate: exit 1; a concurrent newer failure was erased (`failed_sessions=[]`, failure count absent).
- Environment/default check: `SGLANG_ENABLE_FAILED_SESSION_PROBE` evaluated to `False` when unset.
- Source imports resolved to `/job/repo/python/sglang/...`; native Mooncake resolved to `/opt/venv/lib/python3.12/site-packages/mooncake/engine.cpython-312-x86_64-linux-gnu.so` and exposes `TransferEngine.send_probe`.

Raw logs and the adversarial program are retained outside the revision-switching checkout at `/job/review-evidence-j-0d3a83bf29e8/`.

## Architecture and environment limitations

The assigned system has one AMD Instinct MI355X-class `gfx950` device with PyTorch `2.11.0+rocm7.2`. The report concerns an eight-GPU NVIDIA CUDA/XCCL deployment using multi-node RoCEv2 RDMA and GLM5.2. No matching network fabric, node count, XCCL stack, or model weights were available. GPU execution would not validate this control-plane/distributed-network contract and was not used as substitute evidence. No native source changed in PR 1101, so no native rebuild was required; the installed native Mooncake extension's path and `send_probe` symbol were verified.
