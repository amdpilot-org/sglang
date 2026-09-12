# Independent review of amdpilot-org/sglang PR 2453

Candidate commit: `8b1804ecee27f2c877529f501f53d79a5c8ef1fc`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Upstream issue: https://github.com/sgl-project/sglang/issues/30314

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2388

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2485

## Recommendation

Request changes. The candidate is a useful partial mitigation: it enables by default an already-present, upstream-merged path that makes the redundant matched-prefix Mamba state evictable during decode. The source issue's independent reporter found that opt-in path discriminating, and a deterministic allocation fixture confirms that an exhausted 2x pool can reclaim a decode-evictable state.

It does not fully resolve the original contract. The reported failure family includes scheduler non-progress when the Mamba pool cannot satisfy admission. The candidate changes only an environment-variable default and comments; it does not bound eviction/admission retries, reject an unsatisfiable request, or make no-victim exhaustion observable. In an independent boundary probe at the candidate commit, the same 2x pool still failed with `Can not alloc mamba cache` when its prefix states remained admission-locked. This is relevant to the issue's post-flush/no-running and request-larger-than-pool reports, which are not explained or covered by decode-lock release.

The candidate regression proves that a Boolean default changed. Its existing fixture tests the underlying eviction mechanism, but there is no serving-path regression proving scheduler responsiveness or request termination under an unsatisfiable Mamba allocation. Accordingly this is a partial fix, not a full original-issue fix.

## Evidence

- The prepared checkout was exactly the recorded base before testing.
- Base source imported from `/job/repo/python/sglang/...`; its default and effective value were both `False`.
- At the exact candidate commit, source imported from `/job/repo/python/sglang/...`; its default and effective value were both `True`. A real subprocess with `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=0` produced effective `False`, confirming the rollback behavior.
- Candidate focused suites passed: 10 Mamba ratio/lock cases, 9 streaming-session cases, and 17 allocation-aware eviction cases.
- Independent boundary probe at the candidate commit:
  - decode-evictable, pool `2*N`: allocation succeeded after reclaiming one prefix state;
  - admission-locked, pool `2*N`: allocation failed after an eviction attempt;
  - admission-locked, pool `3*N`: allocation succeeded due to headroom.
- `git diff --check base..candidate` failed because the committed candidate report contains trailing whitespace in `reports/j-a636cc8b00d1/streaming-session.log`. This is report hygiene, not the substantive rejection reason.

Raw command output and fetched issue/PR snapshots were preserved outside the checkout under `/job/review-evidence/` while revisions were switched.

## Architecture and environment limitations

The available machine has one AMD Instinct MI350X reported by ROCm as `gfx950:sramecc+:xnack-`, with PyTorch `2.11.0+rocm7.2` and HIP `7.2.26015`. The original workload requires eight H100 80GB GPUs, TP=8, unavailable Qwen3.5-397B-A17B-FP8 weights, 100K+ token traffic, EAGLE, and direct hierarchical-cache I/O. Therefore the production hang, TTFT, health-check behavior, CUDA architecture, and multi-GPU behavior were not reproduced. GPU availability was observed, but GPU execution is not claimed as issue evidence because the changed policy is CPU-side and no qualified model fixture covers this hybrid-Mamba contract. No native source changed, so a native rebuild was not applicable.
