# Mooncake failed-session recovery correction generation 2

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/1285 at exact commit `e190107d29fdc776b6b0c9466b4561bc6d579ac9`

Independent review: https://github.com/amdpilot-org/sglang/pull/1345

Upstream issue: https://github.com/sgl-project/sglang/issues/37022

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1376

The candidate's valid changes are preserved: failed-session probing is enabled by default, can be explicitly disabled, forwards through the Mooncake wrapper, tolerates probe exceptions, and rejects a successful probe when the failure count changed while it was in flight.

The review's ABA counterexample was reproduced independently on the exact candidate. A probe snapshots failure count 1; registration clears that counter; a subsequent failure recreates count 1; the old successful probe then clears the new failed state. See `raw/exact_candidate_aba.log`.

The correction adds a monotonic per-session registration generation. Registration advances it while clearing the old failed state, and probe recovery now requires both the registration generation and failure count to match the snapshot. The regression coordinates the interleaving with events and verifies the newly failed registration remains blacklisted. It failed before the correction and passes afterward. A boundary test verifies registration advances the generation and clears the prior failure state.

## Reproduce

```bash
/tmp/amdpilot-repo-j-7cea7af9337b/venv/bin/python -m pytest -q test/registered/unit/disaggregation/test_mooncake_transfer_batching.py
/tmp/amdpilot-repo-j-7cea7af9337b/venv/bin/python -m compileall -q python/sglang/srt/disaggregation/mooncake/conn.py test/registered/unit/disaggregation/test_mooncake_transfer_batching.py
git diff --check 358c163250ad3b1f62939b01ce1314a0a31a0365
```

Final result: 14 passed and 3 subtests passed. No native source changed, so no native rebuild was applicable.

## Remaining limitation

This is a justified control-plane recovery correction, not a diagnosis of the recurring transfers themselves. The reported 5P3D GLM5.2 NVIDIA/XCCL multi-node RoCEv2 deployment, model weights, NVIDIA GPUs, XCCL stack, and network topology were unavailable. The prepared host exposes one AMD `gfx950` GPU; it was inspected but not used as substitute evidence. Consequently, the underlying cause of the deployment's recurring transfer failures remains unidentified and uncorrected.
