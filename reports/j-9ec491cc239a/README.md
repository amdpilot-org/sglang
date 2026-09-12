# Independent review of PR 3456

Reviewed exact commit `cd287687e83f1b82e3d52b25f016e852ac3b63ba` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **accept as test-only hardening**, with `fully_resolves_original=false`.

The candidate changes only tests and its own prior report. On the assigned AMD Instinct MI350X, all three focused candidate tests passed. They exercised committed Full and MAMBA host state, device demotion, a discoverable host leaf, host locks and accounting, real load-back transfers, and exact restoration of KV plus MAMBA temporal/convolution tensors. The two negative cases cover missing MAMBA host state and MAMBA-only host state.

The required failing-before could not be produced at the recorded base: that revision already contains generalized host-leaf demotion and auxiliary MAMBA host-LRU handling, and 40 adjacent Full+MAMBA HiCache tests passed there. Therefore this review does not identify the historical fix and does not claim the test-only candidate repairs an affected revision.

The reported TP4 hybrid-KDA serving workload remains unverified. Only one MI350X was assigned and no Ling-3.0-flash/hybrid-KDA weights were available. The tiny Llama fixture is architecturally inapplicable, so it was not substituted for that requirement. No native code changed and no rebuild was applicable.

Raw command output is retained in `raw/`. The mutation run removed one redundant component-local host-LRU insertion; the positive test continued to pass because the tree demotion path separately performs the required insertion. The mutation was reverted before switching away from the candidate.

Upstream issue: https://github.com/sgl-project/sglang/issues/33713

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3441

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3459
