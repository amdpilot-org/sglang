# Independent review of PR 2312

Upstream issue: https://github.com/sgl-project/sglang/issues/31178

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2347

Candidate: https://github.com/amdpilot-org/sglang/pull/2312 at exact commit
`04848e4fa8f9196f9c20183d7f6b0dd805040528`.

## Verdict

Recommendation: **accept** as test-only hardening. The candidate does not add a
production fix; the recorded base already contains the relevant width-admission
guard. The candidate accurately adds a regression for that guard, and its prose
appropriately limits the GPU evidence to the tensor-shape mechanism.

The review does not mark the original deployment issue fully resolved because
GLM-5.2-FP8 weights and the reported TP16/two-node PD-disaggregated Mooncake
topology were unavailable. No counterexample was found within the guard's
direct admission contract.

## Evidence

The prepared checkout began at the required base
`358c163250ad3b1f62939b01ce1314a0a31a0365`, matching `REPOSITORY.md`; there was
no image-prepared revision difference. Base tests passed 3/3. After temporarily
checking out the exact candidate, its focused suite passed 6/6.

Removing only the existing `num_tokens_per_req` versus `captured_req_width`
guard made the candidate's mismatch test fail: width 2 was admitted to a graph
captured at width 1. The guard was restored and the candidate tree was checked
clean before returning to the delivery branch.

An independent matrix covered unset and non-positive legacy widths, multiple
positive mismatches, a different capture width, padded and non-padded graph
admission, TP-gather sizing, and MLP-sync gating. All results matched the
intended contract.

On the assigned AMD Instinct MI355X (`gfx950`), Torch/ROCm reproduced the exact
reported error when copying a `[2, 2048]` tensor into `[1, 2048]`. The matching
shape copied successfully with checksum `2096128`, independently equal to
`sum(range(2048))`.

Python loaded SGLang and the runner directly from `/job/repo/python`. The
candidate changes only Python tests and report files; it changes no native
source, so a native rebuild was neither required nor applicable. Raw logs are
preserved outside the checkout in
`/tmp/amdpilot-repo-j-6ed1ffc67988/evidence/`.

Detailed commands, exit codes, limitations, and structured review fields are in
`result.json`.
