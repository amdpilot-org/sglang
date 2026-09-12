# Independent review of PR 2504

Candidate: https://github.com/amdpilot-org/sglang/pull/2504 at
`81bdd3011b40fea2c36354e3ae6f5b5d94df4189`

Upstream issue: https://github.com/sgl-project/sglang/issues/26794

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2476

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2533

## Finding

Recommendation: **request changes**.

The candidate is test-only hardening for a production correction already in the
recorded base. The focused test passes and independent reported-size and TP=2
loader checks pass. Source inspection confirms that current NPU unquantized MoE
post-processing retains canonical `w13_weight`/`w2_weight` shapes and the NPU
grouped matmul consumes them without a persistent transpose. This addresses the
shape invariant implicated by the issue.

The original DeepSeekV3.2 `update_weights_from_disk` failure could not be
reproduced on the required base because that base already includes the fixes
from upstream PRs #26717 and #29503. The historical persistent-transpose layout
does fail when followed by the current separate-projection loader, but this is a
synthetic failing-before reconstruction rather than an execution of the
reported Ascend scheduler path.

The requested change is to correct the candidate's retained GPU evidence. Its
raw output and `result.json` identify the assigned device as “AMD Instinct
MI350X”; the prepared runtime reports “AMD Instinct MI355X” with
`gfx950:sramecc+:xnack-`. Because architecture reporting is an explicit review
requirement, the committed claim is inaccurate even though both are gfx950.

## Scope and limitations

- No Ascend NPU, CANN runtime, `torch_npu`, or reported DeepSeekV3.2 checkpoint
  was available. NPU format casting, grouped matmul execution, scheduler
  survival, and the end-to-end HTTP reload remain unverified independently.
- The assigned AMD GPU validates tensor loading only. It cannot qualify an
  Ascend-specific kernel or model semantics.
- No native source changed in the candidate, so no native rebuild was required.
- The source and imported module both resolved under `/job/repo/python/sglang`;
  tests used the interpreter prescribed by `REPOSITORY.md`.

Commands and measured outcomes are recorded in `result.json` and `raw/`.
