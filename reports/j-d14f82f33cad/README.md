# MiMo-V2.5-Pro-W8A8 candidate correction review

Upstream issue: https://github.com/sgl-project/sglang/issues/37755

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1089

Candidate PR: https://github.com/amdpilot-org/sglang/pull/971

Independent review PR: https://github.com/amdpilot-org/sglang/pull/1058

## Outcome

The review counterexample is reproducible: all six tests added by candidate commit
`2e9ee6b921a55734331216fef05ff3390b9073ca` pass against the recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`, and also pass at the exact
candidate commit. The candidate therefore adds useful synthetic boundary
coverage but no failing-before/passing-after evidence for the reported bug.

No production correction is justified from the available environment. The
reported ModelSlim checkpoint is absent, as are Ascend/CANN and the two-node
TP32/DP4 topology. Consequently, real checkpoint tensor metadata, the NPU load
path, generated tokens, and semantic accuracy cannot be inspected. The
candidate's test coverage is preserved without claiming that it resolves the
open issue.

Exact commands and outputs are retained in `raw/reproduction.txt`.
