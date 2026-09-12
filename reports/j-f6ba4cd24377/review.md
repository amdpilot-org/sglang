# Independent review of PR 896

Candidate: https://github.com/amdpilot-org/sglang/pull/896 at exact commit
`a007eace504e630a2bf4666fa32a78b30db0b282`

Upstream issue: https://github.com/sgl-project/sglang/issues/39173

Mirror issue: https://github.com/amdpilot-org/sglang/issues/931

## Verdict

Recommendation: **request changes**. The candidate contains two valid partial
corrections, but it does not fully resolve the original Engram compact-ragged
contract.

On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the actual layout helper
reproduced the reported 42-token capture geometry: eight slots with
`[6, 6, 5, 5, 5, 5, 5, 5]`. The base DSV4 raw verify and decode metadata
copies also raised the reported overlapping-storage `RuntimeError` on an AMD
Instinct MI355X.

At the exact candidate commit, the 42-token tier becomes seven uniform rows,
and both metadata copy classes safely handle overlapping GPU views. The
candidate's focused tests pass (14 tests and 8 subtests), and independent
boundary checks confirm its helper outputs.

The original issue is nevertheless not fully fixed. Non-multiple tiers remain
genuinely ragged:

- 41 tokens -> `[6, 6, 6, 6, 6, 6, 5]`
- 43 tokens -> `[6, 6, 6, 5, 5, 5, 5, 5]`
- 47 tokens -> `[6, 6, 6, 6, 6, 6, 6, 5]`

All violate the issue-described Engram equal-block assumption
`num_tokens == request_count * 6`. The prepared base and candidate contain no
`EngramHasher`, `python/sglang/srt/layers/engram.py`, or DeepSeek-V4.1 Engram
integration. Consequently the required layout-aware Engram behavior,
per-request hash context, and numerical hash outputs cannot be tested or fixed
from this source tree. The candidate's added tests document these ragged tiers
but do not exercise an Engram consumer, so that portion is test-only hardening,
not proof of the original contract.

The DSV4 overlap correction validates the isolated mechanism from the later
first-forward traceback. It was not exercised through a DeepSeek-V4.1-Flash
request, so the full first-forward claim remains unverified.

## Environment and limitations

Tests used `/tmp/amdpilot-repo-j-f6ba4cd24377/venv/bin/python`, importing
SGLang from `/job/repo/python/sglang`, with PyTorch 2.11.0+rocm7.2 and HIP
7.2.26015 on one AMD Instinct MI355X. The reported four-node NVIDIA GB10
TP4/EP4 CUDA deployment and model weights are unavailable. No native C++,
FlyDSL, or LLVM source changed, so a native rebuild was not applicable.

Raw command output is retained under `reports/j-f6ba4cd24377/raw/`.
