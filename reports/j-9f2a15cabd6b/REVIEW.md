# Independent review of PR 3243

Upstream issue: https://github.com/sgl-project/sglang/issues/16255

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3246

Candidate: https://github.com/amdpilot-org/sglang/pull/3243 at exact commit
`0565e838f452a2b82d05adf2b50b7817ee613bfa`.

## Recommendation

Request changes. The candidate adds the requested AMD and CPU mixin files and
removes the corresponding attention dispatch branches from `deepseek_v2.py`,
but it changes two AMD execution routes during that extraction:

- `MHA_ONE_SHOT_ROCM` previously called `forward_normal_one_shot_core`; the
  candidate calls `forward_normal_core`.
- `MHA_CHUNKED_KV_ROCM` previously called `forward_normal_chunked_kv_core`; the
  candidate calls `forward_normal_core`.

The candidate's regression test expects `forward_normal_core` for all three AMD
MHA variants, so its passing result does not prove preservation of the original
contract. An independent dispatch oracle based on the exact pre-refactor code
passes the ordinary ROCm route and fails the one-shot and chunked-KV routes.

## Reproduction and validation

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` lacks the proposed
`deepseek_common.hardware_backend` package, and the independent regression
therefore fails at collection with `ModuleNotFoundError`. At exact candidate
commit `0565e838f452a2b82d05adf2b50b7817ee613bfa`, source imports resolved from
`/job/repo/python`, not an installed wheel.

The candidate focused suite passed (`30 passed`, `8 subtests passed`), while the
independent route-preservation test failed (`2 failed, 1 passed`). The failures
are deterministic mock dispatch checks and do not require model weights.

No native source changed, so no native rebuild was applicable. The assigned
device was one AMD Instinct MI355X (`gfx950`, ROCm 7.2, Torch 2.11.0+rocm7.2).
The failing contract is reached before kernel execution; no GPU numerical claim
is made from the mock test.

## Scope and remaining limitations

This candidate is a partial structural refactor, not a full resolution of the
original issue. `deepseek_v2.py` remains 3,098 lines and still contains many
CPU, CUDA/ROCm, NPU, and MUSA conditionals outside attention dispatch. No real
DeepSeek V3.2 checkpoint exercised Indexer or IndexerKPool loading, logits,
serving, or semantic accuracy. Only one GPU was assigned, so no multi-GPU or
pipeline-parallel stage boundary crossed a real top-k handoff. NPU, NVIDIA,
CPU-only, and MUSA runtime paths were unavailable. These unavailable paths are
unverified, not evidence for or against speculative source changes.
