# Independent review of PR 3366

Upstream issue: https://github.com/sgl-project/sglang/issues/36395

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3333

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3369

Candidate: https://github.com/amdpilot-org/sglang/pull/3366 at
`b663c98b8df05af0b927a19e1cb0c40d33ce8bab`.

## Recommendation

Accept. The candidate is a source-level full fix for the original initialization
contract. On the recorded base, correctness-mode and direct `load_model` calls
observe `auto` instead of explicit `flashinfer_mxfp4` or `aiter`. The candidate
initializes the published MoE configuration at the shared model-loading boundary,
before `ModelConfig` and `ModelRunner`, and all candidate and independent tests pass.

The candidate test fails 1/3 on the recorded base and passes 3/3 on the exact
candidate. An independent five-case suite fails 4/5 on the base and passes 5/5 on
the candidate. It covers the exact reported backend, direct loading, correctness
mode, the `auto` boundary, and replacing a prior published backend.

## Environment and limitations

Imports were confirmed from `/job/repo/python/sglang`, including the candidate's
`one_batch.py`; no installed wheel shadowed the reviewed Python source. The patch
contains no native/C++ changes, so no native rebuild was required.

The assigned GPU is AMD Instinct MI355X (`gfx950`) with ROCm 7.2, not NVIDIA
GB300/SM103. Kimi-K3 MXFP4 weights and the NVIDIA trtllm-gen implementation were
not available. Therefore this review did not execute the reported checkpoint or
independently confirm NVIDIA kernel tables, BF16 materialization, model semantics,
or distributed behavior. Those are environment limitations, not remaining
source-level counterexamples to the initialization fix.

Raw commands/results and the reviewed diff are in `raw/`.
