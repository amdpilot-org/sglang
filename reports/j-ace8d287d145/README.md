# Independent review of PR 1117

Upstream issue: https://github.com/sgl-project/sglang/issues/36894

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1154

Candidate reviewed: https://github.com/amdpilot-org/sglang/pull/1117 at exact commit `974cd3283b0360c7f2c71b044edbe3438c3d13b9`.

Recommendation: **accept**, with the original Qwen3.5 HTTP/model scenario still explicitly unverified.

The prepared checkout initially matched recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`. On that base, the candidate's regression failed with five tokenizer-side LoRA releases after four acquisitions for one `n=4` prompt and eight releases after six acquisitions for two `n=3` prompts. An independent test drove the checked-out `_handle_batch_request` together with the real `LoRARegistry` and `ConcurrentCounter`: `wait_for_unload()` timed out after the reported three-request `n=16` traffic shape and after a mixed adapter-A/no-adapter/adapter-B batch.

At the exact candidate commit, the submitted regression passed. The independent tests also passed: all three `n=16` lifecycles reached zero and unloaded, both adapters in the mixed batch reached zero, and scheduler-bound prefix warm-up requests retained their adapter IDs. The change clears LoRA ownership only from the tokenizer-side completion state of the extra zero-token warm-up request, preventing the unacquired decrement while retaining scheduler/cache adapter identity.

No native source changed, so no native rebuild was applicable. Imports resolved to `/job/repo/python/sglang`. The prepared environment used Python from `/tmp/amdpilot-repo-j-ace8d287d145/venv`, Torch 2.11.0+rocm7.2, ROCm 7.2, and one AMD Instinct MI350X (`gfx950`). GPU execution was checked only as environment evidence; the defect and fix are tokenizer-manager lifecycle accounting and do not require a GPU kernel.

The reported Qwen3.5-0.8B weights and rank-128 LoRA adapter were unavailable. Therefore no full HTTP serving reproduction on that hybrid architecture was performed, and this review does not claim equivalence to either reported CUDA 13.1 deployment. No deterministic counterexample to the candidate was found within the directly affected lifecycle contract.
