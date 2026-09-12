# Independent review of amdpilot-org/sglang PR 1386

Candidate reviewed: `672567d14035030cc18c6075f5c8c5cee8d5efd5`

Upstream issue: https://github.com/sgl-project/sglang/issues/35642

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1420

Recommendation: **accept**. The candidate fully resolves the original issue in its stated scheduler-control-flow scope.

At the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the candidate regression failed in both exhausted-budget cases: the waiting queue was iterated for admission after the resumed chunk changed the budget state to `OTHER` or `NO_TOKEN`. The `CONTINUE` boundary passed. This reproduces the original within-pass defect without model weights or a GPU.

At the exact candidate commit, the focused regression and existing `PrefillAdder` suite passed (31 tests plus 12 subtests). An independent adversarial harness used an actual waiting-request mock and verified that `init_next_round_input`, the cache matching/load-back entry path described by the issue, was not called after a resumed chunk exhausted the budget. It also verified that the path is still called when the budget remains `CONTINUE`, and when there is no resumed chunk. With a Mamba allocator mock, the skipped scan used `alloc_group_begin(0)` and still called `alloc_group_end()`.

The production change is narrow: after `add_chunked_req`, any non-`CONTINUE` budget state replaces only the admission iterable with an empty tuple. It does not return early, so the resumed chunk proceeds into the new batch and later bookkeeping remains intact. The allocator's expected group size is changed consistently to the selected iterable length.

Source imports were confirmed from `/job/repo/python/sglang/srt/managers/scheduler.py` and `/job/repo/python/sglang/srt/managers/schedule_policy.py` while detached at the candidate commit. The candidate changes Python and test/report files only; no native source or build definition changed, so native rebuilding was not applicable. The prepared environment imported an existing AITer cache library from `/tmp/amdpilot-repo-j-bd15c0f82951/cache/aiter/module_aiter_core.so`, but it is unrelated to this Python scheduler change.

Raw logs and numeric exit codes were preserved outside the revision-switching checkout in `/job/review-evidence-j-bd15c0f82951/`.

## Limitations

No GPU, model-serving, full-model, or distributed run was performed. The original issue explicitly identifies a deterministic scheduler-level control-flow defect requiring no model or device, and those larger runs would not strengthen the relevant contract. Accordingly, `gpu_execution` is false and no claim is made about model semantics or distributed behavior. The prepared host reported ROCm 7.2 / Torch 2.11.0+rocm7.2 and an AITer NUMA warning during imports; neither affected the CPU regression.
