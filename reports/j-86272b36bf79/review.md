# Independent review of PR 2004

Candidate: https://github.com/amdpilot-org/sglang/pull/2004  
Exact candidate commit: `8d70020a737b6a506a1038d2d6ae32d5b600e310`  
Upstream issue: https://github.com/sgl-project/sglang/issues/33199  
Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/1932  
Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2047

## Verdict

Recommendation: **accept**. The candidate fully resolves the original source-level
argument-resolution defect for both supported DeepSeek-V4 speculative algorithms,
EAGLE and DSPARK.

On the recorded base, an omitted `max_running_requests` first received the
DeepSeek-V4 model default of 256, and the later speculative hook failed to replace
it. At the exact candidate commit, the later hook consults the preserved raw
operator input. Omitted input therefore produces a declaration chain of 256 then
48, while explicit values retain precedence.

This is a source-level, pre-engine configuration bug. No model weights, serving
process, GPU kernels, or multi-GPU topology are required to exercise the reported
contract.

## Independent evidence

- The prepared branch was exactly the recorded base
  `358c163250ad3b1f62939b01ce1314a0a31a0365`; there was no difference between
  the image-prepared checkout and the required failing-before revision.
- Imports resolved to `/job/repo/python/sglang/__init__.py` using the required
  `/tmp/amdpilot-repo-j-86272b36bf79/venv/bin/python` interpreter.
- On the base, omitted EAGLE and DSPARK inputs both resolved to 256; explicit 96
  remained 96.
- At the candidate commit, its focused regression passed: 3 tests plus 2
  subtests.
- The resolution declaration and raw-input snapshot suites passed together with
  the candidate regression: 20 tests plus 25 subtests.
- An independent matrix checked EAGLE and DSPARK with omitted input and explicit
  values 1, 48, 96, and 256. Omitted input resolved to 48 after the earlier 256
  declaration; every explicit value remained unchanged.
- A deliberately incomplete legacy fixture without `_raw_input` remains at 256.
  This is not a production counterexample: the real resolution pipeline creates
  `_raw_input` from every `ServerArgs` field before handlers run, and the passing
  raw-input invariant suite pins that behavior.
- `git diff --check` passed for the candidate.

## Source and native paths

The only executable source change is
`python/sglang/srt/arg_groups/speculative_hook.py`; the regression is
`test/registered/unit/server_args/test_deepseek_v4_speculative_max_running_requests.py`.
No C, C++, HIP, CUDA, FlyDSL, extension, or other native source changed, so no
native rebuild was applicable. The prepared environment did load its existing
AITer module from
`/tmp/amdpilot-repo-j-86272b36bf79/cache/aiter/module_aiter_core.so`, but the
review did not use that module as evidence for the fix.

## Environment and limitations

The host reports a `gfx950` device and Torch `2.11.0+rocm7.2` with HIP 7.2.
No GPU code was executed because the original failure and fix occur entirely in
deterministic argument processing before engine startup. DeepSeek-V4 weights were
not available, so no full model-serving or multi-GPU run was attempted or
claimed. This does not leave the original source-level contract unverified.

No remaining counterexample tied to the original contract was found.
