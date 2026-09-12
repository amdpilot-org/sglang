# Independent review of PR 1451 at `7a71a1546b5b27eb420f37fe167ecbe2d76919d4`

Recommendation: **request changes**. The candidate is a partial fix.

The prepared checkout was exactly the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`. With `PYTHONPATH=/job/repo/python`, imports resolved to the checked-out `python/sglang/srt/entrypoints/openai/encoding_dsv4.py`; Torch was the prepared ROCm 7.2 build.

On the base, the issue's exact tool-call sequence with a final system reminder was accepted and had no assistant generation boundary. The user-tail control ended in `<｜Assistant｜><think>`.

At the exact candidate commit, the reported sequence raises `ValueError`, and `OpenAIServingBase.handle_request` maps that exception to an HTTP 400 response before inference. The candidate's 10 focused DSV4 tests pass, as do Python compilation and diff whitespace checks. Context-plus-new-message ordering and developer/assistant followed by system were independently checked and rejected.

The remaining counterexample is a system-only request. It is a valid OpenAI-shaped request whose final message has `role=system`; the candidate accepts it and renders only `<｜begin▁of▁sentence｜>a`, without `<｜Assistant｜><think>`. The actual serving preparation leaves a first system message in place, so this request can still reach tokenization/inference with the malformed generation prompt. The new validator only rejects system messages after a non-system message and does not enforce the broader invariant that every accepted generation request has an assistant boundary.

No native files changed, so no native rebuild was applicable. One AMD Instinct MI355X (gfx950) was visible, but DeepSeek-V4 weights were absent and the repository's affected model recipes require larger/multi-GPU configurations. No model-level or long-context semantic claim is made.

Raw command output was preserved outside the revision-sensitive checkout during switching in `/job/base-review-cases.log`, `/job/candidate-review-cases.log`, `/job/candidate-dsv4-tests.log`, `/job/candidate-compile.log`, `/job/base-import-paths.log`, and `/job/gpu-environment.log`.
