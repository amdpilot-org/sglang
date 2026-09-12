# Independent review of PR 865

Upstream issue: https://github.com/sgl-project/sglang/issues/38104

Mirror issue: https://github.com/amdpilot-org/sglang/issues/898

Candidate: https://github.com/amdpilot-org/sglang/pull/865 at `01e371c94736e20b568185c1c947c03a69fd1c3a`

Recommendation: **accept**. The candidate fully resolves the original request-versus-server-default precedence defect at the chat-template render boundary.

The prepared checkout exactly matched the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`. Running the candidate's three focused regression cases against that untouched base produced two expected failures: both nested `chat_template_kwargs.reasoning_effort=xhigh` and top-level `reasoning_effort=xhigh` rendered as the server default `medium`. The silent-request/default case passed.

At the exact candidate commit, all three regression cases passed. The complete `test_serving_chat.py` file also passed (138 tests and 68 subtests). Independent cases confirmed explicit `none`, numeric `0.0`, and top-level `max` beat the server default; a silent request still inherited the default; unrelated request keys retained request-first precedence; and the established nested-request value continued to win when both request locations were supplied.

The imported implementation was `/job/repo/python/sglang/srt/entrypoints/openai/serving_chat.py`. The candidate changes Python, documentation, tests, and its prior report only; it changes no native code, so no native rebuild applies.

No model-serving/GPU execution was claimed. The available hardware is one AMD Instinct MI350X/gfx950 under ROCm 7.2, not the reported 2x NVIDIA H200/CUDA deployment, and Qwen3.8-Flash-Next-FP8 weights were not prepared. Consequently this review verifies the deterministic conversion/template-render contract, not full-model reasoning length, semantic quality, NVIDIA behavior, or TP/EP=2 behavior.
