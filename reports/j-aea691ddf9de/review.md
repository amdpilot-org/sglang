# Independent review of PR 583

Candidate: `b26d4ce2502264cf0309d3e90df308b3038eec81`

Recommendation: accept. The candidate fully resolves the two behaviors in the original issue at the request-conversion boundary.

On base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the independent probe observed `xhigh_mapping max` and generic HTTP 500 responses for a template-validation `ValueError` in both streaming modes. On the exact candidate, it observed `xhigh_mapping xhigh` and descriptive HTTP 400 `invalid_request_error` responses in both modes. Independent `TypeError` and `RuntimeError` cases remained HTTP 500, so unrelated failures were not reclassified by the new handler.

Source inspection also confirmed that the real OpenAI chat-template path catches Jinja template client errors and re-raises them as `ValueError` before control returns to the Anthropic handler. Thus the candidate's caught exception matches the current production propagation path rather than only its test fake.

The full Anthropic serving unit suite passed: 62 tests and 7 subtests. Imports resolved to `/job/repo/python/sglang/...` using the prepared interpreter.

No native source changed, so no native rebuild was applicable. No GPU inference was run. The assigned environment exposes one AMD Instinct MI355X with ROCm 7.2, not the issue's RTX PRO 6000 Blackwell/CUDA 13.0.3 and unavailable Qwen3.8-Flash-Next-NVFP4 model. The defect is entirely before generation, so the review makes no numerical GPU-output claim.

Raw evidence is retained outside the checkout under `/job/review-evidence-j-aea691ddf9de/` while revisions were switched.

Upstream issue: https://github.com/sgl-project/sglang/issues/36741

Mirror issue: https://github.com/amdpilot-org/sglang/issues/588
