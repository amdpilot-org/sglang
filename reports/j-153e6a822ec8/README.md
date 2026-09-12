# Independent review of PR 1721

Reviewed exact candidate commit `38c4cbff0291764be04514426e2654e2e08d3b9f`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **request changes**. The source change is a valid narrow fix for
an independently reproduced cleanup-before-normalization exception, but it does
not fully resolve the original issue's public `/abort_request` contract. In this
checkout the endpoint directly calls `TokenizerManager.abort_request`; it never
calls `create_abort_task`, the only function changed by the candidate. Moreover,
the same real GPU/HTTP probe cleanly aborted the stream on both base and
candidate, so the endpoint result cannot be attributed to this patch.

## Evidence

- Base, unnormalized cleanup: `GenerateReqInput` had no `is_single`, and running
  its background task raised the reported `AttributeError` at
  `tokenizer_manager.py:2191`.
- Candidate regression: `3 passed, 26 deselected` for `TestCreateAbortTask`.
- Independent candidate cases passed for an unstarted single request, an active
  unnormalized single request, a mixed active/missing batch, and an empty batch.
- Base and candidate HTTP probes each started a real server on the assigned AMD
  Instinct MI355X (`gfx950`), generated tokens with the qualified deterministic
  tiny Llama fixture, received HTTP 200 from `/abort_request`, and ended the
  stream with `finish_reason.type == "abort"`. Neither server log contained the
  reported exception.
- Imported `sglang.srt.managers.tokenizer_manager` from the checked-out source at
  `/job/repo/python/sglang/srt/managers/tokenizer_manager.py`. Torch came from
  `/opt/venv/lib/python3.12/site-packages/torch`, version `2.11.0+rocm7.2` with
  HIP `7.2.26015`.
- No native source changed, so no native rebuild was applicable.

Raw runtime evidence was preserved outside revision switches under
`/tmp/amdpilot-repo-j-153e6a822ec8/review-evidence/`. The tiny fixture weights
were also kept outside the worktree; SHA-256:
`6632fab7c351a0bd85518d80a89ec673f68139587f5ad3fa8d5ab38fad87e75f`.

## Limitations

The original `meta-llama/Llama-3.2-1B-Instruct` weights and RTX 4090/CUDA
environment were unavailable. Testing used one gfx950 GPU with ROCm 7.2 and a
tiny random Llama. That qualifies transport and engine execution only, not model
semantics, the original CUDA architecture, or distributed execution. Both owned
server process groups required SIGKILL after the runner's 30-second graceful
shutdown window; descendants were reaped.

