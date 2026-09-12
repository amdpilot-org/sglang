# Independent review of PR 2153

Reviewed candidate commit `a69473368a934197e72bbf7b0794e78316c2e5ee`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Upstream issue: https://github.com/sgl-project/sglang/issues/32276

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2106

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2192

## Recommendation

Request changes. The candidate is a real partial fix: on the exact candidate,
`thinking: {"type":"disabled"}` is retained by `ChatCompletionRequest` and
mapped to `chat_template_kwargs.thinking_mode="disabled"`. The focused four
candidate tests and the complete protocol test module pass.

It does not fully implement the declared MiniMax API contract:

1. The candidate maps official `thinking.type="adaptive"` to
   `thinking_mode="enabled"`. MiniMax-M3's actual template has separate
   `adaptive` and `enabled` branches. At Hugging Face model revision
   `f0e1c1e04d40177e4673a22097036854f536e9c0`, `adaptive` emits no forced
   prefix and lets the model decide, while `enabled` emits `<mm:think>` and
   requires reasoning. The candidate's adaptive regression asserts the wrong
   behavior rather than protecting the upstream contract.
2. The original report's independent `reasoning_effort="none"` counterexample
   remains. It produces only `thinking=false` and `enable_thinking=false`;
   MiniMax-M3's serving path reads `thinking_mode`, so the request still does
   not disable M3 reasoning.

## Evidence

- Base source import: `/job/repo/python/sglang/srt/entrypoints/openai/protocol.py`.
- On the recorded base, the official `thinking` object is silently discarded
  and `chat_template_kwargs` remains null.
- On the candidate, disabled maps to `{"thinking_mode":"disabled"}` and
  adaptive maps to `{"thinking_mode":"enabled"}`.
- The downloaded model template was kept outside the checkout at
  `/tmp/amdpilot-repo-j-83aa64c23885/minimax-m3-chat-template.jinja`; response
  headers identifying its model revision are in the external review evidence.
- Full external evidence is under
  `/job/review-evidence-j-83aa64c23885/`, preserved across revision switches.

## Limitations

The environment has one AMD Instinct MI355X (`gfx950:sramecc+:xnack-`) with
ROCm 7.2 and Torch 2.11.0+rocm7.2. MiniMax-M3 weights and the reported TP=4
topology were unavailable, so no model-semantic or distributed serving run was
claimed. GPU availability was inspected but no GPU inference was executed.
The candidate changes only Python protocol/tests/reports; there is no native
source change and no native rebuild was required or performed.
