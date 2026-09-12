# Qwen3 gateway reasoning correction generation 2

Upstream issue: https://github.com/sgl-project/sglang/issues/35148

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1831

Candidate parent: https://github.com/amdpilot-org/sglang/pull/1691 at `fac8ec3773db9c151b20afd6cff8133a9f64bea2`

Independent review parent: https://github.com/amdpilot-org/sglang/pull/1786

The candidate's valid HTTP non-streaming and gRPC changes are preserved. Independent inspection reproduced the remaining non-streaming defect: `parse_chat_reasoning_payload` skipped every non-null `reasoning_content`, so an empty string left `work</think>answer` untouched. The regression was added before the guard was corrected to treat null, missing, and empty-string reasoning as unpopulated while continuing to preserve non-empty worker reasoning.

HTTP streaming still returns the worker SSE body as a passthrough. No streaming rewrite is included: a correct implementation must frame SSE events across arbitrary transport chunk boundaries and retain parser state per choice. That larger behavior was not established by the original non-streaming curl reproduction and could not be honestly validated within this correction. It remains explicitly unverified rather than being represented as solved.

The Qwen3.8-27B-FP8 weights and reporter's NVIDIA L40S/CUDA environment were unavailable. No model-semantic or GPU claim is made.
