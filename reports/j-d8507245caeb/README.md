# Multimodal preprocessing admission verification

This contribution ports and verifies the complete implementation from the current related upstream candidate, [sgl-project/sglang#38036](https://github.com/sgl-project/sglang/pull/38036), against the prepared base rather than duplicating a narrower solution.

The opt-in `--max-mm-preprocessing-inflight-items-per-worker` budget counts image, video, and audio leaves, admits without queuing retained payloads, returns HTTP 400 when a request can never fit, and returns retryable HTTP 503 when current reservations fill the budget. Reservations cover media I/O, processor work, shared-memory/CUDA-VMM preparation, and scheduler handoff. Executor futures retain ownership after coroutine cancellation until native CPU work actually completes.

Raw logs are retained under `/tmp/amdpilot-repo-j-d8507245caeb/{before,after}`. The exact commands and outcomes are recorded in `result.json`.

The original Kimi-K2.6 workload and ASGI pre-parse memory were not qualified. This feature bounds the tokenizer-side preprocessing pipeline after request parsing; it is not a request-body streaming limit.
