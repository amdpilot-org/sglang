# Independent review of amdpilot-org/sglang PR 2122

- Candidate: https://github.com/amdpilot-org/sglang/pull/2122 at `df03b1321fc8edfb86fd14e950e9e47fe28385a6`
- Upstream issue: https://github.com/sgl-project/sglang/issues/32493
- Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2056
- Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2161
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Recommendation: **accept**
- Fully resolves the original routing defect: **true**

## Finding

The candidate is a full fix for the original issue's prompt-routing contract, not merely test hardening. On the recorded base, a multimodal-capable checkpoint's text-only request with existing prompt IDs routed as `("text", "rendered prompt")`; this reproduces the condition that causes Responses to encode once for its output budget and TokenizerManager to encode again. At the exact candidate commit, the shared router used by Chat Completions and Responses instead forwards the same ID list as `input_ids`.

Actual image, video, and audio payloads retain the text path for standard VLM processors. Empty IDs retain the text fallback, raw string IDs retain the explicit string fallback, and token-first Inkling/Kimi K3 encoders retain their IDs for media requests. An independent 100,000-ID case verified that the candidate returns the original list object without encoding, copying, or converting it.

The candidate changes only Python serving code and Python tests. There are no native source changes, so no native rebuild applies. Runtime imports resolved to `/job/repo/python/sglang/...`, proving the checked-out source—not an installed SGLang wheel—was exercised. Torch resolved to the prepared ROCm installation.

## Evidence

- `raw/base-routing.log`: failing-before reproduction on the exact recorded base. The issue-specific text-only multimodal case fails; real-media and text-model boundaries pass.
- `raw/candidate-routing-tests.log`: candidate's five focused routing regressions pass.
- `raw/candidate-responses-tests.log`: Responses endpoint routing and media boundaries pass.
- `raw/candidate-chat-full.log`: all 140 tests in `test_serving_chat.py` pass.
- `raw/candidate-responses-full.log`: all 36 tests in `test_serving_responses.py` pass.
- `raw/candidate-independent-adversarial.log`: independent long-ID, media-type, empty-ID, raw-string, token-first, and text-checkpoint cases pass.
- `raw/import-paths.txt`: prepared source paths, Torch/ROCm versions, and visible device.

## Scope and limitations

The available host exposes one AMD Instinct MI350X with ROCm 7.2. The report in the original issue used Qwen3.5-122B on four CUDA GPUs (TP4), and those weights/topology were not available. Therefore this review does not independently confirm the reported 45–49% TTFT reductions, semantic behavior of that model architecture, multi-GPU execution, or 262k-context production replay. GPU execution was intentionally not used: prompt selection and the redundant host tokenizations occur before engine/GPU execution, and an unrelated GPU smoke would not strengthen this routing proof.

No remaining counterexample was found within `MessageProcessingResult.prompt_ids: Union[str, List[int]]` and the supported image/video/audio routing contract. Conversation-template paths that produce no usable IDs correctly remain on the text path; this is required for correctness and cannot receive the optimization until they produce IDs.
