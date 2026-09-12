# Independent review of amdpilot-org/sglang PR 998

Candidate reviewed: `4b86ae8a60f8bab36eb56c1cde599d675af3e2b8`

Recorded and prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365` (no difference)

Recommendation: **accept**. The candidate fully resolves the original issue in the affected OpenAI/Jinja chat path. It changes the request preparation layer before the chat template renders structured media, neutralizing bare GLM-V image/video placeholders only in text-bearing values and only when a GLM-V model has real multimodal input. The structured media parts remain untouched and therefore still produce exactly one real placeholder per input.

## Evidence

- On the recorded base, the supplied processor reproducer exited 1 through `BaseMultimodalProcessor.legacy_load_mm_data`: the second bare `<|image|>` exhausted the one-item data iterator, logged `Mismatch: More 'IMAGE' tokens found than corresponding data provided.`, and raised the reported empty-suffix `RuntimeError`.
- At the exact candidate commit, the complete focused serving-chat file passed: 139 tests and 73 subtests.
- An independent adversarial matrix included simultaneous image and video inputs plus literal placeholders in user text, assistant text, `reasoning_content`, nested tool-call arguments, tool-result text, and a tool description. The synthetic rendered prompt had 5 image and 4 video occurrences before preparation, but exactly 1 image and 1 video occurrence afterward—the two structured media placeholders. The original caller-owned request remained unchanged.
- The architecture allowlist exactly matches the non-null model classes registered by `Glm4vImageProcessor` in this checkout.
- Python imports resolved to `/job/repo/python/sglang/srt/entrypoints/openai/serving_chat.py` and `/job/repo/python/sglang/srt/multimodal/processors/base_processor.py` while the candidate was checked out.

## Scope and limitations

No native source changed, so no native rebuild was applicable. The assigned accelerator was an AMD Instinct MI350X with gfx950 and 270,566,162,432 bytes VRAM; GPU execution was not used because placeholder pairing happens entirely during CPU-side request preparation. GLM-5.3-Flash weights were unavailable, so a full model HTTP launch and semantic generation were not verified. This does not weaken the deterministic reproduction of the failing processor contract or the candidate-path validation, but it remains an environment limitation. The tiny Llama serving fixture was not used because it cannot qualify GLM-V architecture-specific template behavior.

No remaining counterexample was found within the original contract. Text-only GLM-V requests and non-GLM models retain their prior behavior, while real structured image/video parts remain paired.
