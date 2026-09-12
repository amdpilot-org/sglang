# Independent review of amdpilot-org/sglang PR 3423

Candidate: `f2e033dc00060fbab489fe5225dfb8fcdf061385`

Recorded failing base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Recommendation: **request changes**. The patch is a partial fix.

## Findings

On the recorded base, the candidate's real-PNG regression reproduced the original mismatch through `_apply_jinja_template` and `BaseMultimodalProcessor.load_mm_data`: one supplied image plus a literal complete Qwen vision marker rendered two markers for one attachment, for both user text and tool-result text.

At the exact candidate commit, the focused regression passed and the loader decoded exactly one 1x1 PNG. The broader serving-chat and multimodal suites also passed (160 tests, 92 subtests).

The patch nevertheless changes client text rather than separating template-authored attachment markers from client-authored text. `neutralize_qwen_vl_message_markers` runs for every request handled as a multimodal Qwen-family model, even when that request contains no image. An independent assertion requiring exact text preservation failed: the chat template received `<| vision_start |><| image_pad |><| vision_end |>` instead of the submitted `<|vision_start|><|image_pad|><|vision_end|>`.

That behavior contradicts the full original expectation that ordinary text and tool output can contain these spellings without becoming attachments. The candidate prevents the 500-producing cardinality mismatch, but does not preserve the ordinary text, and its own text-only test codifies the mutation rather than detecting it.

## Environment and architecture limits

Python imports resolved to the checked-out sources under `/job/repo/python/sglang`, including `serving_chat.py` and `base_processor.py`. No native files changed, so no native rebuild was applicable.

The environment exposed one AMD Instinct MI350X through torch 2.11.0+rocm7.2 / HIP 7.2, and a deterministic matrix multiplication matched an independent CPU reference exactly. Qwen3.8-27B weights were unavailable. Therefore this review validates the real request models, chat-render boundary, multimodal loader dispatch, and image decoding, but not a full Qwen3.8-27B server launch or semantic inference. The tiny Llama fixture cannot qualify the Qwen-VL architecture and was not used as substitute proof.

Raw outputs and the candidate patch were retained outside the checkout at `/job/review-evidence-j-b23b7050f3e2/` while revisions were switched.
