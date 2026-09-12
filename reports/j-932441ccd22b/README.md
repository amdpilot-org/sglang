# Correction review for Qwen multimodal preprocessing cache

Upstream issue: https://github.com/sgl-project/sglang/issues/1932

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3174

Candidate PR: https://github.com/amdpilot-org/sglang/pull/3081

Independent review PR: https://github.com/amdpilot-org/sglang/pull/3173

The exact candidate `5a5fa419998d16bb1c1cba5a91023b1272e06444`
was exercised independently. Its focused cache/hash suite passed (38 tests plus
5 subtests). A separate counterexample confirmed that changing only the prompt
changes the key for the same image content ID. Source inspection and the
candidate's plain `get` followed by later `put` also confirmed that identical
cold requests could both enter preprocessing.

This correction preserves the candidate's bounded exact-request Qwen cache and
routes misses through `MultimodalPreprocessCache.get_or_compute`. Concurrent
identical cold requests now share one cancellation-safe computation, and every
caller receives an isolated deep copy. The corrected focused suite passes 39
tests plus 5 subtests, including a regression that holds the owner computation
open while a second request joins it.

This remains a partial implementation of the original feature request. The
rendered prompt is still part of the key because the current Qwen Hugging Face
processor fuses prompt tokenization with media preprocessing. There is no
explicit OpenAI save/load operation, no caller-selected cache namespace beyond
trusted media content hashes, no cross-tokenizer-worker or restart-persistent
storage, and video/audio remain excluded. Implementing those contracts requires
an API and shared/durable storage design plus qualifying Qwen multimodal model
validation; the unavailable Qwen weights were not used to justify speculative
changes. The available tiny Llama fixture cannot validate this architecture.

No native source changed, so no native rebuild applies. GPU inventory was
recorded only; no GPU execution or numerical claim is made.
