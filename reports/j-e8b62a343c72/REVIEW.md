# Independent review of PR 3108

Reviewed exact candidate commit `5870e01368cead04c7d273ffbd0b3a89b4108f88` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and upstream issue https://github.com/sgl-project/sglang/issues/3642.

## Finding

Request changes. The candidate correctly fixes the reproduced accounting defect for an overlong prompt with nonzero speculative-decoding reserved slots: at context 10 with 3 reserved slots, it retains 7 prompt tokens rather than 10. Its focused and adjacent unit tests pass.

The implementation still accepts a request whose accounted token total exceeds the model context when reserved slots alone exceed the context. With context 10, one prompt token, one requested completion token, and 11 reserved tokens, the candidate truncates prompt and completion to zero but returns normally with an accounted total of 11. `compute_num_reserved_tokens()` derives this value from configurable EAGLE top-k and step settings and does not cap it to context length. An impossible budget should not be reported as successfully truncated.

The candidate's retained HTTP/GPU evidence uses `speculative_algorithm=None`, so `num_reserved_tokens` is zero and the exercised server path does not cover the changed behavior. It demonstrates the pre-existing completion truncation path, not the candidate-specific reserved-slot fix. No qualified EAGLE draft model was available for independent end-to-end validation.

## Scope and environment

- Prepared checkout exactly matched the recorded base before review.
- Candidate source import resolved to `/job/repo/python/sglang/srt/managers/tokenizer_manager.py`.
- The candidate changes Python and report/test files only; no C++, HIP, CUDA, or FlyDSL source changed, so no native rebuild applies.
- Prepared stack reports Torch `2.11.0+rocm7.2` and HIP `7.2`. Candidate artifacts identify an AMD Instinct MI355X / gfx950 run, but this review did not independently execute GPU inference because the changed request-validation logic has no numerical GPU operation and the available qualified tiny-Llama fixture cannot validate the EAGLE reserved-token path.

Raw command output is retained in `evidence/`.
