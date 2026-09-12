# Independent review of PR 1076 at `3b1f061a7a8c650318803e6ee46f6332557f56f7`

Upstream issue: https://github.com/sgl-project/sglang/issues/37111

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1111

Candidate: https://github.com/amdpilot-org/sglang/pull/1076

## Verdict

Recommendation: **request changes** for use as a resolution of the original issue. The candidate is a technically valid partial fix for a related QSA/NEXTN draft-extend metadata crash, but it neither reproduces nor resolves the reported HTTP-200 silent corruption. `fully_resolves_original` is false.

On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, a real `EagleDraftExtendInput(num_tokens_per_req=4)` raises `AttributeError` because `_speculative_max_row_length` reads the nonexistent `draft_token_num`. At the exact candidate commit, using the canonical `num_tokens_per_req` fixes that crash. The candidate's focused tests, independent width cases, full QSA unit file, and three gfx950 GPU reference comparisons passed.

This is not the original failure mode. The report describes successful HTTP responses with corrupt generations on 2x NVIDIA GB10 (SM121), TP2/RoCEv2, Qwen3.8-Flash-Next-NVFP4, QSA, NEXTN, and decode CUDA graphs. The candidate only changes a Python slice-width calculation and adds tests. It does not disable or reject the unqualified graph profile, add a semantic eager/graph comparison, or show model-level correctness. Indeed, it removes a metadata-construction crash and permits execution to proceed into the graph path whose semantics remain unverified.

## Environment and source paths

- Prepared checkout started exactly at the recorded base and was restored to `amdpilot/j-59ee8384fc7b` before this report was committed.
- Python: `/tmp/amdpilot-repo-j-59ee8384fc7b/venv/bin/python`.
- Imported SGLang: `/job/repo/python/sglang/__init__.py`.
- Imported reviewed module: `/job/repo/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py`.
- GPU: one AMD Instinct MI350X (`gfx950`), ROCm 7.2, Torch 2.11.0+rocm7.2.
- No native source changed, so no native rebuild was applicable.

## Remaining counterexamples and limitations

The original graph profile remains a counterexample until it passes the issue's two semantic gates against eager on the required model and architecture: the 1,024-token non-looping generation and approximately 25K-token ordered-marker retrieval. No GB10/SM121 CUDA device, second GPU/node, RoCEv2 TP2 setup, or Qwen3.8-Flash-Next-NVFP4 weights were available. The SM121-only test was skipped. A tiny Llama HTTP fixture would exercise transport and a different architecture, so it was not substituted as proof.

Raw concise command evidence is retained in `evidence/review-output.txt`.
