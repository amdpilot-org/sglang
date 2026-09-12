# Independent review of PR 1166

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/1166 at exact commit `379c0b38f931d162821029425582820834c7f784`.

Upstream issue: https://github.com/sgl-project/sglang/issues/38821

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1201

## Conclusion

The candidate contains a credible partial processor-path fix and its adjacent-placeholder correction passes both its regression and independent cardinality boundary cases. It does **not** fully resolve or reproduce the original issue's semantic contract. No request using the exact Statue of Liberty JPEG was sent through `/v1/chat/completions` with `GLM-5.3-Flash-sglang0907-128k-0907t3`, and the observed visual-but-wrong bird response remains unexplained. The candidate's own `result.json` labels the outcome `fixed`, which overstates the available evidence. Recommendation: request changes to characterize the result as partial/unverified against the original issue rather than fixed.

## Revision and import-path evidence

The prepared checkout began clean at recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, identical to the image-prepared checkout. The candidate was fetched and checked out detached at the exact requested commit, then the checkout was returned to `amdpilot/j-da7e127a9d3b` before this report was added.

Candidate tests used `/tmp/amdpilot-repo-j-da7e127a9d3b/venv/bin/python` with `PYTHONPATH=/job/repo/python`. Imports resolved to `/job/repo/python/sglang`, including candidate files `python/sglang/srt/configs/glm5_next_processing.py` and `python/sglang/srt/multimodal/processors/glm4v.py`. No C/C++ or other native source changes exist in the candidate, so a native rebuild was not applicable.

## What was reproduced

On the recorded base, the candidate regression file fails during collection because `sglang.srt.configs.glm5_next_processing` does not exist. This confirms the prepared base lacks the candidate's GLM-5.3 compatibility processor, but it is not a reproduction of the original semantic failure.

At the exact candidate commit, 14 focused GLM multimodal tests passed. Independent direct probes covered zero through four requested images, adjacent expanded runs, separated runs, absent placeholders, and image counts larger than available placeholders. The helper retained no more placeholder tokens than existed, preserved separated runs, and retained adjacent placeholders up to request cardinality.

The original failure was not reproducible here. A filesystem search did not locate the exact JPEG SHA-256 `ff13fd6f991b37253d3745dc6b9ef8e7a92f17cee7c4c8bc84735000a668fcd7`. The private model weights, reported 0907 image, eight NVIDIA H20 GPUs, and CUDA environment were unavailable. The assigned device is one AMD Instinct MI355X (`gfx950`). A tiny Llama serving fixture would only test transport/engine execution and cannot establish GLM-5.3 visual semantics, so it was not substituted as proof.

## Classification

- Adjacent pretokenized placeholder defect: fixed by the candidate and independently verified at unit level.
- Synthetic JPEG data-URL decoding/preprocessing: covered by a passing unit test, but this is transport/preprocessing hardening rather than semantic verification.
- Exact original issue: unverified and not fully resolved.
- Deployment's already-visual but unrelated bird semantics: remaining counterexample with no demonstrated causal explanation.

The candidate diff also fails `git diff --check` because retained raw pytest logs contain trailing whitespace. This is report-artifact hygiene, not the substantive reason for the recommendation.
