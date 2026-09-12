# Independent review of PR 1927

Reviewed candidate: `efe036384e2b805fdacd493f51d539a85b5770eb`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Recommendation: **accept**. The candidate fully resolves the original GLM-4.7 non-streaming `tool_index` issue.

## Findings

The image-prepared checkout was already on the required base commit with no revision difference. On that base, an independent direct-parser harness reproduced both examples from the report: three repeated `get_weather` calls returned `[0, 0, 0]`, and `get_weather -> search -> get_weather` returned `[0, 1, 0]`.

I then detached at the exact candidate commit. Its source change overrides the tool-list position returned by `parse_base_json` with `len(calls)` for each successfully emitted non-streaming call. This gives dense local message positions and also behaves correctly when undefined calls are filtered. No native source is changed.

The candidate's two new tests passed, the complete `TestGlm47MoeDetector` class passed, and an independent harness passed cases for five repeats, reversed request-tool order, unknown tools surrounding valid calls, and no-argument calls mixed with normal text. No remaining counterexample was found within the original issue's GLM-4.7 non-streaming contract.

The other detectors listed in the issue's scope note are not changed or qualified. This does not make the GLM-4.7 fix partial because the report explicitly identifies GLM-4.7 as its reported case, but it remains a limitation if broader shared-helper cleanup is desired.

## Environment and source evidence

Tests used `/tmp/amdpilot-repo-j-017b1350fe05/venv/bin/python` and loaded `/job/repo/python/sglang/srt/function_call/glm47_moe_detector.py`. The environment has torch 2.11.0+rocm7.2, HIP 7.2.26015, and one AMD Instinct MI350X (`gfx950:sramecc+:xnack-`). GPU execution was not relevant to this CPU-only parser defect and was not performed. No GLM model weights or serving reproduction were used, so no model-generation, semantic, or HTTP claim is made.

Raw command outputs are retained in `reports/j-017b1350fe05/raw/`.
