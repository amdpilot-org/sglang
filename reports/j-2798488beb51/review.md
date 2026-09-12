# Independent review of PR 3435 at `1fefcc22`

Recommendation: **request changes**. The candidate is a partial fix and does not fully resolve the original issue.

Upstream issue: https://github.com/sgl-project/sglang/issues/35582

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3432

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3436

Candidate: https://github.com/amdpilot-org/sglang/pull/3435 at exact commit `1fefcc22ba785ac966c77bef6c7a9a93a4b3fae4`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`. The image-prepared checkout matched this commit.

## Findings

The recorded base reproduced the report through `BaseMultimodalProcessor.load_mm_data`: a one-marker/one-image control loaded one 1x1 PNG, while one attachment marker plus one literal marker selected `legacy_load_mm_data` and raised `RuntimeError('An exception occurred while loading multimodal data: ')` from an exhausted iterator.

The candidate's focused regression passes. Its common contiguous-marker case now presents one authoritative marker to the loader and loads one image. Text-only content also avoids the neutralization branch.

However, the implementation renders exact messages once, discards that result, renders a deep copy with literal markers rewritten to spaced spellings, and encodes the rewritten render. Thus the first render observed by the regression is not the model input. Independent assertions against the encoded prompt show that mixed user and tool content is changed.

There is also a direct remaining original-failure counterexample. The neutralizer operates on each text part independently. If the literal marker is split across adjacent text parts, neither part matches the complete-marker regex. A Qwen-like template that concatenates parts reconstructs the complete marker after neutralization. With one image, the resulting prompt contains two markers and the actual loader again raises the same empty-detail runtime error.

This is not merely missing semantic verification: it demonstrates that the candidate does not establish attachment identity at the parser boundary and still permits ordinary text to become another attachment.

## Evidence summary

- Base loader reproduction: control succeeded; issue case failed with the reported empty-detail runtime error.
- Candidate focused tests: 4 passed, 9 subtests passed.
- Candidate full serving-chat unit file: 139 passed, 77 subtests passed.
- Independent candidate adversarial test: exact mixed user/tool text was absent from the encoded model prompt; split-across-parts marker produced two markers and reproduced the loader error.
- Source imports resolved to `/job/repo/python/sglang/...`, confirming the checked-out source was tested.
- The candidate changes only Python, tests, and reports; no native source changed, so no native rebuild was applicable.
- One assigned AMD Instinct MI350X ran a deterministic torch/ROCm numerical check. This is environment evidence only.

Raw command output and standalone reproduction scripts are retained outside the checkout at `/job/review-evidence-j-2798488beb51/` so revision switching could not overwrite them.

## Architecture limitations

Qwen3.8-27B weights were unavailable. I therefore did not run a full OpenAI-compatible Qwen-VL server or claim semantic inference coverage. The supplied tiny Llama fixture cannot qualify Qwen-VL prompt rendering or multimodal processing and was not substituted as proof. The actual repository prompt-processing and multimodal-loader boundaries were exercised with a real decoded PNG.
