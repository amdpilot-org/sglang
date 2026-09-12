# Independent review of candidate PR 1743

Reviewed exact commit `acfc84690fe81bebb30c94e0f4a086def3190787` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the contract in https://github.com/sgl-project/sglang/issues/33902.

Recommendation: **accept**. The candidate fully resolves the original issue in the directly affected parser paths.

The recorded base and image-prepared checkout were identical. On that base, character-wise parsing with forwarding disabled emitted `[(0, "unknown", ""), (0, null, "{}")]` for both `Glm4MoeDetector` and `Glm47MoeDetector`, while one-shot parsing already filtered undeclared names. This independently reproduces the reported mismatch.

At the exact candidate commit, imports resolved to `/job/repo/python/sglang/...`, proving the changed checkout sources were exercised rather than an installed copy. The candidate changes only Python parser/test/report files, so no native rebuild applies. Its focused regression passed 8 tests. Existing related GLM and unknown-name coverage passed as part of a 36-test run.

The independent matrix used one-chunk, character-wise, and closing-boundary splits. It exercised the reported empty-argument calls, unknown calls with arguments, unknown followed by known, two unknown calls followed by known, and unknown followed by normal text under both forwarding settings. With forwarding disabled, no unknown call delta escaped and the following declared call remained usable with index zero. With forwarding enabled, the existing unknown-call output remained intact. No remaining counterexample tied to the original contract was found.

Architecture/environment limitation: this is deterministic CPU-side parsing. No gfx950 GPU, model weights, HTTP serving, native compiler, or distributed setup was needed or used, and this review makes no claim about those layers.

Raw command outputs were preserved outside revision switches in `/job/review-evidence-j-f671680afd5a/`; concise copies are included beside this report.
