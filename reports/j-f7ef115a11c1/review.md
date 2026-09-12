# Independent review of PR 2591

Reviewed exact commit `4bf4438097a431ef3d4ded6e4deb5b812101aa0d` against https://github.com/sgl-project/sglang/issues/30781.

Recommendation: **accept**. The candidate fully resolves the original protocol mismatch at the exercised gateway boundary.

On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, a temporary regression using the issue's exact `{type, name, description}` custom tool failed during `ResponsesRequest` deserialization with the reported four-variant error. On the candidate, all 12 Python-supported spellings deserialize, custom validates, and the typed request reserialization used by the HTTP router preserves the exact custom object.

Independent adversarial cases also preserved a Codex-style custom grammar payload, null-valued extension fields, a namespace containing heterogeneous nested tools, and modeled MCP fields combined with future object/null fields. The full spec binary passed 95 tests with one ignored.

`cargo tree -i openai-protocol` resolved `/job/repo/sgl-model-gateway/vendor/openai-protocol`, confirming candidate source was compiled rather than the original registry crate. Python source at `/job/repo/python/sglang/srt/entrypoints/openai/protocol.py` declares the same 12 spellings exercised by the candidate regression.

Raw logs were preserved outside the revision-switching checkout at `/job/review-evidence-j-f7ef115a11c1/`. No GPU or native rebuild was relevant. The reporter's gfx942 multi-node deployment and GLM-5.2 weights were unavailable, so this review makes no architecture, semantic-quality, or distributed-serving claim.
