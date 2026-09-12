# Correction generation 2: explicit empty OpenAI model

This correction preserves candidate PR https://github.com/amdpilot-org/sglang/pull/2463 at commit `df765c682bc88703e09f19714d75dfd4bbf42aa2` and addresses the concrete counterexample reported by independent review https://github.com/amdpilot-org/sglang/pull/2606.

The candidate distinguished omitted model fields with Pydantic's `model_fields_set`, but then treated every falsy parsed value as omitted. `CompletionRequest(model="", prompt="hello")` therefore reached request conversion instead of returning `404 model_not_found`. The correction changes the second guard from `if not model` to `if model is None`, preserving omission/optional-`None` behavior while validating an explicit empty string.

The regression asserts that the empty model is in `model_fields_set`, receives the exact OpenAI-compatible error body, and is rejected by `handle_request` before `_convert_to_internal_request` can run.

No GPU or model weights are needed for this pre-engine validation path. The candidate's Rust gateway already rejects an explicit empty model consistently, but the prepared environment has no `cargo` or `rustfmt`, so its native tests remain unexecuted here.

