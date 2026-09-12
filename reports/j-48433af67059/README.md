# Independent review of PR 2463

Reviewed candidate commit `df765c682bc88703e09f19714d75dfd4bbf42aa2` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` for the OpenAI-compatible unknown-model contract.

## Result

Recommendation: **request changes**. The candidate is a substantial partial fix, but it does not fully resolve the original issue.

The prepared base lets an explicitly unknown model reach request conversion. On the candidate, its focused Python regression passes and a normal unknown model returns the expected HTTP status and OpenAI error fields, including `"param": null`. The gateway source also changes explicit model resolution from unconditional acceptance to membership validation.

However, the Python validation contains `if not model: return None`. `CompletionRequest(model="", ...)` is accepted by the protocol model, records `model` as explicitly supplied, and advances beyond the new validation gate into conversion. An empty string is not the served model and remains a concrete unknown-model counterexample to the original contract. The gateway implementation rejects the analogous explicit empty ID, so the candidate also leaves Python/gateway behavior inconsistent at this boundary.

## Evidence

- `evidence/base-original-behavior.log`: direct execution at the recorded base shows `gpt-nonexistent-999` reaching conversion.
- `evidence/base-candidate-regression.log`: the candidate's regression run against the base fails all four tests because validation is absent.
- `evidence/candidate-regression.log`: the unchanged candidate regression passes (4 tests and 2 subtests).
- `evidence/candidate-adversarial-empty-model.log`: source import paths, ordinary served/unknown behavior, exact 404 body, and the explicit-empty-model bypass.
- `evidence/candidate-native-tooling.log`: no `cargo` or `rustfmt` is installed; `git diff --check` reports no whitespace errors for the Rust change, but this is not a native build.

## Scope and limitations

No GPU execution was needed or claimed: the measured defect and fix occur before engine execution. No model weights or full serving process were used. The prepared interpreter imported SGLang and `serving_base.py` from `/job/repo/python`, confirmed in the retained log. The Rust gateway change could not be compiled or run because the prepared environment has no Rust toolchain, so native gateway correctness remains unverified beyond source review. No candidate files were modified or copied into this review branch.
