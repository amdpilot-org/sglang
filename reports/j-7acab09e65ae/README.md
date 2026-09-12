# Function-call output array correction

Upstream issue: https://github.com/sgl-project/sglang/issues/33867

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1911

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1762

Independent review PR: https://github.com/amdpilot-org/sglang/pull/1873

The candidate was reviewed at exact commit
`abf75d1860543491105348b87de9615268245e20` against prepared base
`358c163250ad3b1f62939b01ce1314a0a31a0365`. Its production implementation
is identical to the base; it only adds tests and reports.

## Reproduction

`evidence/candidate_before.txt` validates the counterexample payloads through
`ResponsesRequest` and then calls the actual normalization implementation.
Before this correction, image-only and file-only outputs became empty strings,
mixed text/image/text became `beforeafter`, and the original two-text example
became `First resultSecond result`.

Chat tool messages cannot natively contain image or file content parts. The
correction therefore keeps text parts as text, serializes non-text parts as
compact JSON, and joins array entries with newlines. This preserves every
schema-valid part and its boundary while retaining a chat-compatible string.

## Validation

- Focused normalization class: 8 passed. See
  `evidence/after_focused_tests.txt` and its exit-code file.
- Complete Responses serving unit file: 39 passed and 2 subtests passed. See
  `evidence/after_full_tests.txt` and its exit-code file.
- Direct post-fix schema/normalization probe: all five boundary cases passed
  schema validation and retained their content. See
  `evidence/after_counterexamples.txt`.
- Repository pre-commit hooks passed for both changed Python files.

## Limitations

No model execution is needed to establish this deterministic request-schema
and normalization defect, so GPU kernels were not executed. The assigned
gfx950 GPU was detected only; that detection is not claimed as execution
evidence. DeepSeek-V4-Flash-0731 weights and the reported parser/DSPARK serving
configuration were unavailable, so architecture-specific generation remains
unverified. No native code changed and no native rebuild was applicable.
