# Independent review of PR 1762

Reviewed candidate: `abf75d1860543491105348b87de9615268245e20`

Recommendation: **request changes**

The recorded base commit could not reproduce the exact reported text-array
failure. It already converts the issue's two `input_text` parts into the tool
message string `First resultSecond result`. The candidate changes no production
code; it adds three tests and prior-job evidence. Its exact-payload and empty-list
tests are useful regression coverage.

The candidate does not fully cover the declared API contract. The installed
OpenAI schema defines `function_call_output.output` as either a string or a list
of input text, input image, and input file parts. Independent schema-valid cases
show that the base/candidate implementation silently turns image-only and
file-only lists into an empty string and removes an image from a mixed list.
The candidate's `test_function_call_output_ignores_non_text_content_parts`
explicitly makes the image-loss behavior an expected result. This is test-only
hardening of a partial implementation, not a full original-issue fix.

## Evidence

- `evidence/base_exact_payload.txt`: the issue payload validates and normalizes
  successfully on recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.
- `evidence/base_existing_normalization_tests.txt`: the base's four existing
  normalization tests pass.
- `evidence/candidate_regression.txt`: all seven candidate normalization tests
  pass at the exact candidate commit.
- `evidence/candidate_adversarial_valid_parts.txt`: schema-valid image and file
  output arrays lose their content; the file also records the source import path.
- `evidence/environment.txt`: the assigned device is one AMD Instinct MI350X
  (`gfx950` capability 9.5), with PyTorch 2.11.0+rocm7.2 and HIP 7.2.26015.

## Classification and limitations

This review classifies the candidate as **test-only hardening for a partial
fix**. The exact text-array example is already fixed on the recorded base, but
the wider content-part-list contract remains incomplete.

No native source changed between the recorded base and candidate, so no native
rebuild was applicable. The checked-out Python implementation was confirmed at
`/job/repo/python/sglang/srt/entrypoints/openai/serving_responses.py`.

The review did not launch a model server or execute model kernels because the
remaining defect is deterministic request normalization and does not require a
GPU. The assigned single gfx950 GPU was visible, but `gpu_execution` is false
for this review. DeepSeek-V4-Flash-0731 weights, its parsers, DSPARK, CUDA, and a
distributed workload were not available; no claim is made about those paths.

Upstream issue: https://github.com/sgl-project/sglang/issues/33867

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1806
