# Independent review of PR 1835

Upstream issue: https://github.com/sgl-project/sglang/issues/33563

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1879

Candidate: https://github.com/amdpilot-org/sglang/pull/1835 at
`a373085734cbb1bfc47f6029307a71014a98c9a2`.

## Verdict

Recommendation: **accept**.

The candidate is test-only hardening, not the runtime fix. The recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365` already contains the functional
correction: completion conversion forwards `extra_key` and `cache_salt`
separately to `GenerateReqInput`, whose batch normalization supports elementwise
lists, scalar broadcasting, and length validation. The candidate adds focused
regression coverage for those behaviors without modifying runtime or native
source.

The original issue is fully resolved by the source already present on the base,
and the candidate's regression tests accurately exercise that fix. This is not
an unverified claim or a new candidate-side source fix.

## Evidence

- At the issue's recorded commit `d257b58e67193780ff8a59ab54b48219b9dc28d2`,
  the actual repository schema accepts both lists and
  `OpenAIServingBase._compute_extra_key()` raises the reported `TypeError` for
  each field. The probe records imported files under `/job/repo/python/sglang`.
- At the recorded prepared base, an independent source-level probe passes for
  per-prompt `extra_key`, per-prompt `cache_salt`, both fields together, scalar
  broadcasting, and independent wrong-length rejection.
- At the exact candidate commit, its three focused tests pass and all 22 tests
  in `test_serving_completions.py` pass.
- Independent candidate-side adversarial checks pass for `n=2` namespace
  expansion, empty-string normalization, and HTTP 400 responses from the
  non-streaming completion response path for wrong-length lists of either
  field.
- `git diff` from the recorded base to the candidate contains no `python/` or
  native-source changes. Imports resolve to the editable checkout, not an
  unrelated installed SGLang copy.

Raw output is retained in `evidence/`.

## Architecture and limitations

The assigned accelerator is one AMD Instinct MI355X (`gfx950`), with PyTorch
`2.11.0+rocm7.2` and HIP `7.2.26015`. GPU execution was intentionally not used:
the defect and its validation occur in CPU-side Pydantic/request adaptation and
normalization before tokenization or engine execution. No model weights, full
HTTP server, multi-node workload, or semantic model behavior were exercised.
The independent HTTP check invokes the real completion response method with a
deterministic mocked tokenizer-manager generator; it validates error mapping,
not transport or inference.

No native source changed, so no native rebuild was applicable. The imported
SGLang, protocol, and serving modules all resolved beneath
`/job/repo/python/sglang`.

The candidate's own report and PR body refer to mirror issue 1809, whereas this
review assignment names mirror issue 1879. Both mirror issues currently have
the same title and upstream subject; this provenance discrepancy does not alter
the technical result, and this review records the assigned mirror URL.
