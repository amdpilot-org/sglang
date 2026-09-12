# Investigation: OpenAI completion cache namespace lists

Upstream issue: https://github.com/sgl-project/sglang/issues/33563

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1809

The reported defect reproduces at the issue's recorded source commit,
`d257b58e67193780ff8a59ab54b48219b9dc28d2`. Both schema-valid list fields are
rejected by `OpenAIServingBase._compute_extra_key()` before model execution:

- `extra_key` raises `TypeError: Value of extra_key must be a string, but got list`.
- `cache_salt` raises `TypeError: Value of cache_salt must be a string, but got list`.

The prepared base, `358c163250ad3b1f62939b01ce1314a0a31a0365`, already contains
the functional correction from upstream PR #30827 (commit
`385903b0acd69455cb688b5cb5e3afcc0fd91598`). That change removed the scalar-only
composition helper and forwards `extra_key` and `cache_salt` separately from
`CompletionRequest` to `GenerateReqInput`. Its existing batch normalization:

- preserves per-prompt lists elementwise;
- broadcasts scalar values across the prompt batch;
- rejects list lengths that differ from the prompt batch size with `ValueError`;
- keeps `extra_key` and `cache_salt` as distinct namespaces.

This PR adds completion-adapter regression coverage for the first three behaviors,
including independent wrong-length checks for both public fields. It does not
change runtime source because the prepared implementation is already fixed.

Raw command output is retained under `evidence/`. The historical reproduction
explicitly reports the imported source path to distinguish it from the prepared
editable checkout.

No GPU execution, model weights, HTTP server, or native rebuild was used. The
reported failure and the verified correction occur during CPU-side request
adaptation and normalization, before tokenization or engine execution. Thus this
does not claim model inference, semantic accuracy, or distributed-workload
coverage.
