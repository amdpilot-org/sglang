# Multimodal pre-body weighted admission correction

Upstream issue: https://github.com/sgl-project/sglang/issues/33035

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3076

Candidate parent PR: https://github.com/amdpilot-org/sglang/pull/2938

Independent review parent PR: https://github.com/amdpilot-org/sglang/pull/3044

The exact candidate commit `7da983b52f8e95df6c41bebe4a685775e3e65f9f`
was reproduced before modification. With an item budget of four, four concurrent
4 MiB bodies all entered ASGI `receive()` while each held only one provisional
item, retaining 16 MiB even though each request's eventual weight was four.

The correction reserves the controller's maximum possible request weight before
an admitted endpoint consumes an unparsed body. Once JSON validation exposes the
real media count, the candidate's existing atomic resize releases excess capacity
and preserves weighted ownership through preprocessing, executor futures,
cancellation, and scheduler handoff. The matching after-case admits one body,
retains 4 MiB, rejects the other three before `receive()`, and returns accounting
to zero.

This conservative pre-parse reservation serializes request-body parsing while the
option is enabled because media weight is unknowable until JSON has been consumed.
It does not claim to bound kernel or protocol-server buffers outside application
`receive()`.

The exact 200-client, 28-image Kimi-K2.6 workload was not run: the required model
weights and equivalent 1.5 TiB host were unavailable. The tiny Llama fixture is
text-only and cannot qualify Kimi-K2.6 multimodal behavior. Rust frontend and EPD
language-only mode remain explicitly unsupported by the preserved validation.
No native source changed, and no GPU or numerical claim applies to this CPU/ASGI
correction.

## Reproduction

Focused corrected tests:

```bash
PYTHONPATH=python:test/registered/unit/entrypoints/openai \
  /tmp/amdpilot-repo-j-affe5e7ddee0/venv/bin/python -m pytest -q \
  test/registered/unit/entrypoints/test_mm_body_admission.py \
  test/registered/unit/managers/test_multimodal_preprocessing_admission.py
```

This reports `12 passed`. Full commands, measurements, exit codes, and raw
outputs are recorded in `result.json` and `evidence/`.
