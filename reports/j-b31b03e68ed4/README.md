# Independent review of candidate c44430d3a88b2ae1d2c8bd7cae34f4f955f3d0bf

Upstream issue: https://github.com/sgl-project/sglang/issues/33505

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1920

Recommendation: **request changes**.

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`
already contains a one-level fix in `get_config`: when an override value is a
dictionary and the current value is a `PretrainedConfig`, it calls
`current.update(value)` rather than replacing that sub-config with a plain
dictionary. The base regression passed. Candidate `c44430d` changes no runtime
source or native code; it strengthens tests and adds an earlier investigation
report. Its five focused tests pass and its issue-shaped complete
`rope_parameters` replacement preserves fields directly under `text_config`.

That does not establish the original issue's stated recursive-merge contract.
The independent adversarial case in `raw/adversarial_recursive.txt` overrides
only `text_config.rope_parameters.rope_theta`. The candidate produces
`{"rope_theta": 20000.0}` and loses the existing sibling
`"rope_type": "default"`; the assertion fails with `KeyError`. The candidate's
new plain-dictionary test explicitly expects replacement rather than recursive
merging. Thus this is useful test-only hardening for the known one-level fix,
but the full recursive contract remains unresolved.

No native files differ, so no native rebuild was applicable. Imports were
confirmed from `/job/repo/python/sglang`, not an installed SGLang wheel. The
available device is one AMD Instinct MI355X (gfx950) under ROCm 7.2. The
reported two-RTX-5090 CUDA/NVFP4 server, unavailable model weights, TP=2, and
model semantics were not reproduced; no GPU execution was needed for these
configuration-only tests.

