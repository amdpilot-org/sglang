# Independent review of PR 1128

Upstream issue: https://github.com/sgl-project/sglang/issues/37755

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1161

Candidate: https://github.com/amdpilot-org/sglang/pull/1128 at exact commit
`9f5b71e3aae3ad3a53b8c36683fd4d760c8cb22e`.

## Finding

Request changes. The candidate does not modify production or native code. Its
six-test regression passes unchanged against recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365` and also passes at the exact
candidate commit. It is therefore test-only hardening of behavior already on
the base, not a failing-before/passing-after correction for the reported
MiMo-V2.5-Pro-W8A8 accuracy failure.

An independent gfx950 tensor-order check passed, but it only checks a synthetic
Python tensor transformation. The prepared environment has one AMD Instinct
MI350X and ROCm 7.2. It has no Ascend device, CANN installation, reported
ModelSlim checkpoint, or two-node TP32/DP4 topology. Consequently the real
checkpoint tensor metadata, Ascend load/serve path, generated tokens, and
semantic accuracy remain unverified.

No native source changed, so a native rebuild was neither required nor
performed. Imports were confirmed from `/job/repo/python/sglang`, using the
prepared interpreter rather than an installed copy.

Raw outputs are retained under `raw/`.
