# Investigation: DSA-disabled seed metadata assertion

Upstream issue: https://github.com/sgl-project/sglang/issues/36598

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1140

## Result

The reported assertion is not reproducible at the recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`. The source already contains the
necessary non-DSA guard in `get_dsa_seed_metadata_dim`: when `index_topk=None`
makes `is_deepseek_dsa(hf_config)` false, the helper returns zero before calling
the DSA-only MTP-width helper.

The guard entered upstream `main` incidentally in commit
`52fecfdf0908dca24f4c6799ff5967125cc4110e` (PR #37500, merged 2026-09-08) to
prevent QSA configs from producing DSA metadata. It also resolves the state in
the issue. The issue-specific PR #36645 remains open and proposes the same
guard against an older implementation of the width calculation.

## Evidence and scope

The regression covers the exact reported state (`index_topk=None`, sharing
still true), disabled DSA with sharing false, enabled DSA with sharing false,
and enabled DSA with both the default and a nontrivial pooled seed width. The
focused test passes all five cases; raw pytest output is retained beside this
report.

This path is configuration-derived CPU logic, so no GPU kernel is executed by
the regression. The assigned accelerator was observed as one AMD Instinct
MI355X (`gfx950`), which cannot reproduce the reporter's NVIDIA GB10 (`sm_121`)
kernel limitations. The GLM-5.3 checkpoint weights were not available. No full
model, HTTP serving, semantic-accuracy, SM121, TP2/EP2, or multi-node result is
claimed. The deterministic tiny Llama fixture would validate a different model
architecture and therefore was not substituted for the reported GLM path.

No native component changed or was rebuilt.
