# Null optional config sections investigation

Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Upstream issue: https://github.com/sgl-project/sglang/issues/37846

Mirror issue: https://github.com/amdpilot-org/sglang/issues/899

Related upstream candidate: https://github.com/sgl-project/sglang/pull/37928

The prepared base does not contain the candidate fix. The issue's exact
`MiniMaxVLBaseConfig()` reproducer raised `AssertionError` when
`text_config=None`, while deleting the attribute returned the parent config.
The same base dereferenced `linear_fp8_config=None` in
`CompressedTensorsConfig.from_config`, raising `AttributeError`.

The source correction matches upstream PR #37928's narrow value checks. This
change additionally commits regression tests using the actual repository
implementations. The tests demonstrate the failures before the source change
and verify null/missing equivalence without changing populated-section behavior.

Raw evidence is retained in `reports/j-06ee346cd724/raw/`.
