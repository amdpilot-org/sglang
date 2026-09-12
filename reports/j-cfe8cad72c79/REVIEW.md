# Review of DeepSeek V4 modern-config correction

Candidate: https://github.com/amdpilot-org/sglang/pull/2950 at
`8ae54e54289174deeb7f13f3f306126f430ca108`.

Upstream issue: https://github.com/sgl-project/sglang/issues/34092

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2985

## Recommendation

Request changes. The candidate fixes the reported crash for the default config
emitted by the installed Transformers 5.12.1 and correctly expands that
config's compression mapping to `[128, 128, 128, 4]`. Its focused regression
suite passes. It does not, however, deliver the claimed parser correction or
fully validate modern schema inputs, so it is a partial original-issue fix.

## Evidence

On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, a real
`transformers.DeepseekV4Config` was saved and loaded through SGLang. The loaded
config had `compress_rates` but no `compress_ratios`; indexer count was zero and
`ModelConfig` raised the issue's `AttributeError`.

At the exact candidate commit, the same end-to-end config path passed,
returned one C4 indexer layer, and populated `[128, 128, 128, 4]`. The
candidate's full `test_model_config.py` passed (11 tests and 4 subtests).

Two independent counterexamples remain:

1. SGLang still registers `deepseek_v4` as `_DeepseekV4ConfigAlias`, a subclass
   of Transformers' DeepSeek V3 parser. The resulting loaded class is
   `sglang.srt.utils.hf_transformers.common._DeepseekV4ConfigAlias`, not
   `transformers.models.deepseek_v4.DeepseekV4Config`. This directly
   contradicts the candidate's claim that the native Transformers V4 parser is
   left in control and leaves the prior forced-parser compatibility concern
   unresolved.
2. A Transformers-created modern config can contain a partial
   `compress_rates` mapping. For layer types HCA then CSA and rates containing
   only CSA=4, candidate normalization silently returns `[0, 4]`. Similarly, a
   nested `rope_parameters` object missing `compress` silently returns `{}`.
   The candidate uses `.get(..., 0)` and `...get("compress") or {}` at the
   runtime boundary. These values select materially different attention/RoPE
   behavior instead of rejecting an incomplete schema. Its tests cover a
   missing `layer_types` list but not missing mappings or nested sections.

Raw logs and numbered source excerpts were preserved outside the checkout in
`/job/review-evidence/j-cfe8cad72c79/` while revisions were switched.

## Environment and limitations

The pinned interpreter uses Transformers 5.12.1 and Torch 2.11.0+rocm7.2.
Imports resolved to `/job/repo/python/sglang/...`, confirming the checked-out
source was exercised. One ROCm GPU was visible, but no DeepSeek V4 weights were
available. Therefore weight loading, engine startup, GPU numerical parity,
semantic accuracy, and distributed execution remain unverified. The candidate
changes only Python and report/test files; no native source changed, so a
native rebuild was not applicable. The deterministic tiny Llama fixture cannot
qualify DeepSeek V4 architecture or RoPE accuracy and was not used as substitute
evidence.
