# Independent review of DeepSeek V4 config candidate

Candidate: https://github.com/amdpilot-org/sglang/pull/3104 at exact commit
`1b37c3e6f5659c7bd291b8cca345933c83a76cdf`.

Upstream issue: https://github.com/sgl-project/sglang/issues/34092

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3043

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3126

## Recommendation

Request changes. The candidate is a substantial partial fix: it fixes the
reported valid Transformers-created config path, selects the native
`transformers.DeepseekV4Config`, rejects incomplete compression-rate mappings,
and preserves legacy fields. It does not close the serialized missing-RoPE
counterexample it claims to close.

## Base reproduction

The prepared branch was exactly the recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`. With pinned Transformers 5.12.1,
a generated `DeepseekV4Config` serialized and loaded through SGLang became the
V3-derived `_DeepseekV4ConfigAlias`, not the native V4 class. Constructing
`ModelConfig` then failed at `self.hf_config.compress_ratios` with the original
`AttributeError`; only `compress_rates` existed. This independently reproduces
the upstream failure.

## Exact-candidate results

At exact candidate commit `1b37c3e`, imports resolved to the checked-out
sources under `/job/repo/python/sglang/`. The candidate focused suite passed:
12 tests and 4 subtests. The independent original-case probe loaded the exact
native Transformers V4 class and `ModelConfig` normalized its four layers to
`[128, 128, 128, 4]`, with one indexer layer.

Independent adversarial checks also showed that a serialized partial
`compress_rates` mapping is rejected, conflicting legacy/modern compression
fields are rejected, legacy configs remain accepted, and custom valid native
rates/RoPE values survive loading and normalization.

One real config-file path remains broken. Starting with a native generated
config, removing `rope_parameters.compress` from `config.json`, and loading it
through SGLang succeeds. Transformers reconstructs a default `compress`
subsection (`rope_theta=160000.0`) during parsing before the candidate's
validator runs. Consequently `_normalize_deepseek_v4_config` cannot observe
that the serialized active subsection was absent. The candidate regression
only mutates `rope_parameters` after constructing `DeepseekV4Config`, so it
does not exercise this parser behavior. This is test-only hardening for that
counterexample, not a complete fix of it.

Raw evidence is retained outside the checkout in
`/job/review-evidence/j-52d864784ab3/`, including `base-probe.json`,
`candidate-probe.json`, `candidate-regression.log`, and
`candidate-adversarial.json`.

## Environment and architecture limits

The pinned interpreter is
`/tmp/amdpilot-repo-j-52d864784ab3/venv/bin/python`, with Transformers 5.12.1,
Torch 2.11.0+rocm7.2, and one visible AMD Instinct MI350X. No DeepSeek V4 model
weights were available, so weight loading, full engine startup, GPU numerical
accuracy, semantic behavior, and distributed serving were not verified. The
tiny Llama fixture cannot qualify DeepSeek V4 architecture behavior and was
therefore not used as proof.

The candidate changes only Python and report/test files; there are no native
C++, CUDA, HIP, or FlyDSL changes. A native rebuild was not applicable. The
installed AITER extension loaded from the private runtime cache, but no
unrelated GPU smoke is claimed as evidence for this configuration fix.

