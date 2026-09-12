# Correction generation 2: reviewed quantization prefixes

Upstream issue: https://github.com/sgl-project/sglang/issues/37848

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1196

Parent candidate PR: https://github.com/amdpilot-org/sglang/pull/1094

Independent review PR: https://github.com/amdpilot-org/sglang/pull/1163

Candidate commit: `ee6033d403ad5790a1580c7a83c36d206cf01ec3`

Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Reproduction

The candidate was checked out exactly and its retained broad audit printed 38
sites passing `quant_config` without an explicit `prefix`. The retained script
does not return a failure status, so its printed count was used only as a broad
diagnostic.

An independent focused reproduction was then run against an archive of the
five candidate model sources. It found none of the 13 reviewed qualified leaf
prefixes and exited 1:

```text
qualified_review_counterexamples=0/13
exit_code=1
```

The missing sites were Persimmon's four projections and output head, Voxtral's
QKV/output projections, Falcon-H1's QKV/output projections, Hunyuan's fused
experts and output head, and GPT-J's radix attention and output head.

## Correction and validation

The correction threads the existing outer prefix through parent modules and
adds the checkpoint-qualified leaf name with `add_prefix`. It also preserves
the candidate's Bailing, Phi, Whisper, Grok, Exaone, and related fixes.

```text
$ python reports/j-1243ec051691/reproduce_review_counterexamples.py
qualified_review_counterexamples=13/13
exit_code=0

$ python -m pytest -q \
    test/registered/unit/models/test_reviewed_quantization_prefixes.py \
    test/registered/unit/layers/quantization/test_compressed_tensors_lm_head.py \
    test/registered/unit/models/test_phi_quantization_prefix.py
40 passed, 42 subtests passed
```

The new suite includes root and nested constructor-boundary checks proving
that `PersimmonForCausalLM(prefix="nested")` passes `nested.lm_head` to
`ParallelLMHead`. Pre-commit passed on all newly changed files.

After the correction, the broad syntactic audit prints 25 sites. These were
not changed speculatively: some pass literal `quant_config=None`, while the
rest require separate architecture-specific checkpoint-name and resolver
validation.

## Environment limitations

No affected model weights were available. The deterministic source and
constructor-boundary regression is CPU-only; no full-model, serving, semantic,
distributed, or GPU numerical claim is made. The prepared host exposes one
AMD gfx950 GPU, but GPU execution was not relevant to this routing defect. No
native code changed, so no native rebuild was applicable.
