# Correction-generation result

Upstream issue: https://github.com/sgl-project/sglang/issues/31924

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2962

Candidate parent: https://github.com/amdpilot-org/sglang/pull/2819 at `f112ca15d33b84559c5e7ca3372cf38d8a771b5c`

Independent-review parent: https://github.com/amdpilot-org/sglang/pull/2922

## Outcome

The candidate is rejected as a correction for issue 31924, while its valid
fused-QKV regression coverage is preserved in this change.

The candidate's two tests pass at the exact reviewed commit. Independent CPU
and gfx950 execution also confirms both concrete review claims: a mapped
`img_in.weight` tensor with a deliberately incompatible layout reaches the
ordinary two-argument loader unchanged and raises the empty `AssertionError`,
while an exact-name BF16-style `to_q.weight` tensor takes the identity path and
loads successfully without exercising fused routing.

No runtime transpose or reshape is proposed. The review fixture establishes
that the updater does not infer layouts, but neither the candidate nor review
has the unavailable Qwen-Image/FLUX.2 checkpoint tensor that failed in the
historical AMD job, its exact shapes, or an initial-load transformation that
would justify a particular reconciliation. Upstream discussion additionally
reports that the available real FLUX.2 transformer checkpoint used exact
runtime names and skipped zero transformer tensors on NVIDIA. Guessing a
transpose from an arbitrary `(2, 3)` fixture could corrupt other mapped linear
weights.

The prepared base still uses `named_parameters()` and generic mapping for the
secondary text-encoder/VAE path, and the ROCm standalone test remains skipped.
Those facts are retained as limitations rather than labeled fixed. Re-enabling
the full test requires the unavailable architecture weights and a successful
end-to-end run on the assigned AMD GPU.

## Evidence

- `raw/candidate-tests.log`: candidate's two tests pass at exact commit.
- `raw/candidate-counterexamples.log`: independent CPU and gfx950 review
  reproduction, including device name and ISA.
- `raw/base-preserved-candidate-tests.log`: the preserved regressions pass on
  the prepared base.
- `reproduce_candidate_counterexamples.py`: retained reproduction source.

No native source changed, so no native rebuild was applicable.
