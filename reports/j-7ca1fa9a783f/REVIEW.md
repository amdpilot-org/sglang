# Independent review of amdpilot-org/sglang PR 2749

Candidate reviewed exactly at `e3af256016618895e5e59a98a79c454d5ba80bfb` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **request changes**. The patch adds API types, validation, metadata, a mean reducer, and reducer-focused tests, but it does not successfully execute the required physical Cosmos3 candidate batch.

## Blocking counterexample

The common input-validation path creates one `torch.Generator` per output. With `candidate_trajectory.count=2`, Cosmos3 therefore receives a generator list. `Cosmos3LatentPreparationStage.forward` passes that list directly to `torch.randn`, which accepts only a single generator and raises `TypeError`. The same stage derives its noise batch dimension from the conditioning image (`1` for the supported one-observation request), not from `num_outputs_per_prompt`, so no candidate axis is created there either.

Run:

```bash
PYTHONPATH=python /tmp/amdpilot-repo-j-7ca1fa9a783f/venv/bin/python reports/j-7ca1fa9a783f/adversarial_candidate.py
```

The retained output in `adversarial_candidate.txt` shows `logical_effective_batch 2`, `conditioning_batch 1`, followed by the generator-list `TypeError` at the actual Cosmos3 latent creation call. This is tied directly to the original contract's independent per-candidate RNG and single physical batch requirements, and it occurs before the claimed reducer can run.

## Other evidence

- On the base, `SamplingParams` has no `candidate_trajectory` field and importing the proposed runtime contract fails. This reproduces the missing original feature.
- The candidate's advertised focused suite passes: 153 tests and 43 subtests. Those tests exercise validation and reduction with synthetic candidate tensors, but do not execute Cosmos3 candidate latent creation.
- The candidate's standalone GPU numerical script passes on one AMD Instinct MI350X with Torch `2.11.0+rocm7.2` / HIP `7.2.26015`. It verifies seeded `torch.randn` draws and a mean against a CPU float64 reference, but it bypasses the Cosmos3 pipeline and therefore does not overcome the blocking failure.
- Source imports resolved from `/job/repo/python`. The prepared AITer extension loaded from `/tmp/amdpilot-repo-j-7ca1fa9a783f/cache/aiter/module_aiter_core.so`.
- No native source changed in the candidate, so no native rebuild was applicable.

## Scope and limitations

This review had one AMD Instinct MI350X but no Cosmos3 action model weights. End-to-end model serving, semantic quality, TP/SP, CFG, cancellation, failure recovery, dynamic co-batching identity, and the requested N={1,2,4,8,10} benchmarks remain unverified. These limitations do not mask the deterministic pre-denoising counterexample above.

The contribution is a partial API/reducer implementation and test hardening, not a full fix of the original issue.
