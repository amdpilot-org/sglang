# Independent review of PR 988

Upstream issue: https://github.com/sgl-project/sglang/issues/37475

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1023

Candidate: https://github.com/amdpilot-org/sglang/pull/988 at `a28674bb68b8ec9d901023986bcfb6df2c9a813c`

Recommendation: **accept**. The candidate fully resolves the original issue's executable contract.

## Evidence

The image-prepared checkout exactly matched the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`; there was no difference to reconcile. On that base, a DFlash config with five draft layers and no `target_layer_ids` returned `[1, 16, 31, 46, 61]`. This has the same count as the issue's trained `[5, 19, 33, 47, 61]` but different IDs, reproducing the silent collision.

At the exact candidate commit, `sglang`, `dflash_utils.py`, `models/dflash.py`, and `spec_aux_hidden_state.py` all imported from `/job/repo/python`. The candidate deletes `build_target_layer_ids`, changes both production callers to the reduced API, rejects a missing list with a diagnostic explaining that training-fixed taps cannot be inferred, and returns supplied IDs unchanged.

Independent adversarial checks covered the reported 64-target/5-draft collision, a minimal missing-key config, explicit trained IDs, the supported explicit top-level compatibility field, and a nested empty list overriding a valid top-level value. Missing and empty lists rejected; explicit lists remained verbatim. There are no remaining counterexamples within the original issue contract.

## Test results

- Base count-collision reproduction: passed its assertion that the guessed list had the trained count but different IDs.
- Candidate targeted regression: 6 passed, 10 deselected.
- Full `test_dflash_logits.py`: 16 passed.
- Full `test_spec_aux_hidden_state.py`: 5 passed.
- Independent adversarial config script: passed.

## Architecture and limitations

The environment exposed one AMD Instinct MI355X with `gfx950:sramecc+:xnack-`, Torch `2.11.0+rocm7.2`, and HIP `7.2.26015`. GPU execution was not used: this defect and fix are configuration-only and deterministic on CPU. No native rebuild was applicable because the candidate contains no native source changes. Published DFlash weights were unavailable, so this review does not claim a full serving run, model-quality measurement, or acceptance-rate reproduction. It also does not independently repeat the external 20-checkpoint survey; acceptance rests on reproducing and eliminating the concrete unsafe fallback in the prepared source.
