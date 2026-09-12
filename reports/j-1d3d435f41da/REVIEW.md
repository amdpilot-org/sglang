# Independent review of PR 2900

Candidate: https://github.com/amdpilot-org/sglang/pull/2900

Exact commit: `a9ce297679dec8a6cb680ac3a3b68d7b94d39e3a`

Upstream issue: https://github.com/sgl-project/sglang/issues/35334

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2939

Recommendation: **request changes**. The patch is a partial correction, not a full original-issue fix.

## Findings

1. **Stale driver records still pass startup validation.** `get_runtime_environment()` records device type/name, PyTorch, accelerator runtime, and distributed backend, but no driver version. `resolve_calibrated_plan()` compares only keys supplied by the current environment. Therefore a report with a stale `driver_version` plus matching emitted fields is accepted as `exact_calibration`. This contradicts the original requirement to fail closed on stale driver/backend/device revisions and the candidate PR's statement that it compares the current driver environment. The first test in `candidate-adversarial.log` reproduces this at the exact candidate commit.

2. **Product-consistent but workload-illegal plans are still selected.** Startup calls `_intrinsic_plan_rejection()`, which checks only `CFG * TP * SP * replicas == num_gpus`. It does not call `validate_plan()` or otherwise obtain capabilities. A report can therefore select CFG degree 2 for an exact signature with one CFG branch. Analogous unsupported TP/SP/FSDP plans remain possible. The second test in `candidate-adversarial.log` reproduces this.

3. **The original feature remains substantially incomplete.** There is no calibration command or isolated end-to-end measurement protocol, no model quality/trajectory gate execution, no communication measurement implementation, no benchmark artifact for the requested matrix, and no implemented pool routing. The candidate also changes no distributed execution code, so it does not itself establish the requested world-size-one no-communicator/no-wrapper fast path.

## Verified corrections

The pre-correction commit `1a422dd3f344dd098af11e25a4bad951bd9daf62` fails the candidate regression suite with five failures. At `a9ce297679dec8a6cb680ac3a3b68d7b94d39e3a`, all 15 focused tests pass. The exact impossible product counterexample is rejected, startup supplies the current git revision, and mismatches among the environment fields that are actually collected fail closed. Compatibility tests pass with 199 tests and 67 subtests.

## Environment and architecture limits

The prepared environment exposed one AMD Instinct MI350X with Torch `2.11.0+rocm7.2` and HIP `7.2.26015`. No model weights or model-specific quality reference were available. No GPU numerical serving run or multi-rank run was performed because the demonstrated defects are pure resolver behavior and the required distributed/model prerequisites were unavailable. No native files changed, so no native rebuild was applicable.

Raw command output and exit codes are retained beside this report.
