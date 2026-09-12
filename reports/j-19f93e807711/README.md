# Independent review of PR 1581

Candidate: https://github.com/amdpilot-org/sglang/pull/1581 at
`0a016c8b5c0d352f95c37f7830247ab404f634ca`

Upstream issue: https://github.com/sgl-project/sglang/issues/34719

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1619

## Verdict

Recommendation: **accept**. The candidate fully resolves the original issue's
consumer-side mixed tensor/list crash in both prefill-result and decode paths.
It guards all six conversions: token-ID values, top-logprob values, and
top-logprob indices in each path. No remaining counterexample was found within
the original contract.

## Independent evidence

On the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the
candidate regression failed 3 tests with the reported `AttributeError`; its
homogeneous tensor control passed. This used the candidate test copied outside
the checkout, so the source under test remained the recorded base.

At the exact candidate commit, the candidate regression passed 4 tests. An
independent adversarial script isolated each affected field so an earlier
failure/conversion could not mask a later field. All six combinations passed,
including preservation by identity of the plain-list placeholders. A second
check used actual device tensors on the assigned AMD Instinct MI350X/gfx950 and
passed both consumer paths. Focused plus neighboring scheduler tests passed 11
tests and 2 subtests.

The import-path record confirms that Python loaded
`/job/repo/python/sglang/srt/managers/scheduler_components/batch_result_processor.py`
from the temporarily detached candidate checkout. This is a Python-only change;
there is no native source change and `repository-environment.json` specifies no
native rebuild target, so no native rebuild was needed.

## Limitations

The review did not reproduce the full HTTP workload, Qwen3.5-9B-derived model,
NVIDIA B200/CUDA environment, or distributed serving. Those are not required to
exercise this model-independent normalization defect, but remain architecture
and environment limitations. The real GPU check was on one assigned AMD
Instinct MI350X/gfx950 with ROCm 7.2 and Torch 2.11.0. The prefill and decode
consumers were invoked directly through the actual imported implementation.

Raw command output is retained in `raw/`.
