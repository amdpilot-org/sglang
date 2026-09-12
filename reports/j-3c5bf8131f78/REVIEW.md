# Independent review of PR 3145

Upstream issue: https://github.com/sgl-project/sglang/issues/16255

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3152

Candidate: https://github.com/amdpilot-org/sglang/pull/3145 at exact commit `8f131570f6d56fa3298720cfec2506660bfd4596`  
Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Verdict

Recommendation: **request changes**. `fully_resolves_original: false`.

The candidate is a valid, behavior-preserving partial refactor. It extracts
DeepSeek V3.2 indexer/pipeline policy into `deepseek_common/v32_mixin.py` and
extracts MHA/MLA/DSA NPU dispatch into a lazy-loaded NPU mixin. Its focused
tests and an independent adversarial truth table pass. A real AMD GPU check
also confirms the empty pipeline top-k tensor retains device, shape, and dtype.

It does not complete the original issue's proposed hardware separation:
`deepseek_common/hardware_backend` contains only the NPU mixin, with no
dedicated AMD or CPU hardware mixins, while numerous CPU, ROCm/AMD, NPU, and
MUSA branches remain in `deepseek_v2.py`. Existing attention-forward files do
separate several ROCm and CPU attention paths, so this is substantive progress,
not test-only hardening. No defect was found in the extracted policy itself.

## Reproduction and validation

On the exact base, running copies of the candidate regression tests fails at
collection because `deepseek_common.v32_mixin` and
`deepseek_common.hardware_backend` do not exist. This establishes the concrete
before/after structural change; the source scan records the corresponding
inline V3.2 and NPU logic.

On the exact candidate:

```bash
PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-3c5bf8131f78/venv/bin/python \
  -m pytest -q \
  test/registered/unit/models/test_deepseek_v32_mixin.py \
  test/registered/unit/models/test_deepseek_npu_mixin.py \
  test/registered/unit/models/test_deepseek_mla_dispatch.py
```

Result: 22 passed and 8 subtests passed.

The independent adversarial script in `evidence/adversarial_review.py` checks
all relevant input/output pipeline-boundary truth-table combinations, NPU
prepare/core argument routing for MHA, MLA, and DSA, and a real GPU empty-top-k
allocation. It passed on one AMD Instinct MI350X (`gfx950`). Source import
evidence confirms all reviewed modules came from `/job/repo/python`, not the
installed package. `compileall` also passed for all changed Python modules.

No native source changed, and the prepared environment declares no native
build target, so a native rebuild was not applicable.

## Unverified and remaining scope

- No real DeepSeek V3.2 checkpoint or model weights were available. Actual
  Indexer/IndexerKPool construction with checkpoint loading, logits, serving,
  and semantic accuracy remain unverified.
- Only one GPU was assigned. No multi-GPU or pipeline-parallel stage boundary
  exercised the extracted top-k handoff policy.
- No NPU was available; NPU validation is limited to imports, lazy-loading, and
  mocked call-signature/dispatch behavior, not NPU kernels.
- NVIDIA, CPU-only, MUSA, and other architecture-specific runtime paths were
  not executed.
- The tiny Llama fixture is not relevant evidence for DeepSeek V3.2 model or
  distributed semantics and was intentionally not used as substitute proof.

Raw logs, the candidate code diff, source scans, exact import paths, and the
independent adversarial script are retained under `evidence/`.
