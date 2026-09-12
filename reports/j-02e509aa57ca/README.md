# Independent review of PR 3001

Reviewed `https://github.com/amdpilot-org/sglang/pull/3001` at exact commit
`af65ffbefc2230468aeb533eda9d1080af9b871a` against the original feature request:

- Upstream issue: https://github.com/sgl-project/sglang/issues/36338
- Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2944
- Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3035
- Parent candidate: https://github.com/amdpilot-org/sglang/pull/2856 at
  `6fbcae3f6367d95c85a3f71d3105224145dee8f4`
- Parent independent review: https://github.com/amdpilot-org/sglang/pull/2909

## Finding

Recommendation: **request changes**. The candidate is a useful partial
kernel-level implementation, but it does not fully resolve the original issue.

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` has no selective
dequantization API. At the exact candidate, all four focused GPU numerical and
remapping tests pass. A beneficial 262,144-row case compacted to 4,029 rows,
reduced BF16 staging from 301,989,888 to 4,641,408 bytes, and measured 2.577x
faster on the assigned MI355X.

Independent adversarial execution reproduced the unresolved heuristic cases.
For 131,072 prefix rows and top-k 2,048, random 8/16/24/32-query inputs selected
compact staging but measured 1.057x/1.167x/1.282x/1.340x the separately warmed
full path. With 33 completely overlapping queries, the default gate used the
150,994,944-byte full buffer. Forcing the union used 2,048 compact rows and a
4,930,048-byte peak, although it measured 1.084x full latency in this run rather
than the candidate report's “essentially equal” characterization.

The target BF16 `flashmla_sparse` attention path remains unverified. The
available AMD gfx950 device is not NVIDIA Hopper/Blackwell, and the closest ROCm
integration test stops before attention because the legacy HIP DSA path requires
page size 1 while the fixture uses 64. Thus the review cannot establish BF16
attention output correctness or end-to-end sparse-prefill performance on the
architecture required by the feature.

No native C++, CUDA, HIP, or FlyDSL source changed between the recorded base and
candidate, so no native rebuild was applicable. Imports were verified against
the checked-out `/job/repo/python` source using the prepared interpreter.

## Evidence

- `evidence/base_selective_import.log`: failing-before import at the recorded base.
- `evidence/pytest_selective.log`: candidate focused tests, 4 passed.
- `evidence/adversarial_review.py` and `.log`: independent warmed latency/memory cases.
- `evidence/beneficial.log`: preserved positive compact-staging case.
- `evidence/dsa_integration.log`: architecture/page-size integration blocker.
