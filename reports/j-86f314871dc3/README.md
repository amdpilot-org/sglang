# Investigation report: j-86f314871dc3

Upstream issue: https://github.com/sgl-project/sglang/issues/31093

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2305

## Finding

The prepared source already contains a narrowly matching post-report fix, so
this investigation does not add another runtime change.

The report identified `78dc58151843ff3381b7913e6833ba8f8a668204`, merged as
https://github.com/sgl-project/sglang/pull/30839 after the reporter's failing
`b94ac87e0c411e3c1b120b3d13bba9092b198fe0` revision. Its title is
"Stabilize GLM-5.2 MTP IndexShare across PD and CUDA graph replay." The PR's
motivation explicitly names GLM-5.2, preservation of the draft-extend DSA seed,
and CUDA-graph replay. The relevant correction:

- restores `dsa_topk_indices` after each capture warmup/capture invocation;
- clears the transient carried seed after replay; and
- rejects graph replay when GLM-5.2 IndexShare requires a seed but none is
  available, allowing eager recomputation instead of consuming stale or
  misaligned indices.

Ancestry checks show the fix is absent from `b94ac87e` (the reported failing
main revision) and present in the prepared base
`358c163250ad3b1f62939b01ce1314a0a31a0365`.

## Regression evidence

Using the required interpreter and the assigned single GPU:

```text
HIP_VISIBLE_DEVICES=0 /tmp/amdpilot-repo-j-86f314871dc3/venv/bin/python \
  -m pytest -q test/registered/unit/spec/test_eagle_worker_v2_topk1_fastpath.py

9 passed, 7 subtests passed
```

The issue-specific `test_missing_seed_cuda_graph_fallback` is the regression
introduced with the correction. It covers three independent boundaries:

1. IndexShare enabled and seed missing: graph replay must be rejected.
2. IndexShare enabled and seed present: graph replay remains enabled.
3. IndexShare disabled and seed missing: graph replay remains enabled.

The same file also checks topk=1 chains with one, two, three, and four draft
steps and rejects an inconsistent step/draft-token configuration.

An independent lifecycle suite also passed:

```text
HIP_VISIBLE_DEVICES=0 /tmp/amdpilot-repo-j-86f314871dc3/venv/bin/python \
  -m pytest -q test/registered/unit/layers/attention/test_index_topk_share.py

11 passed
```

It covers enabled and disabled carry, draft-extend versus decode behavior,
cleanup on success and exception, and present/missing draft-extend seeds.

Raw logs and GPU inventory are retained outside the worktree at
`/tmp/amdpilot-repo-j-86f314871dc3/evidence/`.

## Hardware scope and limitations

The tests ran with GPU tensors on one AMD Instinct MI355X (gfx950), using Torch
2.11.0+rocm7.2 and HIP 7.2.26015. They validate the deterministic seed-lifecycle
and graph-admission logic only.

The original 8xB200 TP8 workload could not be reproduced: this job has no
NVIDIA B200, no CUDA runtime, and no GLM-5.2 NVFP4 checkpoint. Consequently this
report does not claim a full-model, ModelOpt FP4, CUDA-graph, TP8, or semantic
accuracy reproduction. A separate B200-only cause cannot be excluded, and the
upstream issue contains conflicting B200 reproduction results. The honest
outcome is `candidate_verified`, not `fixed` or `not_reproduced`.

No native source changed and no native rebuild was applicable.
