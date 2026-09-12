# Independent review of amdpilot-org/sglang PR 1674

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/1674 at exact
commit `5546ca0f85f6e38c689b06bff6ae70a741f7d502`.

Upstream issue: https://github.com/sgl-project/sglang/issues/34920

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1716

Parent candidate: https://github.com/amdpilot-org/sglang/pull/1523 at exact
commit `dc8a878577b35c207a96d45ad671584be75f925b`.

Parent independent review: https://github.com/amdpilot-org/sglang/pull/1589.

## Verdict

Recommendation: **accept**. The exact candidate fully resolves the original
source-level failure contract and the remaining counterexample from PR 1589.
This is a functional fix with regression hardening, not a test-only change.

The prepared base reproduced the reported failure in the real DCP planner:
`prepare_decode_context_parallel_metadata(..., extend_prefix_lens=None, ...)`
raised `TypeError` at `planner.py:64` when `torch.cumsum` received `None`.

At the exact candidate commit, `_get_dcp_extend_metadata` derives the target
verify geometry before entering that planner. For the reviewed DSpark
non-compact case where the committed device prefix is `[3]`, the live host
mirror is absent, and the temporary host total is `[10]`, it returns GPU and
CPU prefixes of `[3]`, extend length `[7]`, total length `[10]`. This corrects
the stale/expanded-host fallback rejected in PR 1589.

An independent gfx950 case used three requests, including a zero-length prefix:
committed prefixes `[0, 8, 15]`, expanded host totals `[7, 15, 22]`, and verify
width `7`. The candidate returned prefixes `[0, 8, 15]` on both host and device,
extend lengths `[7, 7, 7]`, total `44`, and the actual planner allocated a
`[44, 1]` KV buffer with prefix sum `23`. Only index-generation kernels were
replaced by no-op test doubles; helper tensor arithmetic and planner allocation
ran on the assigned GPU.

## Commands and results

- Base reproduction, exit 1 as expected:
  `PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-15fb312e2c92/venv/bin/python /job/review-j-15fb312e2c92/reproduce_planner_none.py`
  measured the exact `torch.cumsum(None)` exception in the imported planner.
- Candidate regression, exit 0:
  `python -m pytest -q test/registered/unit/model_executor/runner/test_eager_dcp_target_verify.py`
  reported `5 passed`; the CUDA parameter ran rather than skipping.
- Independent candidate GPU check, exit 0:
  `PYTHONPATH=/job/repo/python .../python /job/review-j-15fb312e2c92/adversarial_candidate.py`
  measured the multi-request values described above and verified ordinary
  extend metadata remains unchanged.
- Neighboring suites, exit 0:
  `python -m pytest -q test/registered/dcp/test_dcp_layout_unit.py test/registered/spec/dspark/test_ragged_verify.py`
  reported `27 passed, 18 subtests passed`.
- Focused pre-commit, exit 0, against the candidate source and regression test:
  all applicable checks passed.

The imported source paths were
`/job/repo/python/sglang/srt/model_executor/runner/eager_runner.py` and
`/job/repo/python/sglang/srt/layers/dcp/planner.py`, confirming tests exercised
the checkout rather than an installed SGLang copy.

## Scope and limitations

The image-prepared checkout exactly matched the recorded base commit
`358c163250ad3b1f62939b01ce1314a0a31a0365`. GitHub reports PR 1674's current
UI base OID as `a207786205bff0919eb2c8c9126c67f302ccff34`; this review nevertheless
used the mandated recorded base for failing-before comparison and the mandated
exact head commit for passing-after validation.

Hardware was one AMD Instinct MI350X (`gfx950`, ROCm 7.2), not the reported
two-node deployment with eight B300 GPUs per side. Kimi K3 weights were not
available. Therefore Mooncake PD transport, TP/EP/DCP size 8, full Kimi K3
execution, multi-node behavior, and semantic output accuracy were not exercised.
Those architecture/environment gaps limit end-to-end reproduction, but do not
leave a source-level counterexample to the reported metadata contract.

No native source changed in the candidate, and the prepared environment records
no separate native artifact for this review. A native rebuild was therefore not
applicable. The loaded AITER module came from the prepared private runtime cache.
