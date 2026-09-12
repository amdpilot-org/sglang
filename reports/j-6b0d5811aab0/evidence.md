# DSV4 graph-budget correction evidence

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/1608 at `c0d54de7cd2aafe30f38d9b98c4c8e9c034ec70c`

Independent review: https://github.com/amdpilot-org/sglang/pull/1687

## Failing before

Command (the script loads the exact candidate source file with `runpy`, avoiding the prepared editable install):

`/tmp/amdpilot-repo-j-6b0d5811aab0/venv/bin/python /job/repo/reports/j-6b0d5811aab0/graph_budget_regression.py /tmp/j-6b0d5811-before/python/sglang/srt/layers/attention/dsa/utils.py`

Exit code: 1

Output: `graph_budget=1717986918 issue_allocation=1526726656 rows_per_chunk=None`, followed by the expected assertion failure. This independently reproduces the review counterexample: the 1.421875 GiB issue allocation is not chunked by the candidate's 1.6 GiB graph budget.

## Passing after

Command: `/tmp/amdpilot-repo-j-6b0d5811aab0/venv/bin/python reports/j-6b0d5811aab0/graph_budget_regression.py`

Exit code: 0

Output: `graph_budget=536870912 issue_allocation=1526726656 rows_per_chunk=1440`. One chunk is about 500 MiB, below the issue-reported 1.17 GiB free memory.

Focused suite: `/tmp/amdpilot-repo-j-6b0d5811aab0/venv/bin/python -m pytest -q test/registered/unit/layers/test_dsv4_indexer_chunk.py test/registered/unit/layers/test_dsv4_nonpaged_indexer.py`

Exit code: 0; `21 passed, 29 subtests passed`.

GPU equivalence check: actual AMD Instinct MI350X/gfx950 execution compared unchunked top-k selection with row chunks 1, 3, 7, 16, and 40 for 37 rows, width 2048, top-k 64. Exit code 0; all selected page sets matched.

## Limitation

The full DeepSeek-V4-Flash-0731 800K-token TP4 serving reproduction was not run. This environment provides one AMD Instinct MI350X under ROCm 7.2, not four H100s under CUDA 12.9, and the required model weights were unavailable. The correction is supported by exact allocation/budget arithmetic, focused regressions, and GPU row-chunk equivalence; it does not claim full-model or CUDA DeepGEMM validation.
