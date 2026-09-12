# Independent review of amdpilot-org/sglang PR 791

Reviewed exact candidate commit: `e4c781e77f4537963484861602ee68910efab9d4`

Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Upstream issue: https://github.com/sgl-project/sglang/issues/38428

Mirror issue: https://github.com/amdpilot-org/sglang/issues/833

## Recommendation

Request changes. The candidate fixes the operationally critical thread-death and
host-release path, but it does not fully meet the issue's explicit requirement
that the acknowledged operation say what went wrong.

On the base, the issue reproducer killed `HiCacheController.backup_thread_func`,
acked no operations, and left one queued. On the exact candidate, the exception
was logged with a traceback, the worker remained alive, both operations were
acked, and the queue drained. The candidate's six focused tests also passed.

Independent tests injected consecutive, distinct `TimeoutError("remote timed
out")` and `ValueError("malformed response")` failures into both the standard
and hybrid workers. Both workers survived and acknowledged all three queued
operations, including a following success. However, the failed ack objects were
indistinguishable: both contained `failed=True`, `failure_kind="exception"`, and
an unwritten-page count, but neither contained the exception type, message, or
another cause identifier. The traceback exists only in the log. Thus an ack
consumer cannot tell a timeout from a malformed response.

This is a partial fix, not test-only hardening: the source changes repair queue
progress and enable existing ack consumers to release host memory. To fully
resolve the reported contract, exception acks need stable causal information
(for example a safe exception type/message field), with a regression asserting
that distinct backend failures remain distinguishable from the ack itself.

## Environment and scope

- Python: `/tmp/amdpilot-repo-j-df20781f5d94/venv/bin/python`
- Source imports resolved to `/job/repo/python/sglang/...` on both revisions.
- Torch: `2.11.0+rocm7.2`; one AMD Instinct MI350X (`gfx950`) was visible.
- The issue-specific path is CPU-only queue/thread logic. No GPU kernel, model,
  serving, remote backend, or distributed workload was run or claimed.
- The candidate changes only Python and report/test files. No native source was
  changed, so a native rebuild was not applicable.

Raw review evidence was preserved outside the revision-switching checkout in
`/job/review-evidence-j-df20781f5d94/` during the review.
