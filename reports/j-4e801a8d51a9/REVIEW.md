# Independent review of PR 525

Reviewed candidate commit `97c33ff61442e589812954f6d71d59bcddae5c2c` against prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Upstream issue: https://github.com/sgl-project/sglang/issues/38981

Mirror issue: https://github.com/amdpilot-org/sglang/issues/632

Recommendation: **request changes**. The patch is a real partial fix, not test-only hardening: it coalesces touching segments before the SWA split and eliminates the reported shared-boundary-page assertion. The prepared base failed through the actual `free_kv_row_segments -> free_segments -> _page_disjoint` path, while the exact candidate passed the touching, interleaved, empty-segment, page-size-64 boundary, and fresh-request retry cases.

The unresolved counterexample is retrying the same cleanup operation. On the candidate, calling `free_kv_row_segments` twice with the same two touching ranges places the same two physical pages into `free_pages` twice. The free-page count grows from baseline+2 after the first cleanup to baseline+4 after the retry. Thus the requested idempotent cleanup contract is not satisfied.

The candidate changed only Python source and tests, so no native rebuild applied. Imports were verified from `/job/repo/python/sglang/srt/mem_cache/common.py`; PyTorch came from `/opt/venv/lib/python3.12/site-packages/torch`, and the loaded AITER native module was `/tmp/amdpilot-repo-j-4e801a8d51a9/cache/aiter/module_aiter_core.so`.

GPU validation ran on one AMD Instinct MI355X (`gfx950`) under ROCm 7.2 and passed the candidate's focused hybrid SWA test. The reported eight-NVIDIA-B300 (`SM103`) Kimi-K3/DSpark deployment was unavailable, so multi-rank CUDA server behavior remains architecture-limited and was not represented as verified.

Raw logs, import-path evidence, issue/PR metadata, candidate diff, and the independent test remain in `/job/review-evidence-j-4e801a8d51a9`.
