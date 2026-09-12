# Independent review of amdpilot-org/sglang PR 527

Reviewed exact candidate commit `bbfa5b79fe00527779c40a141cb9ccc387a96ab5` against prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the original issue contract.

Recommendation: **accept**. The candidate fully resolves the reported avoidable full-history copying while preserving valid rollback semantics.

The base reproduced accumulated superlinear host work on the actual installed xgrammar matcher: an 8x increase in generated history raised median elapsed time 21.33x. The unchanged independent workload on the candidate grew 7.88x, with each doubling near 2x. Candidate regression tests passed, as did independent empty, zero, partial, full, configured-200-token, and over-history failure-atomicity checks. In particular, the candidate also fixes the base's `rollback(0)` Python debug-history desynchronization.

The loaded source paths were `/job/repo/python/sglang/srt/constrained/xgrammar_backend.py` and `/opt/venv/lib/python3.12/site-packages/xgrammar/__init__.py`. No native source changed, so no native rebuild was applicable. The environment exposed one AMD Instinct MI355X (`gfx950`) under ROCm 7.2, but no GPU work was executed and no GPU performance claim is made. A live speculative server was not run; the controlled test directly exercised the actual backend sequence that accumulates committed history around speculative rollback.

Full structured results are in `result.json`; raw output is retained outside the checkout at `/job/review-evidence-j-0d9fcdc26559`.
