# Independent review of amdpilot-org/sglang PR 1018

- Upstream issue: https://github.com/sgl-project/sglang/issues/37326
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1056
- Reviewed candidate: `23038059316616b7a083187745858d5eeb021674`
- Recorded/prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Recommendation: **accept**, as a verified narrow unified-cache correction
- Fully resolves original issue: **false**

## Finding

The candidate fixes a real, deterministic defect in the active
`UnifiedRadixCache.cache_finished_req` path. On the base, a finished hybrid
Mamba request with no track boundary performs an empty insert, repoints
`req.last_node` to the root despite holding a lock on another node, and exposes
the zero-length insertion result. The candidate skips that insert, frees the
request-owned KV and Mamba resources, and releases the existing request lock.

The candidate's regression failed on the exact recorded base (2 failed, 2
passed) and passed at the exact candidate (4 passed). Independent GPU cases
then ran 256 consecutive short finishes and observed no KV/Mamba allocator
depletion and no inserted prefixes; another case preserved an existing matched
tree node while cleaning up the finishing request. Focused adjacent registry,
lock-reference, and Mamba state-cap tests also passed (37 tests and 2 subtests).

The change is substantively the same focused correction proposed upstream in
https://github.com/sgl-project/sglang/pull/38191, adapted to the newer base.
Only Python source changed. The interpreter imported both `sglang` and
`unified_radix_cache.py` from `/job/repo/python`; no wheel implementation was
tested. No native source changed, and no native rebuild was applicable.

## Scope and unresolved original contract

This evidence verifies the candidate's narrow residual path, not the complete
long-uptime symptom. The reported deployment used a 176B Qwen3.8/Qwen4Exp
NVFP4 model, NEXTN, two NVIDIA SM121 nodes, TP=2, and roughly 16--24 hours of
traffic. This review had one AMD Instinct MI355X (`gfx950`) with ROCm 7.2 and
did not have the model weights or a second node. It therefore cannot measure
acceptance decay, throughput recovery, architecture-specific behavior, or a
full serving restart cycle.

The upstream issue also records a relapse after about 16 hours on a deployment
that already carried both speculative tracking clamps and the legacy cache
fix. That observation supports this unified-cache residual as relevant, but it
does not prove this is the last cause. Accordingly, this is a verified partial
fix rather than proof that the original open issue is fully resolved.

## Reproduction commands

All commands used `/tmp/amdpilot-repo-j-d159406539c7/venv/bin/python` with
`PYTHONPATH=/job/repo/python:/job/repo/test/registered/unit/mem_cache`.

```text
python -m pytest -q /job/review-evidence/test_candidate.py
python -m pytest -q test/registered/unit/mem_cache/test_unified_cache_finished_empty_insert.py
python -m pytest -q /job/review-evidence/adversarial_candidate.py
python -m pytest -q test/registered/unit/mem_cache/test_registry.py test/registered/unit/mem_cache/test_unified_radix_lock_ref.py test/registered/unit/mem_cache/test_mamba_path_state_cap.py
```

Raw outputs are retained under `reports/j-d159406539c7/raw/`.
