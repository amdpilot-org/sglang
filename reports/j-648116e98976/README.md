# Candidate correction verification

Upstream issue: https://github.com/sgl-project/sglang/issues/33656

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1962

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1829 at
`67d1d2f2dc265f1475651e35986b430aab3a960a`.

Independent review PR: https://github.com/amdpilot-org/sglang/pull/1929.

## Result

The candidate's narrow correction is valid and retained.  On a follower rank
for an MLA/DeepSeek-V4 pool, replicated objects are written by TP0.  An empty
set of follower-local sidecar transfers is therefore successful, rather than a
reason to publish `completed_tokens = 0`.

The candidate regression was run against the recorded base and failed exactly
at `0 == 128`; it passes after the correction, with successful-sidecar and
failed-sidecar boundaries also passing.  Adjacent pool assembly and Mooncake
group-semantics tests pass.

## Commands

```bash
/tmp/amdpilot-repo-j-648116e98976/venv/bin/python -m pytest -q \
  test/registered/unit/mem_cache/test_hybrid_cache_controller_backup.py

/tmp/amdpilot-repo-j-648116e98976/venv/bin/python -m pytest -q \
  test/registered/unit/mem_cache/test_hybrid_pool_assembler.py \
  test/registered/unit/mem_cache/test_mooncake_group_semantics.py
```

## Remaining review counterexamples

No additional source correction is justified from the available environment.
The closest repository DSv4 DSpark HiCache L3 test requires a DeepSeek-V4
checkpoint and launches TP=4.  This job has one AMD gfx950 and no DeepSeek-V4
weights, rather than the reported 8x H20/CUDA topology.  Consequently this work
does not establish a FULL+SWA eviction/large-prefix restore, a TAIL_K_SWA
position oracle (`512` versus `8448`), fresh-namespace multi-rank object
identity, or downstream finite logits/sampling probabilities.  The tiny Llama
fixture cannot qualify those architecture-specific claims.

The base regression demonstrates only the follower acknowledgement defect.  It
does not demonstrate the original report's successful restore and position
corruption, so this PR does not claim that the production NaN is removed.
