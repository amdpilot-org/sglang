# Independent review of amdpilot-org/sglang PR 1965

Candidate reviewed: `723d55b5b5f30fba408b11761e3bd15292d4623b`

Upstream issue: https://github.com/sgl-project/sglang/issues/33783

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2002

Recommendation: **request changes**. The candidate is a meaningful partial fix, but it does not fully provide the original issue's complete KV-provenance invalidation boundary.

## Findings

1. The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` reproduces the idle-cache failure. An independent simulated `RadixCache` probe inserted a four-token completed-request prefix, called `pause_generation(mode="retract")` with no active request, and observed `before_match=4`, `after_match=4`, and zero allocator-clear calls.
2. The exact candidate fixes that counterexample. The same probe observes `after_match=0`, and the token allocator is cleared once. Its focused scheduler/retract suite passes 25 tests plus 2 subtests.
3. The candidate correctly replaces generic pressure eviction with `tree_cache.reset()`. Source inspection confirms `HiRadixCache.reset()` resets its cache controller, clears the host token pool and host-leaf/per-request tracking, then invokes the radix reset. This addresses the reviewed HiRadix write-back concern more directly than `evict()`.
4. A live disaggregated-prefill mid-chunk request remains an explicit exception. In an independent adversarial probe, a pre-existing four-token radix prefix remained matchable 4/4 after retract, the live chunk was retained, and the allocator was not cleared. The source comment acknowledges this leaves stale-weight prefix KV. This conflicts with the original contract that no prefix computed before a weight update can be matched after resume.

The result is therefore a partial original-issue fix, not test-only hardening and not a full resolution.

## Commands and measured results

Base existing suite:

```text
PYTHONPATH=python /tmp/amdpilot-repo-j-61bbfee20c03/venv/bin/python -m pytest -q test/registered/unit/managers/test_scheduler_pause_generation.py
21 passed, 2 subtests passed; exit 0
```

Base independent idle-prefix probe:

```text
{'before_match': 4, 'after_match': 4, 'allocator_clear_calls': 0}
exit 0 (probe expected and confirmed the base failure)
```

Candidate focused suite:

```text
PYTHONPATH=python /tmp/amdpilot-repo-j-61bbfee20c03/venv/bin/python -m pytest -q test/registered/unit/managers/test_scheduler_pause_generation.py test/registered/unit/managers/test_retract_all_cache_release.py
25 passed, 2 subtests passed; exit 0
```

Candidate independent boundary probes:

```text
{'case': 'idle_completed_prefix', 'after_match': 0, 'allocator_clear_calls': 1}
{'case': 'disagg_prefill_live_chunk', 'after_match': 4, 'chunk_retained': True, 'allocator_clear_calls': 0}
exit 0 (asserted both observed boundaries)
```

Import-path verification at the candidate revision:

```text
scheduler_source=/job/repo/python/sglang/srt/managers/scheduler.py
radix_source=/job/repo/python/sglang/srt/mem_cache/radix_cache.py
```

`git diff --check 723d55b^ 723d55b` also reports trailing whitespace in candidate-owned report evidence files. This is non-functional and not the basis of the recommendation.

## Environment and limitations

The prepared environment exposes one AMD Instinct MI350X (`gfx950:sramecc+:xnack-`), Torch `2.11.0+rocm7.2`, HIP `7.2.26015`, and a checkout-local Python import path. These scheduler/cache changes contain no native source changes, so no native rebuild was required or performed.

The deterministic unit-level cache fixtures exercise real `RadixCache` matching/reset logic but do not execute model inference. No model weights or multi-node disaggregated deployment were available, so end-to-end output provenance, a real HiRadix host/device stack, and distributed sender teardown were not validated. Those limitations do not erase the source-level and deterministic counterexample: the candidate intentionally skips all reset/allocator clearing in the live disaggregated-prefill path.
