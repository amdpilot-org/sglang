# HiCache cross-request mix-up investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/32605

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2009

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Outcome

`candidate_verified`: current source already contains narrowly relevant fixes and their deterministic regressions pass. The original production report was not reproduced or declared fixed because its H200/GLM-5.2/PD-disaggregation/Mooncake configuration is unavailable and the report lacks an executable reproduction.

## Issue evidence

The source report describes response mix-up with HiCache on 0.5.15, using GLM-5.2 in an H200 prefill/decode-disaggregated deployment backed by Mooncake. It does not provide launch commands, request/output pairs, concurrency, reproduction rate, or an isolating HiCache/Mooncake comparison. A mirror comment reports another production-only cross-user symptom but likewise provides no deterministic reproduction.

## Related fixes already in source

- https://github.com/sgl-project/sglang/pull/24226 documents two slot-reuse races in `DecodeKVCacheOffloadManager`. The old path freed prefill KV slots while the request continued to attend to them, allowing a concurrent request to overwrite live KV. It also freed finished-request slots before asynchronous device-to-host copies necessarily completed. The PR reports peer-request prompt substrings appearing in responses under sustained load and moves release behind request completion and copy acknowledgments.
- https://github.com/sgl-project/sglang/pull/37026 keys decode-offload state and in-flight counters by concrete request identity rather than caller-provided `rid`, preventing a new request that reuses an ID from inheriting stale progress or being modified by an old callback.
- https://github.com/sgl-project/sglang/pull/36382 keys L3 storage prefetch by the requesting namespace (`extra_key` / `cache_salt`) rather than an unnamespaced root anchor. This is relevant to tenant isolation when those features are configured, although the source issue does not state that they were used.

The prepared base contains the resulting behavior in `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py`: no prefill slot is freed during `offload_kv_cache`; request objects are weak-keyed independently; and finished requests are released only after their last in-flight offload acknowledgment.

## Validation

Full deterministic regression:

```text
PYTHONPATH=python /tmp/amdpilot-repo-j-07a2b315b545/venv/bin/python -m pytest -q test/registered/unit/disaggregation/test_specv2_kvcache_offloading.py
15 passed, 2 subtests passed
```

Focused independent cases:

```text
6 passed, 9 deselected
```

The focused selection covers:

- no premature release of the live prefill range;
- a prompt shorter than one page;
- two distinct requests reusing the same `rid`;
- finish while one offload is still in flight;
- finish while multiple writes are in flight;
- release of all committed slots after the final acknowledgment.

Raw outputs are retained at:

- `/tmp/amdpilot-repo-j-07a2b315b545/evidence/specv2_kvcache_offloading.txt`
- `/tmp/amdpilot-repo-j-07a2b315b545/evidence/focused_boundaries.txt`
- `/tmp/amdpilot-repo-j-07a2b315b545/evidence/gpu_inventory.txt`

## Hardware scope and limitations

The prepared interpreter reports PyTorch `2.11.0+rocm7.2`, HIP `7.2.26015`, and one AMD Instinct MI355X (`gfx950`). This is not the reporter's NVIDIA H200 environment. GPU inventory was queried, but no issue-qualifying GPU execution was performed. GLM-5.2 weights, multiple H200 nodes, the reporter's Mooncake configuration, and exact traffic are absent. A deterministic tiny-Llama fixture would validate transport and engine execution only and therefore would not resolve those gaps; it was not substituted for the reported workload.

No source change or native rebuild was warranted. The PR records the investigation and evidence without claiming the original unspecified production bug is conclusively solved.
