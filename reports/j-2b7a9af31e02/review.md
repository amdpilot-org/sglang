# Independent review of PR 2786

Reviewed exact candidate `6c47c937760863072df17141ca2199b0d8e3a3bb` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Upstream issue: https://github.com/sgl-project/sglang/issues/33714

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2726

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2818

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2786

## Finding

Recommendation: **accept**. The change is a functional fix, not test-only hardening. The candidate regression fails on the prepared base and passes at the exact candidate. An independent GPU-backed probe additionally verified that finalization materializes both the FULL host payload and the Mamba host sidecar and leaves cache invariants valid.

The fallback is limited to the case where `mamba_last_track_seqlen` is absent at finish. It uses `cache_protected_len`, which denotes the prefix already owned by the radix tree. The insert walk therefore revisits an existing node, increments the non-chunked write-through accounting, and schedules the existing backup machinery. `MambaComponent.commit_insert_component_data` detects the existing Mamba value (`mamba_exist=True`) rather than replacing its checkpoint with the short final extend's state.

The ordinary-attention finalization path on the recorded base already performs a non-chunked final insert walk. The reproduced gap is the hybrid-Mamba `extra_buffer` case where component preparation previously reduced the effective cache length to zero. No counterexample remained in the exercised contract.

## Independent adversarial probe

The probe reused the candidate's fixture setup but added assertions absent from the candidate test:

```python
node = case._finish_after_chunk_boundary(cache, allocator, pool)
full = node.component_data[ComponentType.FULL]
mamba = node.component_data[ComponentType.MAMBA]
assert node.backuped and node.hit_count == 1
assert full.value is not None and full.host_value is not None
assert mamba.value is not None and mamba.host_value is not None
cache.sanity_check()
```

It ran on an AMD Instinct MI355X (`gfx950`) with Torch `2.11.0+rocm7.2` / HIP `7.2.26015` and passed.

## Source and native paths

- `sglang`: `/job/repo/python/sglang/__init__.py`
- changed module: `/job/repo/python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py`
- pre-existing AITER extension observed during import: `/tmp/amdpilot-repo-j-2b7a9af31e02/cache/aiter/module_aiter_core.so`

The candidate changes only Python and tests. No native rebuild was applicable, and the existing AITER binary was not treated as proof of the fix.

## Qualification limits

The exact 256K hybrid-KDA weights/architecture and a two-process MooncakeStore restart/loadback setup were unavailable. Consequently, the original production topology's semantic/model accuracy and cross-process storage reload were not measured. The deterministic tiny Llama fixture was not used because it cannot qualify the reported hybrid-KDA/Mamba architecture and would only provide an unrelated serving smoke. NUMA balancing was left unchanged despite AITER's performance warning.

Raw logs and fetched issue/PR metadata are retained outside the checkout under `/tmp/amdpilot-repo-j-2b7a9af31e02/review-evidence/` so they survived revision switching.
