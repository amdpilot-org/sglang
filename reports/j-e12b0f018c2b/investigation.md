# Investigation: SWA prefill head-of-line starvation

Upstream issue: https://github.com/sgl-project/sglang/issues/31205

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2279

The prepared base already contains the relevant correction in
`python/sglang/srt/managers/schedule_policy.py`. `_swa_req_never_fits` compares
the request's admission budget with `token_to_kv_pool_allocator.size_swa`. When
the budget can never fit the drained pool, `add_one_req` uses `_swa_chunk_cap`
to admit a page-aligned partial prefill instead of repeatedly returning
`NO_TOKEN`. Requests blocked only by temporary SWA pressure continue to defer.

The added regression uses the report's material geometry: SWA capacity 20,992,
chunk limit 32,768, page size 512, and prompt length 20,742. Patching out the
escape-hatch decision reproduces the old `NO_TOKEN` result with no admitted
range. The unmodified implementation admits a 19,968-token first chunk. Separate
cases cover whole-request admission below the page-rounded boundary, chunking at
exact capacity, and deferral when only current availability is insufficient.

No production code was changed because the checked-out implementation and the
new issue-specific regression demonstrate that the reported scheduler defect is
already corrected. Raw pytest output is retained under `raw/`.

This was not a full DeepSeek-V4-Pro, HTTP, PD-disaggregated, TP4+TP4, mooncake,
multi-node, or 8xB300 reproduction: the required model weights and topology were
not available. The deterministic test validates scheduler admission arithmetic
only. No GPU kernel is involved, and no native library rebuild was required.
