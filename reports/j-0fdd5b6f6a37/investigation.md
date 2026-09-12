# Investigation evidence

- Upstream issue: https://github.com/sgl-project/sglang/issues/32475
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/2063
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Prepared interpreter: `/tmp/amdpilot-repo-j-0fdd5b6f6a37/venv/bin/python`

The base source still assigned token capacity directly to the two fields named
`kv_*_blocks`. A direct call through the actual `emit_kv_metrics` implementation
recorded `8192` total and `4096` active for 8192 tokens at 50% usage. Applying
the scheduler's page size of 16 independently gives 512 total pages and 256
active pages.

Repository and upstream history searches found two open, unmerged proposed
fixes: https://github.com/sgl-project/sglang/pull/32499 and
https://github.com/sgl-project/sglang/pull/32558. Neither correction was present
at the prepared base.

The implemented fix uses the same floor-page convention already present in
`Scheduler.emit_metrics_constants`: `max_total_num_tokens // page_size`.
Raw failing-before and passing-after outputs are retained in `raw/`.
