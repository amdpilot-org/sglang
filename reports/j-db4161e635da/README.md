# Investigation of hybrid Mamba prefix-cache pressure

Upstream issue: https://github.com/sgl-project/sglang/issues/36935

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3328

## Result

The reported `kv_pool_tokens / chunked_prefill_size` coupling is not present at
the recorded base, `358c163250ad3b1f62939b01ce1314a0a31a0365`. No source fix
was added because the current implementation already chooses donated prefill
checkpoint depths on `mamba_checkpoint_grid(tree_page)`, defined as
`lcm(mamba_cache_chunk_size(), tree_page)`, rather than on
`chunked_prefill_size`.

The relevant implementation is in
`python/sglang/srt/managers/schedule_batch.py::_mamba_radix_cache_v2_req_prepare_for_extend`
and `python/sglang/srt/runtime_context.py::mamba_checkpoint_grid`. In particular,
the extend path gates tracking on `extend_range.length >= checkpoint_grid` and
rounds `mamba_last_track_seqlen` to that grid. `chunked_prefill_size` is not read
by this path. This makes checkpoint density independent of the scheduler's
prefill chunk size, addressing the coupling described by the report.

The focused checkpoint-depth suite passed five cases, including widened and
unwidened radix pages plus independent interval/grid boundaries. The focused
LRU test also passed and confirms the current intentional behavior: a match
refreshes only the consumed Mamba checkpoint, not all ancestors. Finite Mamba
pool pressure can therefore still evict old checkpoints; what is no longer
reproduced is the reported control of checkpoint demand by
`chunked_prefill_size`.

## Limitations

The reported workload used Qwen3.8-27B-FP8 on an NVIDIA RTX PRO 6000 (sm_120).
The assigned device is one AMD Instinct MI355X (gfx950), and the required model
weights were not available. Consequently this investigation does not claim a
full-model, NVIDIA, semantic-accuracy, or production HTTP reproduction. The
tests exercise the actual scheduler and cache implementation deterministically
on CPU. No native code changed, so no native rebuild was applicable.

Raw outputs are retained in `reports/j-db4161e635da/raw/`.
