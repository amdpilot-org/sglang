# Investigation report: sglang#32968

The source selected for this task already contains the write-side correction from
upstream PR #32477 (`ee678910f7000aa43886f218de0e159bf418f1b5`). The JIT KV-cache
store has a `reserved_skip_index` argument defaulting to slot 0, and the device
kernel skips copying K/V rows whose destination is that reserved CUDA-graph
padding slot while still executing the PDL trigger.

No additional product-code change is justified. The existing regression in
`test/registered/kernels/ops/kvcache/test_store_cache.py` injects NaNs into
padding rows and covers both index dtypes and split counts 1, 2, and 4. It also
covers the independent opt-out boundary (`reserved_skip_index=-1`).

## Reproduction and validation

On the assigned AMD Instinct MI350X (`gfx950:sramecc+:xnack-`), the explicit
opt-out reproduced the pre-fix write: both reserved K/V rows became entirely NaN
(1,024 NaNs each). With the default implementation, the same inputs left slot 0
bit-for-bit unchanged and wrote non-reserved slots exactly.

The complete focused suite passed: 358 tests. It compiled and loaded JIT
libraries from the checked-out source under the private runtime cache. Raw logs,
the upstream PR metadata/patch, and generated library paths/checksums are in
`raw/`.

## Limitations

This is a deterministic kernel-level validation of the reported write-side
mechanism, not a full Kimi-K3 serving reproduction. The official Kimi-K3 and
DSpark weights were not available. The assigned machine has one AMD gfx950 GPU,
not the report's eight NVIDIA Blackwell GPUs, so TP=8, 90k--237k-token traffic,
CUDA graphs on SM10x, DSPARK acceptance behavior, SSE `[PAD]` storms, and the
TensorCompare CUDA assertion were not reproduced. The evidence does not qualify
model semantics, tokenizer behavior, or distributed execution.

## Commands

```bash
env XDG_CACHE_HOME=/tmp/amdpilot-repo-j-bfe220808037/cache \
  TORCH_EXTENSIONS_DIR=/tmp/amdpilot-repo-j-bfe220808037/cache/torch_extensions \
  SGLANG_JIT_CACHE_DIR=/tmp/amdpilot-repo-j-bfe220808037/cache/sglang_jit \
  /tmp/amdpilot-repo-j-bfe220808037/venv/bin/python -m pytest -q \
  test/registered/kernels/ops/kvcache/test_store_cache.py -x
```

The standalone numerical check is recorded verbatim in
`raw/gfx950_reserved_slot_evidence.log`; it used the same environment and
`store_cache` implementation with `num_split=4` and int32 indices.
