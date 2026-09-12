# MiniMax sparse-prefill score-width investigation

The checked-out implementation at base `358c163250ad3b1f62939b01ce1314a0a31a0365`
still allocated the score tensor from caller-provided `max_seqlen_k`, while its
Triton store bounds came from live `seq_lens`.

On the assigned gfx950 GPU, the direct `[64, 4096]` fixture returned 34 differing
top-k entries when only `max_seqlen_k` changed from 4096 to stale 64. All
differences were in the long sequence. The committed regression also failed with
the guard temporarily removed (116/1040 differences on that allocator layout).

The correction sizes the allocation to the larger required block-column count
and warns only when the live lengths would otherwise exceed the allocation. It
does not change the allocation or warn for equal width or an underestimate that
still rounds to the same block count.

Raw issue snapshots, GPU identity, reproduction output, test output, and lint
output are retained in `raw/`. See `result.json` for commands and limitations.
