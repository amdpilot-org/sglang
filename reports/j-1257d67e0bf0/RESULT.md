# Independent review of PR 2927

Candidate: https://github.com/amdpilot-org/sglang/pull/2927 at
`88a6a9ea6e4ad7751e24a119317e4c8d3c8505c2`

Upstream issue: https://github.com/sgl-project/sglang/issues/38819

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2961

Recommendation: **request changes**. The candidate is a valid partial runtime
fix for C2 boundary-state transfer, not a full resolution of the original
feature request.

On the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the
approved N=101/R=2 case raises `ValueError`, N=101/R=8 returns five rows, and
even N=102/R=8 returns six rows. At the exact candidate commit, its focused
Mooncake/NIXL/wire suite passes (93 tests and 17 subtests). An independent
104,000-case sweep covers distinct source/destination request slots, legal ring
sizes 2/4/8/16/32, lengths 0..259, odd/even boundaries, and wraparound with no
failures. A real AMD Instinct MI350X copied the independently selected 4096-byte
FP32 row from source slot/R=1/2 to destination slot/R=3/8 and left every other
destination row unchanged.

The source import resolves to `/job/repo/python/sglang/__init__.py`; the GPU
probe used Torch 2.11.0+rocm7.2 and HIP 7.2.26015. No native source changed, so
there was no candidate native library to rebuild. The prepared architecture
cannot run DSpark: `_handle_dspark` accepts device strings beginning with only
`cuda` or `npu`, while this node is ROCm/HIP. No DeepSeek-V4.1 target/draft
weights or real multi-process/cross-node Mooncake/NIXL topology were available.

Consequently, the candidate does not demonstrate the original acceptance
criteria: end-to-end PD + DSpark generation; streaming or concurrency; chunked
prefill and odd prefix restoration with distinct H and N; first verify,
rejection, commit, C2 reuse, or Engram history; cancellation and stale-write
prevention; slot reuse and transfer ordering; parity, speculative acceptance,
or throughput. Decode radix cache also remains explicitly incompatible with
speculative decoding. The candidate documentation correctly calls the path
unqualified, but that honesty does not make the original feature complete.
