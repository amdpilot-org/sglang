# DeepSeek-V4.1 PD + DSpark correction review

Candidate: https://github.com/amdpilot-org/sglang/pull/2793 at `7d1a4b02e6ba07ac98aa4ac2aa211f7eccf2c89b`

Independent review: https://github.com/amdpilot-org/sglang/pull/2864

The candidate was reproduced before modification. Its focused suite passed (93
tests and 17 subtests), an independent 5,200-case mapping sweep passed, and an
MI350X copied the selected 4096-byte FP32 C2 row exactly while leaving adjacent
rows untouched.

The C2 correction is valid and retained. Against the recorded base, a legal ring
size of 2 was rejected, and N=101/N=102 with ring size 8 returned five/six rows.
The correction returns one endpoint-local row for the odd boundary and no row
for the even boundary.

The candidate documentation claimed DeepSeek-V4.1 PD + DSpark support despite
the absence of the required end-to-end qualification. This consolidation narrows
that statement to the verified row-transfer primitive and explicitly says it is
not evidence of a supported deployment configuration.

The original feature remains unverified: no DeepSeek-V4.1 target/draft weights,
real Mooncake/NIXL multi-process topology, or CUDA/NPU execution platform were
available. Streaming, concurrency, chunked/prefix restoration, verify/rejection,
commit/reuse, cancellation/slot reuse, parity, acceptance, and throughput remain
unmeasured. Decode radix cache is still guarded as incompatible with speculative
decoding. Raw evidence and exact claims are retained in this report directory.
