# GLM-5.3 DCP no-RoPE MLA investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/38207

Mirror issue: https://github.com/amdpilot-org/sglang/issues/787

The prepared base already contains substantial DCP KV-layout work, but its
consolidated MLA scatter dispatcher left the no-RoPE branch outside that
contract. `set_mla_kv_buffer_kernel_norope` used widened allocator locations
directly against a per-rank buffer, while the RoPE branch filtered locations by
DCP owner and divided them by the DCP width. The no-RoPE branch also ignored
`reserved_skip_index`.

The regression uses a guard half after a simulated per-rank buffer. This makes
the pre-fix writes beyond physical capacity observable without intentionally
causing an illegal GPU access. On the base implementation, both rank cases
wrote raw virtual rows into that guard and the reserved-slot case wrote NaNs
into slot zero. After the change, each rank receives only its owned rows at
`virtual_loc // dcp_size`, the guard remains unchanged, and the reserved slot is
untouched.

Raw evidence is retained in `raw/`. The full B200 TP8/DCP8 GLM-5.3 serving
workload could not be run on the assigned single AMD gfx950 GPU without the
reported model weights. This result therefore validates the faulty kernel
contract and its correction, not full-model semantics or distributed serving.
