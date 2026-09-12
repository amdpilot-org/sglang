# MiMo audio DP-attention investigation

At base `358c163250ad3b1f62939b01ce1314a0a31a0365`, `AudioEncoderAttention`
constructed `VisionAttention` without `use_dp_attention_reduce`. The latter still
sharded QKV over `attn_tp_size`, but its `RowParallelLinear` consequently selected
the full tensor-parallel all-reduce. That matches the communicator mismatch
reported in the source issue.

The correction passes `is_dp_attention_enabled()` to `VisionAttention`. The
regression covers both boundaries so ordinary non-DP-attention construction keeps
the existing full-TP behavior while DP attention selects the attention-TP group.

Current-fix inspection found open upstream PR
https://github.com/sgl-project/sglang/pull/37060 at head
`365904382d8c19701165c8030e4065b7b2fc0456`. It independently contains the same
source correction, but it is not present in the prepared base. Its captured
metadata and patch are retained in this directory.

The reported distributed hang could not be executed on the assigned single AMD
Instinct MI355X GPU, and MiMo-V2.5 weights were unavailable. The retained GPU
probe proves only that the assigned ROCm device executed code; it is not presented
as a model, transport, CUDA/NCCL, or multi-rank reproduction.
