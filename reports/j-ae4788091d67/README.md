# Investigation evidence

Upstream issue: https://github.com/sgl-project/sglang/issues/37215

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1004

Related upstream candidate: https://github.com/sgl-project/sglang/pull/37458

The prepared base still selected a dynamic `nccl_port` with `bind(0)`, then
closed its temporary reservation before the DP worker created its TCPStore.
On this host the Linux ephemeral range is `32768-60999`; all 64 sampled
`get_free_port()` results fell inside that range. This confirms the allocator
uses the same namespace that the kernel assigns to outgoing client sockets.

The regression added for the reported `10000-61000` range failed before the
implementation because no non-ephemeral rendezvous allocator existed. After
the change, 8/8 selected rendezvous ports were outside this host's ephemeral
range and eight real `torch.distributed.TCPStore` servers bound those ports
concurrently.

The implementation follows the still-open upstream candidate PR, adapted to
the prepared main tree. It also excludes ports already planned for HTTP,
per-DP HTTP, gRPC/sidecar, gated launch, disaggregation, encoder/engine
bootstrap, and remote weight-loader listeners. User-specified `--nccl-port`
values remain unchanged.

No model weights were used. The available device was one AMD Instinct MI355X
(`gfx950`), not eight NVIDIA H800 GPUs, so full Qwen3-Embedding serving and the
reported intermittent 8-GPU launch remain unverified.
