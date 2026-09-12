# DP TCPStore port-race correction

Upstream issue: https://github.com/sgl-project/sglang/issues/37215

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1197

Candidate parent: https://github.com/amdpilot-org/sglang/pull/1062 at
`e5ce94b77e1db3a2c4641abf97ebf391926af8ba`.

Independent review parent: https://github.com/amdpilot-org/sglang/pull/1159.

## Findings

All three review counterexamples reproduced against the exact candidate. A real
CPU `torch.distributed.TCPStore` failed with `EADDRINUSE` after a listener took
the selected non-ephemeral port during the controller's close-before-bind gap.
With `ServerArgs(dp_size=8, nccl_port=30101)`, every allocation returned 30101
and repeated reservation failed. Mocked unavailable/exhausted procfs cases
returned the old ephemeral allocator's result.

The correction retains the candidate's useful non-ephemeral allocation for
multi-rank groups, but removes TCP rendezvous entirely for the reported
`DP=8, TP=1` topology: each independent singleton process group now uses an
in-process PyTorch `HashStore`. Explicit `--nccl-port` is treated as a base and
offset by DP rank (`30101` through `30108`), with overflow rejected. Automatic
allocation now fails closed when the Linux ephemeral range cannot be read or
all safe ports are exhausted instead of silently restoring the race.

## Validation

The focused suite passed 47 tests. In an independent real execution, port
30101 was occupied by a listening socket while a one-rank NCCL process group
initialized through the corrected path and completed an `all_reduce` on the
assigned AMD gfx950 GPU; the result remained 7.0. The same probe confirmed the
eight explicit DP ports were unique (`30101..30108`).

No native source changed, so no native rebuild was required.

## Limitations

The prepared host has one AMD gfx950 GPU with ROCm, not eight NVIDIA H800 GPUs,
and Qwen3-Embedding weights were unavailable. Therefore a full eight-GPU model
launch and CUDA/NCCL behavior were not reproduced. Multi-rank (`TP>1` or
`PP>1`) groups still require TCPStore; non-ephemeral selection mitigates OS
ephemeral reuse but cannot prevent an arbitrary unrelated listener from
deliberately binding the released port before TCPStore without support for
adopting a pre-bound socket in PyTorch.
