# Grammar token synchronization singleton-group investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/35826

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1291

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

The prepared implementation still called `torch.distributed.all_reduce` directly
from `Sampler._sync_token_ids_across_tp` whenever grammar sampling or the forced
sync environment option was active. There was no check for the selected process
group's size. Under DP attention that selected group is the attention-TP group,
which can legitimately contain one rank.

The earlier broad constrained-decoding change in upstream PR #13947 contained a
group-size optimization but was wholly reverted by PR #16845 after breaking DP
attention. The current base retained direct synchronization, so this change adds
only a size check at the existing collective call site.

The failing-before log shows that both triggering modes reached `all_reduce` for
a singleton group. After the fix, singleton groups skip the collective, groups
with two ranks retain `ReduceOp.MIN`, and an untriggered batch does not even query
the group size. The GPU check uses a real singleton process group and tensor on
the assigned gfx950 device, with `all_reduce` replaced by a trap to demonstrate
that the call site is not reached.

This environment cannot reproduce the reporter's eight-rank NVIDIA B300 topology
or CUDA/NCCL allocation signature. The evidence qualifies the sampler control
flow correction on the checked-out implementation, not a full model-serving or
multi-node reproduction.
