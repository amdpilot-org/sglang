# Independent review of PR 1605

Reviewed `amdpilot-org/sglang` PR 1605 at exact commit
`684b9cf4091ef76471236cc035e257573e68d9a3` against upstream issue
https://github.com/sgl-project/sglang/issues/34442 and mirror issue
https://github.com/amdpilot-org/sglang/issues/1641.

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` still constructs
MoonViT's two tensor-parallel `RowParallelLinear` projections with
`tp_size=attn_tp_size` but leaves their reductions on the default global TP
communicator. The candidate sets `use_dp_attention_reduce` on exactly those two
sites: the MLP `fc1` projection and the vision-attention output projection.

The candidate regression was installed alone on the recorded base and failed in
the DP-enabled TP4/attention-TP2 case. At the exact candidate commit it passed,
as did the full Kimi-VL unit file. An independent forward-path test constructed
the real MoonViT layer and executed both affected row-parallel projections with
instrumented collectives. With DP attention enabled, both called the attention
TP group and neither called the global TP reducer. With DP attention disabled,
both called the global TP reducer and neither called the attention TP group.

The prepared interpreter imported SGLang model and linear sources from this
checkout. The candidate changes no native source, so no native rebuild was
required. The host exposes one AMD Instinct MI350X with gfx950 ISA. The original
TP4/DP2 HTTP deadlock could not be reproduced end-to-end on one GPU, and the
Kimi-VL weights were not available. Accordingly, this review verifies the
communicator-selection defect and correction, but does not claim a four-rank
serving reproduction, NVIDIA H100 execution, model semantics, or multi-node
behavior.

Recommendation: **accept**. The change fully addresses the identified original
contract at both offending reduction sites; no remaining source counterexample
was found. End-to-end distributed serving remains an environment limitation.
