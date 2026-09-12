# ROCm router GEMM dtype investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/34857

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1480

The prepared base already contains the GLM-5.2-specific correction-bias fix:
`MoEGate` retains the checkpoint parameter as fp32 and the aiter grouped-top-k
path upcasts gating logits instead of downcasting an fp32 bias. Its existing
nine-test regression suite passes unchanged.

The remaining router GEMM defect reproduced on the assigned MI350X/gfx950:
`aiter_dsv3_router_gemm` returned bf16 for both `(M,K,N)=(8,7168,256)` and the
independent fallback case `(65,512,64)`. This change requests `torch.float32`
from `tgemm.mm`, matching every other `MoEGate` GEMM branch. The new GPU test
would fail at the output-dtype assertion on the recorded base and passes after
the change for both shapes.

## Precision limitation

The installed aiter dispatcher reported no tuned configuration for either
tested shape. After the change it returns an fp32 tensor, but comparison with
an independent `torch.nn.functional.linear(x.float(), w.float())` reference
shows bf16-level internal rounding remains in the fallback implementation. For
the production shape, the returned values were exactly the fp32 reference
rounded through bf16. Therefore this result verifies the reported output dtype
contract, not improved GLM-5.2 model accuracy or true fp32 accumulation inside
aiter. No GLM-5.2 weights or multi-GPU model evaluation were available.

Raw command output is retained in `raw/`.
