# Investigation of sglang#37648

Upstream issue: https://github.com/sgl-project/sglang/issues/37648

Mirror issue: https://github.com/amdpilot-org/sglang/issues/933

## Outcome

The reported TP8 GLM-5.3-MXFP4 failure could not be reproduced in the assigned
environment. The report requires eight MI355X GPUs, the GLM-5.3-MXFP4 weights,
EAGLE, and a heterogeneous 32-client workload sustained for roughly 24 minutes.
This job has one MI350X (`gfx950`) and no model weights. A one-GPU tiny Llama
fixture would only exercise transport and generic engine execution, so it would
not qualify the GLM architecture, MXFP4/AITER path, TP8 communication, or the
long-context concurrency failure and was not substituted for the report.

No narrow source correction is justified by the available evidence. No code was
changed.

## Related-fix review

- The original issue is still open and has no comments as of this investigation.
- The similar issue https://github.com/sgl-project/sglang/issues/23784 was closed
  without a comment, linked resolution, or identified fixing commit.
- https://github.com/sgl-project/sglang/pull/23461 remains open. It fixes an
  `IndexError` caused by ragged EAGLE3 draft metadata being supplied to unified
  attention. That explicit failure differs from the delayed KFD memory-access
  fault reported here, and the original report says the fault occurs around an
  untuned BF16 GEMM fallback. Therefore PR #23461 is not evidence that #37648 is
  fixed.
- Current source has dedicated paged EAGLE draft allocation and KV-index logic,
  plus a GPU regression grid covering page sizes 1, 16, and 64, top-k values 1,
  4, and 8, multiple draft-step counts, boundary sequence lengths through
  100,000, and both historical and token-block launches. Those tests passed,
  but they isolate index generation and do not reproduce a server or AITER MoE
  execution.

## GPU evidence

The assigned GPU was an AMD Instinct MI350X, `gfx950`, with 270,566,162,432
bytes of VRAM. PyTorch was 2.11.0+rocm7.2 (HIP 7.2.26015), and AITER was loaded
from `/sgl-workspace/aiter` at commit
`4ad99832823dde2315b361cbd3b54b1c5c12acd5`.

The three exact BF16 GEMM shapes printed immediately before the reported fault
were executed through `aiter.tuned_gemm.tgemm.mm`. In all three cases the AITER
dispatcher independently confirmed the untuned `torch` solution 0 fallback.
After explicit GPU synchronization, every output was finite. Four rows spanning
each output were compared against CPU float32 matrix multiplication:

| M | N | K | max absolute error | max relative error |
|---:|---:|---:|---:|---:|
| 84333 | 2624 | 6144 | 0.00038709 | 0.00383549 |
| 84333 | 512 | 6144 | 0.00023668 | 0.00377296 |
| 84333 | 6144 | 256 | 0.00006071 | 0.00388455 |

This rejects a deterministic single-launch numerical or memory fault in these
three fallback GEMMs on the assigned GPU/software stack. It does not reject an
intermittent fault caused by prolonged concurrency, TP8 communication, EAGLE
state, graph reuse, allocation lifetime, or another kernel adjacent to the log.

Raw outputs are retained outside the worktree at:

- `/tmp/amdpilot-repo-j-21e20452a730/evidence/gpu.txt`
- `/tmp/amdpilot-repo-j-21e20452a730/evidence/environment.txt`
- `/tmp/amdpilot-repo-j-21e20452a730/evidence/reported_gemm_shapes.txt`
- `/tmp/amdpilot-repo-j-21e20452a730/evidence/spec_kv_indices_grid.txt`

## Remaining reproduction requirement

Resolving the issue still requires the reported GLM-5.3-MXFP4 checkpoint and an
eight-GPU MI355X/gfx950 node. The useful next experiment is the reporter's
planned current-main matrix: EAGLE on/off, graph batch 2/16/64 and eager mode,
client concurrency 8/16/32, and AITER MoE/default-path ablation, while retaining
a usable GPU core dump and per-rank logs. A failing-before/passing-after server
regression cannot honestly be supplied until one branch of that matrix
reproduces the fault and identifies a source boundary.
