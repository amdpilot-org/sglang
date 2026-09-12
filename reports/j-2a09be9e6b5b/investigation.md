# Qwen3.8 DSpark forced-reject investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/35150

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1432

The prepared base did not contain the beta-alignment candidate. Upstream PR
https://github.com/sgl-project/sglang/pull/36014 remains open and describes the
same change as partial. The latest upstream issue comment independently reports
equal transition operands followed by a differing SSM transition output, and
reports that beta alignment reduces but does not eliminate model-level errors.

On the assigned gfx950, a deterministic production-shape (`K=V=128`) fixture
reproduced the concrete mismatch before the source change. Packed GDN decode
materializes `sigmoid(b)` in BF16 before its FP32 recurrent update, while the
TARGET_VERIFY kernel retained the value in FP32. For `b=-0.5`, outputs differed
at 128/128 positions with maximum absolute error 0.000244140625.

The correction is restricted to non-KDA calls with state updates disabled,
which is the TARGET_VERIFY scratch-state path. It restores the packed-decode
activation-dtype boundary. Exact output and FP32 state parity passes at
`b=-20`, `-0.5`, `0`, and `20`. Existing direct state-update, non-contiguous
GDN, and KDA tests pass and retain their previous behavior.

Raw logs are retained under
`/tmp/amdpilot-repo-j-2a09be9e6b5b/logs/`. The tested source is
`/job/repo/python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py`.
This is a Triton/Python source change; there is no native library to rebuild.

The required Qwen3.8 target/draft weights and NVIDIA NVFP4 environment were not
available. Consequently this result is `candidate_verified`, not `fixed`; the
reported cumulative 96-token serving divergence remains unverified here.
