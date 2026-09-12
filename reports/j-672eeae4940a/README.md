# Correction investigation report

Candidate reviewed and preserved: https://github.com/amdpilot-org/sglang/pull/2504 at exact commit `81bdd3011b40fea2c36354e3ae6f5b5d94df4189`.

Independent review investigated: https://github.com/amdpilot-org/sglang/pull/2573.

The earlier review's concrete evidence mismatch reproduced in that prepared
environment: its device query reported MI355X while the retained candidate
record said MI350X. That correction is intentionally scoped to that run.
`gfx950` does not by itself establish a marketed model name, and later prepared
environments may identify the assigned device differently.

The candidate's valid regression remains intact. At its exact commit, all
three tests passed. After integration and evidence correction, the same three
tests pass. An independent GPU check also loaded separate `(1408, 64)` BF16
gate and up tensors through the actual `FusedMoE._load_w13` implementation
into the two halves of a `(2816, 64)` destination and compared both halves.
The candidate's retained obsolete-layout failure remains the failing-before
evidence for the production fix already present on `main`.

No Ascend device, CANN/torch_npu runtime, or reported DeepSeekV3.2 weights were
available. Therefore actual `npu_format_cast`, Ascend grouped matmul, and a
full Engine or HTTP `update_weights_from_disk` scheduler-survival run remain
unverified. Those limitations do not justify a speculative source change.
