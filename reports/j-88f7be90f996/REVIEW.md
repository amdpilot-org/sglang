# Independent review of PR 869 at e1bce2f

Upstream issue: https://github.com/sgl-project/sglang/issues/38118

Mirror issue: https://github.com/amdpilot-org/sglang/issues/901

Candidate: https://github.com/amdpilot-org/sglang/pull/869 at `e1bce2f4784b8af0cd8565f2cde9dfcae44a198e`

Recommendation: **request changes**.

The candidate fixes the original non-beam defect. The unchanged candidate regression fails on the recorded base in the issue-specific case because the queued request stays parked, then passes at the exact candidate SHA because the held chunk row is credited during admission.

An independent boundary test found an incomplete Beam interaction in the changed accounting. Given one scheduled width-1 chunk, two genuinely free request-pool rows, a PP ceiling with room, and a queued width-2 request, the width-2 request fits. The candidate nevertheless leaves it queued. It computes `(2 free + 1 held credit) // 2 == 1`, then compares that with `len(can_run_list) == 1` for the width-1 chunk and declares the batch full. The candidate's Beam test uses three free rows and misses this exact-capacity case.

This counterexample predates the candidate in the sense that the base also under-admits this scenario, but PR 869 explicitly modifies and claims Beam-aware slot accounting. The original issue did not use Beam, so `fully_resolves_original` is true while the broader candidate should be revised before acceptance.

No native source changed, so no native rebuild was applicable. Imports on base and candidate resolved to the checkout. Raw logs and the independent test are retained outside the checkout under `/tmp/amdpilot-repo-j-88f7be90f996/evidence`.

The full reported workload could not be reproduced: this environment has one AMD Instinct MI350X (`gfx950`, ROCm 7.2), not eight NVIDIA B200 GPUs, and does not have the GLM-5-FP8 weights or the CUDA/TRT-LLM NSA stack. A small GPU arithmetic check only confirmed access to the assigned device; it is not proof of scheduler correctness.
