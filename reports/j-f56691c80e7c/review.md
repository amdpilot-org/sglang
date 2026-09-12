# Independent review of candidate PR 1317

Upstream issue: https://github.com/sgl-project/sglang/issues/36071

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1361

Candidate: https://github.com/amdpilot-org/sglang/pull/1317 at `40a46418db2ee8fdf9e4ae0fd209e34541180542`

## Recommendation

Accept the candidate as a narrowly justified source fix, with the explicit qualification that the full original eight-GPU/model workload could not be reproduced in the assigned one-GPU environment. No counterexample was observed in the executable scope. `fully_resolves_original` is therefore recorded as false rather than treating the missing TP8/DP4 and model evidence as proof.

## Findings

- The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` uses direct graph-owned input registration during capture unless memory-saver CUDA graphs are enabled. It has no DSA plus DP-attention exception.
- The candidate selects the existing stable pre-registered copy-in path exactly when decode uses DSA and DP attention is enabled. It forwards the policy to both the AITER and SGLang custom-all-reduce implementations, including the deterministic SGLang path.
- The candidate source change matches the live upstream proposed fix in sgl-project/sglang PR 36088. No native source changes are present.
- The candidate regression fails 9/9 on the recorded base and passes 9/9 on the candidate. The broader focused suite passes 104 tests and 140 subtests.
- Independent runtime-context cases verified the actual published configuration path for: issue configuration, DSA without DP attention, split decode-only DSA, and prefill-only DSA.
- On the assigned AMD Instinct MI350X (`gfx950:sramecc+:xnack-`), the candidate fixture captured four graphs sharing one stable BF16 staging address and replayed them in reverse order. Observed FP32 sums `[8192, 4096, 2048, 1024]` exactly matched the independent expected values.

## Classification and limitations

This is a source fix, not test-only hardening. The code changes the capture input contract from graph-owned addresses to one stable registered address for the affected configuration. However, the available hardware is one gfx950 GPU, not the required eight GPUs. The GLM-5.2-MXFP4 weights were not available. Consequently, the original peer-IPC collective failure, TP8/DP4 graph skew, all-NaN logits, repeated token zero, and throughput were not independently reproduced or verified end to end.

The single-GPU graph fixture proves stable-address behavior on the target architecture but cannot prove multi-rank custom-AR correctness. No native rebuild was applicable because the candidate modifies Python and tests only.

Raw logs, fetched issue/PR metadata, import paths, candidate diff, and live upstream comparison are retained outside the checkout at `/job/review-evidence-j-f56691c80e7c/`.
