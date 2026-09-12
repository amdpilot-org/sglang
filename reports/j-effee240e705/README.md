# Independent review of amdpilot-org/sglang PR 2743

Reviewed exact candidate commit `9a7fdb8c7f520ec561be7c7f45a4a5352cc7a7a3` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the complete original issue contract.

Recommendation: **request changes**. The candidate is a substantial partial implementation, but it does not fully resolve the original feature request.

Verified behavior:

- selector-lattice selected-path confidence and cross-request prefix prioritization;
- SPS-table/fixed-width budget plumbing;
- ragged verification compatibility and cutoff-related regression coverage;
- ROCm execution of shared modified kernels against independent PyTorch references;
- independent mixed-quality allocation `[4, 1]` and anchor floor `[1, 1]` on one MI350X.

Material gaps:

- no DFlash2 trained confidence/survival head implementation or checkpoint path;
- no DFlash2 optional sequential temperature scaling calibration;
- no compatible DFlash2 weights, so live serving, graph replay, semantic equivalence, and throughput remain unverified.

No native C++/CUDA/HIP/FlyDSL files changed, so a native rebuild was not applicable. Raw commands and outputs are retained in this directory. `result.json` contains the machine-readable verdict and limitations.
