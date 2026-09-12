# Independent review of PR 1627

Reviewed exact candidate commit `93464111cb0ea89bd5166c17b25adcdd7c717fc1`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the
contract in https://github.com/sgl-project/sglang/issues/34539.

The prepared checkout initially matched the recorded base. Running the candidate's
regression from preserved external evidence against that base reproduced the defect:
LMCache and FlexKV each stored 9 entries with a caller limit of 4 in normal mode,
and 8 entries with a limit of 4 in speculative mode. Four backend subtests failed.

At the exact candidate commit, the regression passed (3 tests and 6 backend
subtests). Independent checks covered limits smaller than, equal to, and larger
than the backend-derived length for both backends, with `topk` values `None`, 1,
2, and 8. All 16 scenarios passed. The imported backend modules resolved to the
candidate checkout under `/job/repo/python/sglang/...`, not an installed wheel.

The two source changes apply `min(kv_committed_len, kv_len_to_handle)` after the
normal/speculative backend calculation and before both token and slot slicing.
This directly enforces the issue's maximum-length contract while preserving the
existing speculative formula. No remaining counterexample was found within that
contract. Recommendation: accept; the candidate fully resolves the original issue.

Limitations: LMCache and FlexKV optional packages/services were not installed, so
the review used dependency stubs while executing the actual SGLang backend methods.
The environment has one AMD Instinct MI350X (`gfx950`) with PyTorch 2.11.0 ROCm
7.2, differing from the issue's NVIDIA/CUDA environment. No GPU computation, model,
semantic, distributed, or multi-node test was relevant or performed. No native
source changed, and no native rebuild was required.
