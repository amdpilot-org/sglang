# MiMo-V2.5-Pro-W8A8 correction generation 2

Upstream issue: https://github.com/sgl-project/sglang/issues/37755

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1245

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1128

Independent review PR: https://github.com/amdpilot-org/sglang/pull/1211

## Outcome

Candidate commit `9f5b71e3aae3ad3a53b8c36683fd4d760c8cb22e` is rejected as a correction for the reported issue. Its complete six-test regression passes unchanged against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and also passes against the exact candidate code. The candidate has no production or native-code diff relative to that base. Therefore there is no issue-specific failing-before/passing-after result.

The synthetic tests are retained because they cover the base's existing MiMo fused-QKV shard merging, incompatible-TP rejection, QKV deinterleaving/requantization order, and projection-layout parsing. They are test hardening only and are not evidence that the reported ModelSlim checkpoint loads accurately.

The prepared base already contains MiMo fused-QKV loading logic in `python/sglang/srt/models/mimo_v2.py`, plus a manual Ascend GSM8K test for `solinliu/MiMo-V2.5-W8A8`. The reported `/home/weights/MiMo-V2.5-Pro-W8A8` checkpoint is absent. This host also has no Ascend/CANN installation, no `torch_npu`, no `/dev/davinci*`, and no two-node TP32/DP4 topology. The only assigned accelerator is one AMD Instinct MI350X.

Consequently no production correction is justified here: the actual checkpoint tensor names, shapes, ordering, and scale metadata could not be inspected; the Ascend/CANN load and serving path could not be executed; and generated-token equivalence or semantic accuracy could not be measured. A synthetic GPU calculation passed on gfx950, but it validates only the existing tensor transformation, not MiMo-V2.5-Pro-W8A8 or Ascend execution.

## Reproduction

Recorded base implementation with the exact candidate test file:

```bash
PYTHONPATH=/job/repo/python SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 /tmp/amdpilot-repo-j-ea9758de30c2/venv/bin/python -m pytest -q /tmp/amdpilot-repo-j-ea9758de30c2/candidate-9f5b71e/test/registered/unit/models/test_mimo_v2_fused_qkv_weights.py
```

Result: `6 passed`, exit 0.

Exact candidate implementation and exact candidate test file:

```bash
PYTHONPATH=/tmp/amdpilot-repo-j-ea9758de30c2/candidate-9f5b71e/python SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 /tmp/amdpilot-repo-j-ea9758de30c2/venv/bin/python -m pytest -q /tmp/amdpilot-repo-j-ea9758de30c2/candidate-9f5b71e/test/registered/unit/models/test_mimo_v2_fused_qkv_weights.py
```

Result: `6 passed`, exit 0.

Consolidated branch:

```bash
PYTHONPATH=/job/repo/python SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 /tmp/amdpilot-repo-j-ea9758de30c2/venv/bin/python -m pytest -q test/registered/unit/models/test_mimo_v2_fused_qkv_weights.py
```

Result: `6 passed`, exit 0.

Raw outputs are in `raw/`.
