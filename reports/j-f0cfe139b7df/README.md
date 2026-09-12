# AMDPilot interrupted-stream RMSNorm qualification

This report records two actual SGLang `RMSNorm` GPU operations on the assigned
AMD Instinct MI355X. Both used normal `forward_hip` dispatch and were compared
against independently expressed float32 Torch references at explicit tolerances.
The initial checkpoint nonce is `f0cfe139-20260912-a7c4-91d2-e6b83f5041aa`.

Reproduce from the prepared checkout and interpreter:

```bash
PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-f0cfe139b7df/venv/bin/python reports/j-f0cfe139b7df/probes/initial_rmsnorm_probe.py
PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-f0cfe139b7df/venv/bin/python reports/j-f0cfe139b7df/probes/second_rmsnorm_probe.py
```

The `*_raw.json` files retain inputs, weights, actual outputs, complete float32
references, and elementwise absolute errors. This qualification is platform
evidence only, not an upstream bug fix or a claim of platform recovery.
