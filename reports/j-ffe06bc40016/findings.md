# Investigation result

The reported test defect is already fixed at base commit
`358c163250ad3b1f62939b01ce1314a0a31a0365`.

The current
`test/registered/unit/hardware_backend/mlx/test_mlx_reference_correctness.py`
defines `_truncate_at_eos` and applies it to both solo and batched streams before
comparison. This is the proposed correction from the issue: equality is required
up to and including the first EOS, while the fixed 24-token horizon still keeps
every request in the batch for cache-isolation coverage.

GitHub's commit API identifies the introducing upstream commit as
`a26735e43e7f33b5c929d2ad24556d4da2be4151`, merged through PR #34166. Its patch
adds `_truncate_at_eos`, replaces the full-horizon assertion, and records the
same measured fixture behavior from the issue (EOS at index 2, divergence at
index 6).

`regression_check.py` exercises the actual helper from the checked-out file. It
covers empty input, EOS first, EOS in the middle, multiple EOS tokens, and no
EOS. It also constructs streams equal through EOS but different afterward: the
old full-horizon premise fails and the checked-out EOS-bounded comparison passes.

The actual MLX model test collected successfully but skipped both tests because
this Linux gfx950 host has neither Apple Silicon nor `mlx`/`mlx_lm`. The assigned
AMD Instinct MI350X was visible and executed a trivial ROCm tensor operation, but
that is not MLX-path execution and is not claimed as reproduction. No model
weights were downloaded.

A separate gpt-oss MLX e2e file currently contains a similar full-horizon
assertion. It targets a different model and sliding-window path and cannot be
qualified on this host, so it was not changed or presented as resolution of the
reported Qwen MLX unit issue.

## Commands and observed results

```text
/tmp/amdpilot-repo-j-ffe06bc40016/venv/bin/python -m pytest \
  test/registered/unit/hardware_backend/mlx/test_mlx_reference_correctness.py -q
=> exit 0; 2 skipped (requires mlx + mlx_lm, Apple Silicon only)

PYTHONPATH=/job/repo/python \
  /tmp/amdpilot-repo-j-ffe06bc40016/venv/bin/python \
  reports/j-ffe06bc40016/regression_check.py
=> exit 0; 5 boundary cases passed; old comparison fails; current comparison passes
```

Source path: `/job/repo/test/registered/unit/hardware_backend/mlx/test_mlx_reference_correctness.py`

Native path: not applicable; no native source was changed or rebuilt.
