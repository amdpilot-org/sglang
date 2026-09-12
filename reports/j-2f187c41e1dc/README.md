# RMSNorm artifact-delivery qualification

This report qualifies the requested SGLang RMSNorm workflow at source revision
`358c163250ad3b1f62939b01ce1314a0a31a0365`; it is not an upstream issue-fix
claim or a qualification of all repository behavior.

Run from the repository root:

```bash
/tmp/amdpilot-repo-j-2f187c41e1dc/venv/bin/python reports/j-2f187c41e1dc/probe_rmsnorm.py
```

The probe requires exactly one visible GPU, uses deterministic float32-generated
inputs rounded to float16 or bfloat16, calls the normal `RMSNorm` module path,
and compares its output with an independently expressed float32 Torch formula.
It covers widths 4096 and odd 4097 with three rows per case. See `result.json`
for tolerances, measurements, paths, commands, exit codes, and limitations;
`raw-success.json` preserves the successful output verbatim in structured form.

On this environment the normal ROCm dispatch resolved to `forward_hip`.
`vllm._custom_ops` was not installed, so that method intentionally invoked
SGLang's `forward_native` float32 fallback. No native source, package, device
visibility, or node-state change was made, and no rebuild or reinstall occurred.
