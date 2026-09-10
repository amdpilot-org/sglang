# Structured-input robustness evidence

## Scope

This job adds a bounded AMD/ROCm regression that drives the real
`sglang.srt.model_executor.model_runner.ModelRunner.forward` path with a locally
generated two-layer Llama configuration and SGLang's deterministic dummy-weight
loader. It does not start a server, download a checkpoint, or load a tokenizer.
The independent control is a raw PyTorch Llama implementation; it is not used as
a substitute for the SGLang engine.

Upstream context was read-only: sgl-project/sglang issue 35003 is the 2026 Q3 AMD
roadmap, has no comments, and does not contain a specific structured-input fix.
A repository and upstream PR search found no duplicate reduced-ModelRunner
structured-input workload.

## Reproduction

From the repository root:

```bash
mkdir -p /tmp/sglang-cache-j-3f6d35a4581a
PYTHONPATH=$PWD/python \
TRITON_CACHE_DIR=/tmp/sglang-cache-j-3f6d35a4581a \
/opt/venv/bin/python -m pytest -q -s \
test/registered/amd/test_llama_structured_input_robustness.py
```

The test generates `config.json` in a temporary directory with vocabulary 128,
hidden size 64, intermediate size 128, two layers, four attention heads, two KV
heads, and bfloat16 dtype. Dummy weights use SGLang's fixed `[-1e-3, 1e-3]`
initialization. The state-dict tensor size is 180,464 bytes (about 0.17 MiB),
well below the 4 GB limit.

## Numerical contract

- SGLang output: complete prefill hidden states, shape `(32, 64)`, dtype
  `torch.bfloat16`.
- Independent control: raw PyTorch RMSNorm, NeoX RoPE, causal GQA attention,
  SiLU MLP, and final RMSNorm in float32.
- Gate: all values finite and `torch.testing.assert_close` with `rtol=1e-2`
  and `atol=2e-6`.
- Cases: six identical-shape `(2, 16)` token-ID blocks: zeros, tiny finite IDs,
  mixed magnitudes, alternating extremes, skewed low IDs, and skewed high IDs.
  Token IDs are non-negative, so the skewed/extreme cases provide the
  appropriate state-distribution stress in place of signed cancellation.

## Raw result

Final run at mirror base `0084030179bfba86bfeb6d43f7997d4076329d2c`:

```text
zeros: max_abs=8.6595537e-07, max_rel=0.003663558
tiny_finite: max_abs=9.3184644e-07, max_rel=0.0037872707
mixed_magnitudes: max_abs=9.4334246e-07, max_rel=0.0037872707
alternating_extremes: max_abs=8.6595537e-07, max_rel=0.003663558
skewed_low: max_abs=9.3271956e-07, max_rel=0.0037878531
skewed_high: max_abs=8.4433123e-07, max_rel=0.0036831424
1 passed, 5 warnings, 6 subtests passed in 29.56s
```

Wall time for the complete pytest invocation was 33 seconds, measured with UTC
`date +%s` immediately before and after the command. The process used one AMD
Instinct MI300X (gfx942), Triton attention, native RoPE, and a job-private
Triton cache under `/tmp/sglang-cache-j-3f6d35a4581a`.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID:
  `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: AMD Instinct MI300X, gfx942, unique ID `0x2e2f615a49e61496`
- Python: `/opt/venv/bin/python` (3.10.12)
- Torch: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`,
  version `2.9.1+rocm7.2.0.git7e1940d4`
- Mirror SGLang source: `/job/sglang/python/sglang`
- Installed SGLang source context:
  `/sgl-workspace/sglang/python/sglang`, commit
  `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`
- Installed native kernel package:
  `/opt/venv/lib/python3.10/site-packages/sgl_kernel`

The installed-source early baseline is recorded separately in
`/job/baseline-first.json`. It ran the existing QuickGELU ROCm kernel test and
passed 24 subtests in 37 seconds; it is not evidence for mirror-checkout changes.

## Limits

This is a reduced ModelRunner prefill test, not a full serving-engine or
end-to-end generation benchmark. No upstream issue, PR, or comment was modified.
No formatter is installed in the qualified image; `python -m compileall` and
`git diff --check` passed.
