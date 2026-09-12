# Investigation report: sglang issue 32156

Upstream issue: https://github.com/sgl-project/sglang/issues/32156

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2128

## Result

The prepared `main` checkout already contains the source fix for the reported
failure. Upstream PR https://github.com/sgl-project/sglang/pull/32736 merged on
2026-07-29 as commit `f69af7b7ad62f055867732196c13f0dd742097d2`.

The published `unsloth/Qwen3.6-35B-A3B-NVFP4` config is mixed precision: its
top-level format is `mixed-precision`, while its expert group declares
`nvfp4-pack-quantized` and 4-bit `tensor_group` input activations. Before PR
#32736, SGLang consulted only the top-level format and discarded those input
activations. `get_moe_scheme()` consequently reached
`_is_dynamic_token_w8a8(weight_quant, None)` and produced the issue's exact
`AttributeError`.

At the recorded base, parsing the real checkpoint config preserves activation
metadata for gate, up, and down expert projections and selects
`CompressedTensorsW4A4Nvfp4MoE`. The already-landed regression also covers the
independent contributing ignore-list boundary bug, exact/regex matching,
suffix matching, per-group activation retention, and a weight-only group.

## Commands and evidence

The checkpoint config was downloaded (configuration only, no weights) to the
private runtime directory:

```bash
curl -L --fail https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4/raw/main/config.json \
  -o /tmp/amdpilot-repo-j-5c779a429c52/qwen-config.json
```

Issue-specific current-versus-historical config selection:

```bash
PYTHONPATH=/job/repo/python \
  /tmp/amdpilot-repo-j-5c779a429c52/venv/bin/python \
  reports/j-5c779a429c52/reproduce_config_selection.py \
  /tmp/amdpilot-repo-j-5c779a429c52/qwen-config.json
```

Output is retained in `config_selection.log`. It records the historical exact
`NoneType.num_bits` failure and current selection of the NVFP4 MoE scheme.

Existing regression suite:

```bash
/tmp/amdpilot-repo-j-5c779a429c52/venv/bin/python -m pytest -q \
  test/registered/unit/layers/quantization/test_compressed_tensors_mixed_precision.py
```

Result: 5 passed.

Assigned-GPU identity and a basic independent numerical execution check are in
`gpu_check.log`: AMD Instinct MI350X, `gfx950:sramecc+:xnack-`, maximum absolute
error 0.0 for a simple elementwise reference comparison. This only establishes
that the assigned GPU was used; it does not validate NVFP4.

## Limitations

The reported checkpoint weights were not downloaded and the full server was
not launched. The assigned AMD gfx950 GPU cannot instantiate the NVIDIA
Blackwell-only `CompressedTensorsW4A4Nvfp4MoE` implementation; doing so raises
`ValueError: Current platform does not support NVFP4 quantization. Please use
Blackwell and above.` Therefore model loading, NEXTN speculation, CUDA kernels,
and output correctness remain unverified. No native code changed or required a
rebuild.
