# Investigation report: Qwen3.8 thinking + tools token-ID-0 loop

Upstream issue: https://github.com/sgl-project/sglang/issues/36537

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1150

## Outcome

The prepared `main` checkout already contains the source correction associated
with this report, so no runtime source change was made.

Current issue follow-up and upstream PR #36845 identify the reported token-ID-0
output as silent long-context QSA decode corruption on NVIDIA SM121/GB10, rather
than a defect in `qwen3_coder` parsing. The merged implementation keeps exact
SM121 off FlashInfer's TRT-LLM sparse-decode route and dispatches it to the
shape-qualified Qwen3.8 SM121 kernel. Exact SM120 and supported SM100 devices
retain the TRT-LLM path. The implementation also rejects calls outside the
qualified SM121 tensor, dtype, topology, batch, and selected-KV contract.

Evidence in the prepared checkout:

- `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py` documents
  the SM121 corruption and limits `_resolve_trtllm_sparse_decode()` to SM100 and
  exact SM120. `_resolve_flash_attn_varlen_func()` selects
  `qwen38_qsa_sm121_varlen` for exact SM121.
- `python/sglang/kernels/kda_kernels/qwen38_qsa_sm121/` contains the specialized
  kernel and its explicit supported-call contract.
- `test/registered/kernel/qsa/test_qsa.py` covers exact SM120/SM121 capability
  detection, positive SM100/SM120 TRT-LLM routing, negative other-SM12x routing,
  SM121 specialized-kernel resolution, and an SM121 numerical sparse-reference
  comparison.

## Local validation

The focused boundary suite passed with 10 tests and skipped the one hardware-
specific SM121 numerical test:

```text
10 passed, 1 skipped, 28 deselected
SKIPPED: SM121-only kernel
```

The independent QSA FP8 cached-prefix numerical test executed on the assigned
AMD Instinct MI355X (`gfx950`) and passed. This confirms real local GPU
execution but does not validate the NVIDIA-only correction.

Raw commands, outputs, exit codes, implementation excerpts, and GPU inventory
are retained in `reports/j-a168bdd9c5ea/raw/`.

## Limitations

The source report used NVIDIA SM120/SM121 and Qwen3.8-Flash-Next-NVFP4. This
environment provides one AMD MI355X gfx950 and does not provide the reported
model weights, so the original HTTP/model reproduction and the SM121 kernel's
numerical execution could not be repeated locally. The deterministic tiny
Llama fixture was not used because it cannot exercise Qwen3.8's architecture,
QSA kernels, NVFP4 checkpoint, or NVIDIA dispatch and therefore cannot qualify
this issue. No full-model, NVIDIA, multi-GPU, or semantic-accuracy claim is
made.
