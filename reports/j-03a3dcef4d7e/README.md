# Investigation result

Upstream issue: https://github.com/sgl-project/sglang/issues/29998

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2396

The prepared source already contains a credible fix for the reported Run:AI
staging-buffer corruption. No production source was changed.

The report predates upstream PR
[#32896](https://github.com/sgl-project/sglang/pull/32896), merged on 2026-07-31.
That change marks tensors yielded by `runai_safetensors_weights_iterator` and
makes `should_async_load` consume marked tensors synchronously, before the
streamer can refill their zero-copy backing buffer. The current registered
regression deterministically demonstrates both sides of the race: its ordinary
CPU-tensor control reads the overwritten value `2`, while the marked Run:AI
tensor is consumed inline and preserves `1`.

The GLM-5.2 architecture (`GlmMoeDsaForCausalLM`) uses
`DeepseekV2WeightLoaderMixin`. Its delayed fused-MLA and DSA-indexer paths also
clone marked tensors before retaining them across iterator yields. The included
gfx950 check exercises the fused block-FP8 indexer pair with simulated staging
buffer reuse. In both weight-first and scale-first order, marked tensors load
with zero maximum absolute error against a separately computed reference;
unmarked controls reproduce corruption with maximum absolute error 1.0.

## Validation

- `/tmp/amdpilot-repo-j-03a3dcef4d7e/venv/bin/python -m pytest -q test/registered/unit/model_loader/test_runai_model_streamer_loader.py`
  passed all 11 tests. This includes the deterministic asynchronous buffer-reuse
  regression and independent marked/unmarked behavior.
- `HIP_VISIBLE_DEVICES=0 /tmp/amdpilot-repo-j-03a3dcef4d7e/venv/bin/python reports/j-03a3dcef4d7e/gpu_runai_indexer_ownership.py`
  passed on one AMD Instinct MI355X (`gfx950:sramecc+:xnack-`). Both tensor
  arrival orders produced zero error for marked inputs and reproduced corruption
  in the unmarked controls.
- Raw GPU output is retained at
  `/tmp/amdpilot-repo-j-03a3dcef4d7e/evidence/gpu_runai_indexer_ownership.log`.
  Issue and PR metadata are retained in the same private runtime tree.

## Limitations

The reported full serving configuration was not reproduced: this worker has one
AMD gfx950 GPU, not eight NVIDIA B300 GPUs, and the `zai-org/GLM-5.2` weights
were not provided. Consequently this is a verified fix candidate for the exact
buffer-ownership mechanism and relevant GLM loader paths, not a claim of TP8
generation correctness, tool-call semantic accuracy, CUDA/TRT-LLM behavior, or
a full-model reproduction. The tiny Llama serving fixture was not used because
it cannot qualify GLM-5.2 architecture-specific weight loading.

