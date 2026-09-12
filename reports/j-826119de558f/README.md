# GLM-5.2 MXFP4 feature intake and verification

Upstream issue: https://github.com/sgl-project/sglang/issues/36447

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2686

## Outcome

`environment_blocked`

The requested implementation is no longer absent from the prepared base. Since
the issue was filed, support landed incrementally in the model, Quark MXFP4,
DSA, context-parallel, MTP, and serving paths. The current checkout maps
`GlmMoeDsaForCausalLM`, recognizes Quark MXFP4 checkpoints, exposes the
`flashinfer_mxfp4` MoE runner separately from the `megamoe` A2A backend, and
contains GLM-5.2 deployment recipes and regression coverage.

The complete acceptance workload could not be reproduced on this job. The
requested deployment needs multiple GPUs (the original proposal says TP/EP=8;
the current AMD recipe uses TP=4), while the assigned node exposes one MI355X.
The 744B-class `amd/GLM-5.2-MXFP4` weights were not staged and were deliberately
not downloaded merely for a startup attempt that could not fit or qualify the
distributed configuration. No model-quality, 1M-context, context-parallel,
Mega-MoE collective, or EAGLE/MTP end-to-end claim is made here.

## Source and checkpoint inspection

The public checkpoint configuration was downloaded without weights into the
private runtime directory:

`/tmp/amdpilot-repo-j-826119de558f/hf/hub/models--amd--GLM-5.2-MXFP4/snapshots/386bd0e4ec821f7b07975701cec3c3b953a5576a/config.json`

It reports:

- architecture `GlmMoeDsaForCausalLM`;
- model type `glm_moe_dsa`;
- maximum position count 1,048,576;
- DSA `index_topk=2048`;
- 256 routed experts, 8 selected per token;
- Quark global FP4 E2M1-style groups of 32 with E8M0 scales, with dense,
  attention, router, and draft-model exclusions represented by 1,319 entries.

Relevant implementation surfaces in the prepared base include:

- `python/sglang/srt/models/glm4_moe.py` — GLM DSA and NextN mappings;
- `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` — GLM
  DSA loading cases;
- `python/sglang/srt/layers/quantization/quark/` — serialized/online MXFP4
  expert layouts and E8M0 scales;
- `python/sglang/srt/layers/moe/flashinfer_megamoe.py` — Mega-MoE adapter;
- `python/sglang/srt/layers/attention/dsa/` and `dsa_backend.py` — paged DSA,
  metadata propagation, top-k, and memory-bounded logits;
- `docs/cookbook/autoregressive/GLM/GLM-5.2.mdx` and
  `docs/src/snippets/configs/zai-org/glm-5.2.jsx` — current deployment recipes.

Related-work evidence is retained under `raw/`, including the source issue,
checkpoint probe, and selected upstream PR metadata. The open source issue is
older than multiple merged GLM-5.2 changes visible in current `main`, so copying
the original prototype wholesale would duplicate and likely regress newer
backend selection and Quark handling.

## Verification performed

All tests used `/tmp/amdpilot-repo-j-826119de558f/venv/bin/python`, Torch
2.11.0+rocm7.2, HIP 7.2, and the assigned gfx950 MI355X. Compilation caches
were kept below `/tmp/amdpilot-repo-j-826119de558f`.

1. CPU contract suite: 16 passed. This covered FP8-to-MXFP4 shape/type
   conversion, MXFP8 activation preparation and handoff behavior, the ROCm
   MQA-logits 2 GiB chunk boundary, and Mega-MoE adapter routing/workspace
   semantics.
2. GPU fast-top-k suite: 23 passed. The native JIT kernel at `topk=2048` was
   compared against independent PyTorch selection for long and short rows,
   ragged row starts, non-contiguous row strides, duplicate-heavy/constant
   scores, zero/negative scores, and unsupported-k rejection.
3. GPU DSA MQA/top-k integration: 1 test plus 3 subtests passed. Real gfx950
   AITER `fp8_mqa_logits` and the SGL kernel top-k produced identical valid
   selections for decode- and prefill-shaped inputs when the redundant
   `-inf` initialization was disabled; the test also proved dirty invalid
   tails existed, avoiding a vacuous comparison.

Raw stdout, exact commands, exit codes, GPU identity, source paths, and related
PR metadata are retained in `reports/j-826119de558f/raw/`.

## Remaining limitations

- No GLM-5.2 weights were loaded and no HTTP server was started.
- No semantic accuracy evaluation was performed.
- TP/EP=8, the current TP=4 AMD recipe, Mega-MoE collectives, context
  parallelism, and distributed symmetric buffers were not executable with one
  visible GPU.
- A 1M-token workload and its memory behavior were not exercised.
- EAGLE/MTP end-to-end generation was not exercised.
- CUDA-only DeepGEMM/Mega-MoE paths from the original NVIDIA-oriented launch
  proposal cannot be qualified on this ROCm node. The ROCm Quark/AITER path is
  a related implementation, not proof of CUDA parity.
- No native C++ source was changed, so no native rebuild was applicable.
- The original issue remains open and mentions a private prototype; this
  report does not assert that every optimization from that unpublished tree is
  byte-for-byte present in current `main`.
