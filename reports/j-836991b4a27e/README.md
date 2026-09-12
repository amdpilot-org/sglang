# VILA1.5-3B-AWQ serving investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/2345

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3018

## Result

The exact checkpoint requested by the issue, `Efficient-Large-Model/VILA1.5-3b-AWQ`, cannot currently be served by SGLang. This is an unsupported checkpoint architecture and artifact format, not a server-transport defect.

The failure was reproduced against base commit `358c163250ad3b1f62939b01ce1314a0a31a0365`. `ModelConfig` stops at `AutoConfig`: the checkpoint declares the custom `llava_llama` model type and `LlavaLlamaModel` architecture, neither of which is registered by Transformers or SGLang.

Adding only a config alias would not implement the requested feature. The checkpoint has three independently missing pieces:

1. The language model is stored only as `llm/vila-1.5-3b-w4-g128-awq-v2.pt`, a legacy TinyChat AWQ state dictionary. Its linear layers contain `int16` `qweight`, `scales`, and `scaled_zeros`. SGLang's AWQ implementation expects the standard AWQ metadata and packed tensors (`qweight`, `qzeros`, and `scales`) used by its kernels. The checkpoint has no `quantization_config` describing this legacy layout.
2. The repository contains a `llm/model.safetensors.index.json`, but both referenced shards (`model-00001-of-00002.safetensors` and `model-00002-of-00002.safetensors`) are absent from the Hugging Face repository and return HTTP 404. It therefore cannot fall back to the nominal unquantized weights.
3. The checkpoint is a split VILA repository (`llm/`, `vision_tower/`, and `mm_projector/`) without a top-level Hugging Face processor. Supporting it requires a VILA-specific processor/conversation implementation and nested component loader in addition to a model class. Reusing NVILA is incorrect: NVILA uses a Qwen2 language model, a different exported config, and a different processor contract.

The assigned accelerator was an AMD Instinct MI355X (`gfx950`). The checkpoint card lists Ampere, Jetson, Hopper, and Lovelace, and its documented quantized inference path is TinyChat. No compatible AMD TinyChat kernel or standard SGLang AWQ export was available, so model execution and numerical comparison were not possible. No GPU reset, package replacement, or node-wide change was performed.

## What would be required for full support

- Define and register a VILA 1.5 config and `LlavaLlamaModel` implementation, including the SigLIP vision tower, `mlp_downsample` projector, and Llama language model.
- Add a VILA-specific multimodal processor implementing the checkpoint's `<image>` placeholder, 384x384 SigLIP preprocessing, conversation template, multi-image ordering, and video-frame behavior.
- Add a loader for the split component directories.
- Either implement and validate the legacy TinyChat `int16`/`scaled_zeros` AWQ layout on supported SGLang GPU kernels, or obtain a standard AWQ/GPTQ export with complete quantization metadata. A config-only rename is insufficient.
- Validate end-to-end image requests against the original VILA implementation with the actual model weights. The deterministic tiny Llama fixture is not applicable because it cannot qualify this architecture, quantization format, or multimodal semantics.

## Evidence

Raw command output is retained in this directory:

- `model_config_failure.log`: failing-before reproduction from the checked-out SGLang implementation.
- `checkpoint_contract.log`: model repository revision, missing safetensor shards, and inspected legacy AWQ tensor names/shapes/dtypes.
- `gpu_inventory.log`: assigned GPU identity and architecture.

No production change is presented as a fix. The remaining work is the complete model, processor, loader, and quantized-kernel integration described above.
