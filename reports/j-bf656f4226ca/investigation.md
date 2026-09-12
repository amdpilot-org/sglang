# Qwen4 NVFP4 Marlin loader retention investigation

The prepared base still contained the issue-specific cycle identified in upstream PRs
[#38102](https://github.com/sgl-project/sglang/pull/38102) and
[#38900](https://github.com/sgl-project/sglang/pull/38900). Inside
`Qwen4ExpForConditionalGeneration.load_weights`, the nested
`load_qwen4_exp_ple_shard` helper read and wrote an attribute on itself. That
self-reference made the nested function part of a cycle, while its closure also
captured `params_dict`, the loader's fully-qualified-name snapshot of the original
Parameters.

The added regression calls the actual Qwen4 loader, disables cyclic GC, replaces a
loaded Parameter as quantization post-processing does, and watches the original by
weak reference. On the recorded base both the no-PLE and PLE cases retained the old
Parameter. With the fix both release it immediately. Separate cases verify that the
warning still occurs once per load, resets across loads, remains silent for matching
dtypes, and that loaded PLE values are unchanged.

The correction moves the warning state to a nonlocal boolean owned by the load
invocation. It does not change Marlin repacking, Parameter identity rules, tensor
layouts, checkpoint mapping, or allocator behavior, and it adds no forced GC or
cache clearing.

Hardware limitation: the assigned accelerator reports `AMD Instinct MI350X`, HIP
7.2, capability `(9, 5)` (gfx950). The selected NVFP4 Marlin tests require NVIDIA
SM80/SM86/SM90 and skipped. Therefore this job does not claim a local full-model,
multi-node, NVIDIA-memory, or Marlin numerical reproduction.
