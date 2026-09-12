# NVFP4 speculative dequant workspace investigation

The recorded base still calls `get_flashinfer_dequant_workspace_kv_buffer` from
`FlashInferAttnBackend.forward_extend` with
`forward_batch.extend_prefix_lens_cpu` and
`forward_batch.extend_seq_lens_cpu`. Speculative GPU-only batches can leave
those fields unset, while `_prepare_dequant_extend_workspace` indexes them.

The failing-before regression records the resulting unhandled `NoneType`
failures. The patch adds an explicit validation at the dequant-workspace
boundary and preserves empty metadata as a valid no-request boundary case.

This is deliberately not reported as a complete fix. The original issue needs
a graph-safe NVIDIA NVFP4 path. Two existing upstream candidates were inspected:

- https://github.com/sgl-project/sglang/pull/36038
- https://github.com/sgl-project/sglang/pull/36045

The assigned device is an AMD Instinct MI350X (`gfx950`) under ROCm 7.2, so the
SM120 FlashInfer/TRTLLM execution and the reported Qwen3.8-27B-NVFP4 model could
not be reproduced here.
