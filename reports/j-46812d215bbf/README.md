# Investigation result

The prepared base already contains the narrow source correction described in the issue. In `DSparkDraftMixin.compute_base_logits`, the target head is projected through `project_through_lm_head`; that helper calls `quant_method.apply(lm_head, hidden, None)` when `should_apply_lm_head_quant_method` admits the runtime head and otherwise uses the existing dense matrix multiplication.

The behavior entered the history in commit `8a1e6e4e461044246739b5a1ad579c8acc556a2d` (`Qwen3.8-27B Model Support (#34859)`). Its parent directly used `torch.matmul(hidden, weight.T)`, matching the reported failure mechanism.

This PR adds direct DSpark regression coverage. The packed test preserves the reported block-size factor (`35 * 7` rows) at reduced hidden/vocabulary dimensions and proves that the legacy direct multiplication fails before the current quant-method dispatch succeeds. Separate tests cover a normal dense head and a stale ModelOpt method attached to dense storage.

Raw test and GPU output is retained under `raw/`. The GPU run used the assigned AMD Instinct MI355X (`gfx950`) and compared the dispatched result against an independent dense PyTorch reference with zero maximum absolute error.

The original NVIDIA GB10 hardware, CUDA/FlashInfer/ModelOpt runtime, and the two 27B checkpoints were not available. Therefore this is a verified source-level candidate with deterministic CPU and gfx950 execution evidence, not a full reproduction of the reported CUDA graph or model-serving workload.
