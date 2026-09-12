# Investigation notes

Upstream issue: https://github.com/sgl-project/sglang/issues/38029

Mirror issue: https://github.com/amdpilot-org/sglang/issues/818

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

The recorded base still contained the reported gap. `ModelRunner.init_attention_backends()` called `bind_and_verify_backends()` only for the runner's initial attention backends. `EagleDraftWorker.init_attention_backend()` subsequently used `DraftBackendFactory` to construct a `DeepseekSparseAttnMultiStepBackend` (whose children are fresh `DeepseekSparseAttnBackend` instances) and a separate draft-extend `DeepseekSparseAttnBackend`. The DSA constructor did not copy `model_runner.kv_index_translator`, leaving the inherited class default `None`.

The ROCm FP8 MHA read path in `forward_mha_rocm.py` unconditionally invokes `get_attn_backend().kv_index_translator.translate_dcp_read_ids(kv_indices)` before reading the pool. The focused failing-before test reproduced `None` on both EAGLE-created shapes using the repository's real DSA mock runner and FP8 DSA token pool on gfx950. The passing test also feeds non-empty prefix ids through the static-pool translator and verifies passthrough output.

The correction is intentionally local to `DeepseekSparseAttnBackend.__init__`: every DSA backend, including those created after the generic bind pass, receives the owning runner's translator. Existing initial backends receive the same object they would receive during the later verification pass.

Raw evidence is retained under `reports/j-828b4d5abce5/evidence/`.
