# Independent review of PR 2914

Candidate: https://github.com/amdpilot-org/sglang/pull/2914 at `1baf5337c192902f8f19b56e05d713945b5ab4ff`

Upstream issue: https://github.com/sgl-project/sglang/issues/19092

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2870

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2948

## Verdict

Recommendation: **request changes**. The patch is a partial initial-load improvement, not a full resolution of the original feature request.

The recorded base was exactly the prepared checkout commit. An independent probe reproduced sequential component loading on the base. At the exact candidate commit, the candidate's focused suites and GPU mock passed, and a separate top-level probe confirmed that the new executor can overlap mocked component work.

The principal counterexample is in the real native-loader boundary. The new `_load_native_with_context` acquires `model_construction_lock` around the entire `load_native` call. `load_native` calls the transformers or diffusers `from_pretrained` method, so checkpoint I/O and weight loading are also inside the lock rather than only unsafe construction. An independent two-thread probe observed `max_active_native_loads=1` and approximately 0.401 seconds for two 0.20-second operations. Thus two native components remain sequential.

The patch also does not alter or validate refit/wake-up, production pinned-memory loading, or multi-rank loading. The candidate itself accurately discloses most of these limitations, but those disclosures mean it cannot be accepted as fully resolving the original issue.

## Environment and source paths

- Prepared base and required comparison: `358c163250ad3b1f62939b01ce1314a0a31a0365` (no difference).
- Interpreter: `/tmp/amdpilot-repo-j-06f052d570f8/venv/bin/python`, Python 3.12.3.
- Imported SGLang: `/job/repo/python/sglang/__init__.py`.
- Imported pipeline implementation: `/job/repo/python/sglang/multimodal_gen/runtime/pipelines_core/composed_pipeline_base.py`.
- Torch 2.11.0+rocm7.2, HIP 7.2.26015.
- One AMD Instinct MI355X, `gfx950:sramecc+:xnack-`.
- NUMA balancing was enabled and deliberately left unchanged.
- No native source changed, so no native rebuild applied.

## Evidence interpretation

The candidate GPU test creates small `torch.nn.Linear` objects in a mocked loader and checks their outputs against NumPy. It is useful proof that the assigned GPU ran and concurrent mock transfers produced correct arithmetic. It is not proof of real Qwen-Image loading, Diffusers/Transformers thread safety, model semantic accuracy, peak-memory safety, or server cold-start improvement.

Raw commands, outputs, and independent probes are retained under `reports/j-06f052d570f8/raw/`.
