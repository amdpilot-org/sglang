# Independent review of PR 2274

Upstream issue: https://github.com/sgl-project/sglang/issues/32572

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2311

Candidate: https://github.com/amdpilot-org/sglang/pull/2274 at exact commit `b53a21a7e9c3ca61f35a09eef878ee8bb9a194d6`.

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`.

## Recommendation

Request changes. The candidate fixes the reported Qwen3-VL deep-stack adapter case and corrects the two counterexamples inherited from PR 2188, but it does not fully support the MOSS-VL part of the original issue.

## Independent evidence

On the recorded base, `get_hidden_dim("linear_fc1"/"linear_fc2", ...)` fails for both Qwen3-VL and MOSS-VL before returning a buffer shape. The base also cannot assign Qwen3-VL `deepstack_merger_list.N` adapter weights to a layer. See `raw/base_source_contract.log`.

At the exact candidate commit, the candidate's two focused test files pass (13 tests). Independent probes confirm:

- Qwen3-VL merger dimensions are `(4608, 4608)` and `(4608, 4096)` for the reported 8B configuration.
- MOSS-VL concatenating-merger dimensions are `(16384, 16384)` and `(16384, 4096)` for a four-feature fixture.
- `deepstack_merger_list.2` maps to layer 2.
- `notdeepstack_merger_list.4` and `otherlayers.9` no longer produce false layer IDs.

The remaining counterexample exercises `LoRAManager.init_lora_modules`, not merely the dimension helper. A model-shaped fixture exposes the actual MOSS names `visual.merger.linear_fc1` and `visual.merger.linear_fc2`. Both are selected targets, but `get_layer_id` returns `None`, so the manager's ordinary-module path skips both. The exact candidate wraps zero modules and leaves every per-layer registration dictionary empty. See `raw/candidate_moss_registration_counterexample.log`.

This follows the production source directly: `MossVLVisionPatchMerger` is a single unindexed `merger`, whereas the candidate only adds indexed recognition for `deepstack_merger_list`. Therefore the new MOSS dimension override prevents the original allocation exception but does not make the MOSS visual adapter layers operational. This is a partial source fix plus insufficient regression coverage, not a full original-issue fix.

## Environment and limitations

The prepared interpreter imported SGLang from `/job/repo/python/sglang`, including the checked-out LoRA and model sources. The candidate changes no C++ or other native source, so no native rebuild was applicable. The installed runtime is PyTorch 2.11.0+rocm7.2 on one AMD Instinct MI355X (`gfx950`), rather than the issue's NVIDIA H100/CUDA environment.

An independent BF16 projection arithmetic fixture ran on the assigned GPU for the Qwen dimensions and compared against CPU FP32 references. This establishes device execution only; it does not establish full model loading, adapter registration, HTTP serving, MOSS execution, or multimodal semantic correctness. Qwen3-VL-8B/EditScore and MOSS-VL weights were not available, and the tiny Llama transport fixture cannot qualify these architectures, so it was not substituted.
