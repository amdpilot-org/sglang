# GLM-5 mHC pipeline proxy investigation

Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

The checked-out runner allocates only `hidden_states` for mHC pipeline proxies in both `runner/base_runner.py` and `runner_utils/buffers.py`. Before this change, `Glm5NextModel.forward` nevertheless indexed `pp_proxy_tensors["residual"]` on every non-first stage and emitted a separate residual on every non-last stage. The issue-specific regression in `regression-before.txt` reproduced the resulting `KeyError: 'residual'` at `glm5_next.py:984`.

Upstream PR https://github.com/sgl-project/sglang/pull/37375 proposed the same model-side contract correction but is closed and its commit is not present in this checkout. DeepSeek-V4, the other mHC model, also transfers only flattened `hidden_states` across pipeline stages. These two observations support correcting GLM-5 rather than reintroducing a redundant runner residual allocation.

The change keeps the non-mHC path unchanged and adds a separate regression for it. Full GLM-5.3-Flash serving was not possible because the model weights were unavailable and only one GPU was assigned. The GPU fixture therefore validates only the exact proxy handoff on gfx950; it is not a full-model, NVIDIA SM120, or multi-stage reproduction.
