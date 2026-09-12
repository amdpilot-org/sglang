# Investigation report

Upstream issue: https://github.com/sgl-project/sglang/issues/36894

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1047

At base `358c163250ad3b1f62939b01ce1314a0a31a0365`, parallel sampling acquires one LoRA reference for every expanded real sample. `_handle_batch_request` also creates one prefix-cache warm-up request per original prompt by copying the same LoRA-bearing request object. Completion accounting therefore releases `n + 1` references for `n` acquisitions per prompt. The counter becomes negative, while unload waits for exactly zero.

The focused regression runs the actual method body from the checked-out source with deterministic lifecycle doubles. Against the recorded base it measured 5 releases after 4 acquisitions for one prompt and 8 after 6 for two prompts. After the correction both cases balance, while the independent `n=1` boundary remains one release for one acquisition.

The correction clears `lora_path` and `lora_id` only on the tokenizer-side state object for the extra warm-up request. The separately copied scheduler request retains its LoRA ID, so prefix caching remains adapter-specific.

Related work inspected before implementation:

- https://github.com/sgl-project/sglang/pull/25747 (closed, focused prefix-cache accounting fix; absent from current main)
- https://github.com/sgl-project/sglang/pull/31808 (open, broader lifecycle accounting proposal; absent from current main)

Raw evidence is retained under `reports/j-6fcda1eec5fe/raw/`. The reported Qwen3.5 model and LoRA weights were unavailable, so no full HTTP/model reproduction is claimed. The gfx950 run only confirms that the assigned GPU was available and executed a separate deterministic numerical smoke.
