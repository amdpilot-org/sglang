# Correction generation 2

Reviewed candidate PR https://github.com/amdpilot-org/sglang/pull/718 at exact commit `f8853833a75e992e618a9e5d74efe493c3bfefe2` and independent review PR https://github.com/amdpilot-org/sglang/pull/779.

The candidate's DP-controller change is retained: probes use a separate round-robin counter and do not refresh or consume user load-balancing state. Its scheduler interception is corrected. A fully idle `/health_generate` request must enter generation admission so a successful response is backed by a model result; only a busy scheduler may synthesize the health response after another completed forward.

The focused regression is deliberately limited to that boundary. It does not treat MagicMock side effects as evidence that real distributed PrefillDelayer state is synchronized.

The source issue's TP8/DP8 DeepSeek-V4-Pro performance claim remains unverified because this job has one AMD MI355X rather than 8 NVIDIA B30Z GPUs and lacks the model weights. No architecture-specific source claim is made from that limitation.
