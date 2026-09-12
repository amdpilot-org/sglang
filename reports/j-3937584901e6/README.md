# Investigation report

Upstream issue: https://github.com/sgl-project/sglang/issues/36395

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3333

At the recorded base, `latency_test` already published `ServerArgs` and called
`initialize_moe_config()` before `load_model`. The independent
`correctness_test` entry point published the same configuration but did not
initialize MoE flags. A deterministic reproduction observed `auto` at the
first model-loading operation for correctness mode when `aiter` was requested;
the latency path observed `aiter`.

The fix moves MoE initialization to `load_model`, immediately before
`ModelConfig` and `ModelRunner` construction. This is the shared boundary used
by both benchmark modes and avoids duplicating initialization across callers.

Raw failing-before and passing-after outputs are retained in `raw/`. No model
weights or NVIDIA GB300/SM103 hardware were available. The assigned MI350X
gfx950 was inventoried, but no unrelated GPU smoke is claimed as validation of
the NVIDIA-only MXFP4 kernel path.
