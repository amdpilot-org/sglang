# Independent review of PR 1445

Reviewed exact candidate commit `aa3d942320f7ea135b518733eb9a8381c002064e` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **accept**. The candidate fully resolves the original host-side config-key mismatch.

On the base, the actual tuner helper generated `E=256,N=128,...,dtype=int4_w4a16.json`, while the runtime helper generated `E=256,N=256,...,dtype=int4_w4a16.json`. The same factor-of-two mismatch appeared for independent minimal and large int4 shapes, while unquantized, fp8, int8_w8a8, and int8_w8a16 controls matched.

At the exact candidate commit, the candidate's six-test regression passed and an independent seven-case probe passed. Inspection also confirmed that the candidate changes the production benchmark-mode lookup, not only the output filename: both now use `get_config_n(shard_intermediate_size)`, which returns `shard_intermediate_size // 2`, matching the runtime's `w2.shape[2]` key.

The loaded tuner and runtime modules came from `/job/repo/benchmark/kernels/fused_moe_triton/common_utils.py` and `/job/repo/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py`. The prepared interpreter was `/tmp/amdpilot-repo-j-5ff0a8225bd6/venv/bin/python`. No native source changed, so no native rebuild was applicable.

The assigned device was one AMD Instinct MI355X (`gfx950`). No GPU kernel was run: the reviewed defect is deterministic host-side filename/lookup construction. The reported model weights were unavailable, so this review does not claim a full model tune-and-serve reproduction, model correctness, performance, HTTP transport, or distributed validation.

Raw outputs and the reviewed diff are retained in `reports/j-5ff0a8225bd6/raw/`.
