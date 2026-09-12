# Investigation result

The prepared `main` source already contains the release-to-main correction for the
reported routing failure. Upstream commit
`60ff1e33a51f363b8426d85e3434a1eb48e1b70a` added automatic
`flashinfer_mxfp4` selection when DeepSeek-V4 header probing identifies MXFP4
routed experts on supported NVIDIA SM90, SM100, or SM120 hardware.

The v0.5.18 release commit (`71de97b264b04dcd514cf904003028aefe9775c8`)
predates that commit. The reporter's working snapshot
(`a1fe4e30a983b04bbb74099dfc71bc7148c5c577`) contains it, as does this job's
base (`358c163250ad3b1f62939b01ce1314a0a31a0365`). This identifies the relevant
upstream change without duplicating it.

The existing callable-level regression already expressed the desired result, but
its synthetic NVIDIA case inherited `is_hip=True` from the physical gfx950 host.
The test now explicitly models a non-HIP platform and checks all three supported
NVIDIA capability boundaries. Its existing negative cases cover an explicit user
backend, A2A dispatch, FP4 dequantization, ordinary FP8 experts, HIP/NPU,
unsupported NVIDIA architectures, and the separate NVFP4 route.

The assigned GPU is an AMD Instinct MI350X (`gfx950`). A numerical GPU check was
run successfully, but the SM90 FlashInfer test correctly skipped. No H800 kernel,
full checkpoint, TP=4, DSPARK, HTTP serving, or model-quality result is claimed.
Raw logs are retained under
`/tmp/amdpilot-repo-j-1ebd62d7dbd9/evidence/`.

Upstream issue: https://github.com/sgl-project/sglang/issues/37342

Mirror issue: https://github.com/amdpilot-org/sglang/issues/981
