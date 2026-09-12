# NGRAM + torch_native startup validation

Upstream issue: https://github.com/sgl-project/sglang/issues/36352

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1178

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

The prepared source still accepted speculative decoding whenever the resolved
attention backend was `torch_native`. A regression against the actual
`handle_attention_backend_compatibility` implementation failed before the
source change because no `ValueError` was raised for the combined backend or
either split backend. The raw failure is in `evidence/regression-before.txt`.

The correction rejects speculative decoding when either resolved target-model
attention backend is `torch_native`, before CUDA-graph adjustment or model
loading. The error tells the operator to select another backend or disable
speculative decoding. It deliberately does not invent missing target-verify
length metadata or claim that Torch Native supports speculative decoding.

The regression covers the reported NGRAM + combined `torch_native` case and
the independent split prefill/decode boundaries. Controls prove that plain
Torch Native remains accepted and still disables both CUDA-graph phases, while
NGRAM with the supported Triton backend remains accepted.

`evidence/regression-after-final.txt` records 4 passing tests plus 2 passing
subtests. The full server-args unit file recorded 228 passes and two unrelated,
pre-existing ROCm context-parallel failures because this gfx950 environment
correctly rejects deprecated HIP prefill CP; see `evidence/test-server-args.txt`.

No model weights were downloaded and no server was launched. The referenced
tiny-Llama fixture scripts at commit
`f1d603677ca76a9ea21124a544e405c5b0cbd315` were inspected, but a transport or
engine run is not needed to validate an argument-policy rejection and would
not add semantic evidence. Therefore `gpu_execution` is false. The assigned
MI355X/gfx950 inventory is retained in `evidence/gpu-inventory.txt`; it was not
used for inference. This does not claim a full-model, NVIDIA, or multi-node
warmup reproduction.

