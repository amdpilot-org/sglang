# Independent review of PR 1049

Recommendation: accept. The exact candidate commit `0fef25c27168988b4134d86d0e360b20e11937e0` fully resolves the deterministic tensor-ownership contract described by the original issue.

The prepared checkout was clean and exactly at the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`; there was no image/base difference. Running the candidate's regression file against that base produced four issue-specific failures and one passing boundary case. The failures directly showed overwritten values reaching the FP8 dequantizer or gated-pair quantizer.

At the exact candidate, the loader source imported from `/job/repo/python/sglang/srt/layers/quantization/nvfp4_online.py`. The patch adds ownership at all three relevant retention sites: pending FP8 weights, pending FP8 scales, and pending gated w1/w3 tensors. Immediate-consumption paths remain unconditionally un-cloned, and unmarked tensors retain their prior reference semantics.

The candidate's five regressions passed. Independent cases also passed for two expert keys concurrently retaining views from one repeatedly reused FP8 buffer, reverse gated arrival (`w3` before `w1`), an explicitly false marker, and a real FP8 reusable device buffer on the assigned AMD Instinct MI350X/gfx950. Six neighboring ModelOpt NVFP4 tests passed.

No native files changed, and `repository-environment.json` supplies no native build target, so native rebuilding was not applicable. Python came from `/tmp/amdpilot-repo-j-71cfb076a89c/venv/bin/python`; the loaded SGLang module was the candidate checkout, while Torch was the prepared ROCm 7.2 installation at `/opt/venv/lib/python3.12/site-packages/torch`.

This review does not claim a full checkpoint, semantic model-output, distributed RunAI, Kubernetes, tensor-parallel-size 8, multi-node, or NVIDIA NVFP4 backend reproduction. Those environments and weights were unavailable. The deterministic contract in the issue is nevertheless fully exercised, including on real gfx950 device storage; no remaining counterexample was found.
