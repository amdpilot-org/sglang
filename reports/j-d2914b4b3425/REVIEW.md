# Independent review of PR 2819

Upstream issue: https://github.com/sgl-project/sglang/issues/31924

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2754

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2847

Candidate: https://github.com/amdpilot-org/sglang/pull/2819 at `f112ca15d33b84559c5e7ca3372cf38d8a771b5c`

## Recommendation

Reject. The candidate adds tests and reports only; it changes no runtime source. Its regression proves fused Q/K/V shard routing for a ModelOpt-style fused parameter, but the original issue explicitly rules that hypothesis out for the reported BF16 FLUX.2 checkpoint. It therefore constitutes test-only hardening for a neighboring configuration, not a fix for the original issue.

The prepared base still contains the ROCm skip explaining that issue 31924 remains unfixed. The candidate does not remove the skip.

## Evidence

The prepared checkout was clean and exactly at the required base, `358c163250ad3b1f62939b01ce1314a0a31a0365`. The PR ref resolved exactly to the requested candidate commit. I preserved outputs under `raw/` before switching revisions and returned to `amdpilot/j-d2914b4b3425` for this report.

The candidate's two new tests pass. However, they construct only `to_qkv` and `to_added_qkv` fused targets and install a custom three-argument sharded loader. In the BF16 configuration described by the issue, Q/K/V parameters are separate; exact parameter names take the identity short-circuit and never exercise the candidate's fused mapping.

An independent contract case used the real `Flux2Transformer2DModel.param_names_mapping` and the real `_load_weights_into_module` path. `img_in.weight` mapped to `x_embedder.weight`, but a differently laid-out `(2, 3)` disk tensor was passed unchanged to a `(3, 2)` parameter's ordinary two-argument loader. Both the recorded base and exact candidate raised the same bare `AssertionError` with an empty message. The candidate therefore supplies no reshape/layout reconciliation.

The adversarial case also ran with both tensors resident on the assigned single AMD GPU. PyTorch reported `AMD Instinct MI350X`, ISA `gfx950:sramecc+:xnack-`; it reproduced the empty assertion. This is a focused GPU copy-path check, not a full model or HTTP-serving reproduction.

Current upstream issue discussion also reports that a real FLUX.2-klein-base-4B `transformer/` checkpoint used identity runtime names and skipped zero transformer tensors, while text-encoder and VAE state was silently dropped. The candidate neither covers those real-checkpoint observations nor the issue's secondary text-encoder contract.

## Source and native paths

The imported updater and FLUX.2 model came from `/job/repo/python/sglang/...`, using `/tmp/amdpilot-repo-j-d2914b4b3425/venv/bin/python` with Torch `2.11.0+rocm7.2`. The candidate changes no C++, HIP, FlyDSL, or other native source, so no native rebuild was applicable. Aiter loaded its prepared native module from `/tmp/amdpilot-repo-j-d2914b4b3425/cache/aiter/module_aiter_core.so`; no claim depends on rebuilding it.

## Limitations

The Qwen-Image and FLUX.2-klein-base-4B weights were not available locally, so I could not boot either full model or replay the HTTP 400 end to end. The fixture reproduces the exact mapping/layout/ordinary-loader failure contract and establishes that the candidate does not change it. It does not identify the precise real-checkpoint tensor responsible for the historical AMD CI failure. Multi-GPU, tensor-parallel, offload, FSDP, and multi-node execution were not exercised.
