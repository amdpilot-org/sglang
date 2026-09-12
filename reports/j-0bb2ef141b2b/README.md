# Independent review of BWAP candidate 1b55c6b

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/3094 at exact commit `1b55c6bd3cf8c53858bc78a7806fea3b6db8ca17`

Upstream issue: https://github.com/sgl-project/sglang/issues/35987

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3116

## Recommendation

Request changes. The candidate fixes the two correction-generation counterexamples, but it does not fully preserve the original per-request Equation 2 then Equation 3 scoring contract when a request is processed through chunked prefill.

The prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365` has no `sglang.srt.bwap` package, reproducing the original missing-feature state. Against the candidate's immediate parent `8590564acd3e0105ed35e755c1358f647a343bd7`, the independently reproduced failures were:

- `floor((1 - 0.8) * 5)` retained zero neurons because the binary product was `0.9999999999999998`.
- pooling the packed rows of two requests before Equation 2 selected neuron 0, while per-request Equation 2 followed by Equation 3 selected neuron 1.

At exact candidate commit `1b55c6b`, the focused suite passed 23 tests plus 3 subtests. On the assigned AMD Instinct MI350X, the corrected `D_FF=5`, sparsity `0.8` FP16 gathered path retained one neuron and matched an independent dense masked PyTorch reference with maximum absolute error 0.

## Remaining counterexample

For one request with four rows `[0.8, 0, 0.6]` followed by one row `[0, 1, 0]`, Equation 2 over the complete five-token prompt produces `[0.71554, 0.44721, 0.53666]` and selects neuron 0. A one-shot candidate forward agrees. If the same request is delivered as chunk lengths 4 then 1, the candidate independently computes Equation 2 for each chunk and applies its running max, producing `[0.8, 1.0, 0.6]` and selecting neuron 1. Chunk boundaries therefore change the mask for an otherwise identical request.

This is caused by treating every extend forward as a complete per-request sample. Correct chunked behavior needs sufficient statistics across chunks (or an explicit and documented exclusion of chunked prefill); merely forwarding `extend_seq_lens` only separates requests within the current packed forward.

## Environment and limitations

- Python imports resolved to `/job/repo/python/sglang/...`, not an installed SGLang wheel.
- Torch was `2.11.0+rocm7.2`, HIP `7.2.26015`, with one AMD Instinct MI350X visible.
- No native C/C++/CUDA/HIP/FlyDSL source differs in the candidate, so no native rebuild was applicable.
- No production model weights were available. Production semantic accuracy, throughput, actual serving with chunked prefill, NVIDIA, quantized, biased, TP>1, speculative decode, and unsupported model architectures remain unverified or explicitly unsupported.
- The deterministic tiny-Llama transport fixture was not used as semantic proof because it cannot qualify BWAP scoring accuracy or a different architecture.

Raw commands and outputs are retained in `evidence/`.
