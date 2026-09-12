# Independent review of amdpilot-org/sglang PR 1489

Candidate commit: `0b2941a997932e6d40e960861846f66f929aa6b4`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Upstream issue: https://github.com/sgl-project/sglang/issues/34947

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1528

## Recommendation

Accept. The candidate fully resolves the original issue's stated correctness contract: a TRT-LLM sparse-MLA launch with more than 65,535 flattened query rows must either execute attention or fail loudly. The candidate makes the affected shared `_forward_trtllm` path raise before the native launch, so it prevents the reported normal return with an unwritten output.

This is a fail-loud fix, not a capability enhancement. A single unchunked extend above 65,535 tokens remains unsupported and now raises `ValueError`; chunking or `cum_seq_lens_q` adoption is still required to compute such an extend.

## Evidence

The prepared checkout was exactly the recorded base before review. The candidate's regression is absent there and fails to start because the new validation module does not exist. Source inspection on the base confirms that `page_table_1.shape[0]` becomes the TRT-LLM batch dimension and reaches `trtllm_batch_decode_with_kv_cache_mla` without a limit check.

At the exact candidate commit, its seven boundary cases passed. An independent adversarial check accepted 0, 1, 65,534, and 65,535, and rejected 65,536, 65,537, 131,072, and 2,147,483,647 with `ValueError` and chunked-prefill guidance. An AST/source-order check confirmed that the validation call follows derivation of `batch_size` and dominates both counter-buffer growth and the FlashInfer launch in the shared `_forward_trtllm` method. Thus it covers the non-DP direct runner prefill path described by the issue rather than only server configuration.

The patch changes Python only. No native source or build definition changed, so no native rebuild was applicable. Imports resolved to the candidate helper under `/job/repo/python`; the prepared interpreter's Torch resolved from `/opt/venv` and is ROCm 7.2.

## Architecture limitations

The assigned device is an AMD Instinct MI350X (`gfx950`), not NVIDIA B200/SM100. `flashinfer` is not installed in the prepared interpreter, and the affected CUDA `trtllm-gen` kernel cannot run on this architecture. Therefore the original sentinel-output corruption, CUDA `gridDim.z` launch error, profiler kernel count, full DeepSeek model, and multi-rank serving behavior could not be independently reproduced. A real gfx950 identity-matmul check completed with maximum absolute error 0.0, but it is only evidence that the assigned GPU was usable and is not evidence for the NVIDIA failure.

Raw commands, exit codes, import paths, and output are retained under `raw/`.
