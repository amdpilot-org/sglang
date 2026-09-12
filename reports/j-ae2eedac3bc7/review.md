# Independent review of amdpilot-org/sglang PR 1702

Candidate reviewed: `95c8c67ac2ad6b47b0680c5f76fa8f3c89a1752a`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Recommendation: **request changes**. The candidate fixes the exact fully-past-prompt example in the issue, but it does not enforce the underlying contract for an extend that begins within the precomputed positions and ends beyond them.

## Finding

In `ForwardBatch._compute_mrope_positions_extend`, the candidate only enters its new expansion path when the precomputed slice has zero elements. A partially available slice remains non-empty, so it is accepted even when it has fewer columns than `extend_seq_len`.

Independent counterexample with a 16-position multimodal prompt:

- `prefix_len = 14`
- `extend_len = 4`
- forwarded tokens = 4
- candidate output shape = `[3, 2]`

This reproduces on CPU and on the assigned gfx950 GPU. It leaves the rotary embedding with fewer positions than query rows, the invariant whose violation causes the original native shape mismatch or fused-kernel out-of-bounds access. The submitted tests cover an entirely in-range slice and an entirely out-of-range slice, but not a slice crossing the boundary.

The reported EAGLE3 state where decoding has already moved beyond the prompt does take the empty-slice path. For that exact state, the candidate changes `[3, batch_size]` to `[3, batch_size * num_draft_tokens]`, and its delta-shifted values match an independent arithmetic reference. Thus this is a substantive partial source fix, not test-only hardening, but it does not fully establish the general one-position-per-forwarded-token contract claimed by the issue and regression.

## Evidence

- Untouched recorded base: 10 requests x 4 draft tokens produced shape `[3, 10]`, confirming the original failure deterministically.
- Exact candidate: all four submitted CPU regression tests passed.
- Exact candidate: the issue's fully-past-prompt case produced `[3, 40]` as expected.
- Exact candidate on AMD Instinct MI355X (gfx950): a ragged `[2, 5]` extend produced `[3, 7]` and exactly matched a separately constructed CPU arithmetic reference.
- Exact candidate on the same GPU: the crossing case produced `[3, 2]` for four forwarded tokens and failed the independent assertion.
- Import verification showed both `sglang` and `forward_batch_info` loaded from `/job/repo/python`, so the tests exercised the checked-out source.
- The candidate changes only Python and test/report files. No native source changed, so no native rebuild was applicable.

Raw commands and output are retained under `reports/j-ae2eedac3bc7/raw/`.

## Limitations

The available device is one AMD Instinct MI355X (gfx950) with PyTorch ROCm 7.2. The original report used NVIDIA H100/CUDA and the fused Triton CUDA mRoPE path. Qwen3-VL and EAGLE3 checkpoint weights were not available, so the full HTTP/model-serving failure was not rerun. The supplied tiny Llama fixture cannot qualify Qwen3-VL mRoPE or EAGLE3 and was not substituted. GPU execution here validates position construction and transfer only, not the CUDA fused rotary kernel, model semantics, throughput, multi-GPU, or multi-node behavior.
