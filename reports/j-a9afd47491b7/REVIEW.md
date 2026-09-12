# Independent review of PR 3224

Candidate: https://github.com/amdpilot-org/sglang/pull/3224 at `362ed639d20c2e872dd054f05c609d8e8bb0baad`

Upstream issue: https://github.com/sgl-project/sglang/issues/1715

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3230

Recommendation: **request changes**. The candidate is a meaningful partial eager-decode implementation, but it does not fully resolve the original cascade-attention integration request and its actual FlashInfer execution remains unverified on the prepared AMD host.

## Findings

1. The implementation is explicitly eager-only. `init_forward_metadata` can select cascade for ordinary eager decode, while CUDA-graph replay continues to use the existing decode wrappers. This leaves the principal graph replay serving path outside the feature.
2. Sliding-window attention and encoder-decoder cross attention do not allocate cascade wrappers. Speculative decode and dequant-workspace KV configurations also bypass cascade. These are deliberate fallbacks, not evidence of full integration.
3. The only real `BatchPrefillWithPagedKVCacheWrapper.forward_return_lse` plus production `_safe_merge_state` test is CUDA/FlashInfer-gated and skipped here. The six passing tests exercise selection logic, mocks, and the mathematical split-softmax identity; they do not establish wrapper planning, cache layout, or kernel compatibility.
4. No NVIDIA end-to-end shared-prefix serving workload or benchmark was run, so intended serving correctness and latency/throughput effects are unverified.
5. Unlike the earlier parent candidate discussed in this correction series, exact candidate `362ed63` passes `git diff --check` against recorded base `358c163`.

No native source changed in this candidate, so a native rebuild was not applicable. Python import inspection confirmed the reviewed module came from `/job/repo/python/sglang/srt/layers/attention/flashinfer_backend.py` at the exact candidate commit rather than an installed copy.

## Reproduction and checks

- Recorded base `358c163`: the candidate regression fails with missing cascade helpers/metadata/updater (`11 failed, 2 passed, 1 skipped`, counting seven failed subtests). This preserves failing-before evidence for the newly introduced eager functionality.
- Exact candidate `362ed63`: `6 passed, 1 skipped, 7 subtests passed`. The skipped case is precisely the real NVIDIA FlashInfer wrapper and merge-kernel integration.
- Independent adversarial checks ran on one AMD Instinct MI350X and passed for first-mismatch prefix handling, profitability boundaries, an all-shared/zero-suffix page-table shape, and confirmation of the explicit mode exclusions.
- Changed Python source and test compile successfully.
- `git diff --check 358c163..362ed63` exits zero.

Raw commands, outputs, source diff, issue/PR snapshots, import paths, and exit codes are retained in `reports/j-a9afd47491b7/raw/`.

## Environment limitation

The prepared interpreter is Python 3.12.3 with Torch 2.11.0+rocm7.2 on an AMD Instinct MI350X. `torch.version.cuda` is null and FlashInfer is unavailable. The GPU executed the candidate's independent PyTorch numerical identity and the separate adversarial checks, but no FlashInfer CUDA kernel executed. The tiny Llama transport fixture cannot qualify a CUDA-only FlashInfer backend and was therefore not used as unrelated proof.
