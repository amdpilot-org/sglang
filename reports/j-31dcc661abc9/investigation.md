# Independent review of PR 985

Candidate: https://github.com/amdpilot-org/sglang/pull/985 at `a4c8fbf3a96e6e935572f45d581cb8554514c09a`

Upstream issue: https://github.com/sgl-project/sglang/issues/37712

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1021

## Finding

Request changes. The candidate activates the previously unused memory predicate and query-row chunking, and its regression fails on recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` (5 failed, 1 passed) then passes on the candidate (9 passed across the new and existing suites). However, the chunked PAGED top-k path slices two tensors that do not share the same coordinate space.

At candidate lines 962-976, `page_table_all[start:end]` slices the global request page table using query-row offsets, while `page_table_row_index_all[start:end]` retains global request IDs. The existing `_topk_from_kpool_logits` logic deliberately keeps the full page table when a row-index mapping is supplied. An independent GPU orchestration test uses non-identity request IDs and forces a second chunk; RAGGED passes, but PAGED fails as soon as the second chunk receives page-table rows 2:4 with row indices `[9, 9]`. This can select the wrong request table or index beyond the sliced table.

The candidate's regression did not cover this because it sets `SGLANG_DSA_FUSE_TOPK=False` and mocks `_topk_from_kpool_logits` without checking any metadata arguments. It therefore verifies chunk counts and padding but not PAGED/RAGGED mapping equivalence.

The query-only scheme also cannot make a single output row smaller than `total_k_rows * 4` bytes: `max(1, budget // bytes_per_row)` necessarily exceeds the budget when one concatenated K row is itself larger than the budget. That is not the observed 73.65 GiB case, but it is a remaining counterexample to a strict allocation-bound claim; the related upstream PR 38469 narrows K per request before row chunking.

## Environment and scope

The prepared checkout exactly matched the recorded base. Tests used `/tmp/amdpilot-repo-j-31dcc661abc9/venv/bin/python`, which imported SGLang from `/job/repo/python/sglang/...`. The assigned device is one AMD Instinct MI350X, `gfx950:sramecc+:xnack-`, with PyTorch `2.11.0+rocm7.2` / HIP `7.2.26015`.

GPU execution occurred for tensor allocation and orchestration, but DeepGEMM and the top-k operation were mocked because this is ROCm rather than the reported CUDA/B300 platform. No GLM-5.3-Flash or RadixArk weights, four B300 GPUs, TP=4/EP=4 serving workload, or NVIDIA DeepGEMM were available. Thus the original 73.65 GiB CUDA allocation and full serving behavior remain unverified here.

The candidate changes only Python and tests. There are no C++/FlyDSL/native source changes, so no native rebuild was applicable. Torch/ROCm were left intact.

## Evidence

- `base-candidate-regression.log`: candidate regression against the recorded base, 5 failed / 1 passed.
- `candidate-regressions.log`: candidate plus existing budget suites at the exact candidate, 9 passed.
- `candidate-adversarial.log`: independent exact-candidate suite, 2 passed / 1 failed; the failure exposes PAGED page-table slicing.
- `gpu-arch.log`: architecture probe excerpt. Full PyTorch device evidence is recorded in `result.json`.
