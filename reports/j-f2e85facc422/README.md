# Independent review of overlap-spec candidate

Upstream issue: https://github.com/sgl-project/sglang/issues/11762

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3169

Candidate: https://github.com/amdpilot-org/sglang/pull/3164 at
`476b730e789427a42b363114e5523c1dec4264cb`

## Recommendation

Request changes. The candidate is test/report-only and does not fully resolve the
original overlap-spec roadmap. Its exact relay regression passes unchanged on
the recorded base (`4 passed`), so it supplies test hardening but no
failing-before/passing-after implementation correction. The candidate diff is
three added files: a CPU relay test and its prior review reports; there is no
Python implementation or native-code change.

The concrete NGRAM overlap execution path remains blocked on both CPU and the
assigned AMD Instinct MI355X. Repository source contains CPU and GPU definitions
for `reconstruct_indices_from_tree_mask`, but the required interpreter imports
`sgl_kernel` and its wrapper from the pinned
`sglang_kernel-0.4.6.post1-py3.12-linux-x86_64.egg`. That loaded Torch namespace
registers neither `reconstruct_indices_from_tree_mask_cpu` nor
`reconstruct_indices_from_tree_mask`, and direct invocations fail with
`AttributeError`. No candidate native source changed, and the prepared
environment supplies no native artifact or rebuild command, so no rebuild was
applicable or justified for this review.

The candidate's relay test passes at both revisions. Two independent edge cases
(zero accepted tokens combined with a full-width neighboring row, tail
truncation, and an empty batch) also pass. These establish only the isolated
Python relay behavior. They cannot pass through `_prepare_for_speculative_decoding`,
which immediately requires the missing reconstruction operator.

The real-GPU speculative KV-index grid passes all three cases, including its
independent Python reference for top-k 4/page-size 16. This is adjacent kernel
coverage, not NGRAM serving or proof of the complete feature request.

## Reproduction

Use the interpreter required by `REPOSITORY.md` and set `PYTHONPATH` to the
checked-out repository's `python` directory.

Base and candidate relay regression:

```bash
python -m pytest -q test/registered/cpu/test_ngram_overlap_relay.py
```

CPU reconstruction failure:

```bash
python -m pytest -q \
  test/registered/cpu/test_spec_kernels.py::TestReconstructIndicesFromTreeMask::test_tree_and_chain
```

MI355X GPU reconstruction failure and adjacent grid:

```bash
HIP_VISIBLE_DEVICES=0 python python/sglang/kernels/aot/tests/speculative/test_ngram_utils.py
HIP_VISIBLE_DEVICES=0 python -m pytest -q \
  test/registered/kernel/speculative/test_spec_kv_indices_grid.py
```

Raw base/candidate logs, the independent adversarial test, import-path output,
and candidate metadata were preserved outside the revision-switching checkout
under `/job/review-evidence/` during review.

## Scope still unresolved

End-to-end overlap NGRAM serving is unverified because its required native
operator is absent. The candidate also neither implements nor qualifies the
open roadmap items: combined top-k/page-size serving, allocation optimization,
optional verify synchronization, penalty support, universal
`SpecTpWorker`/`TpModelWorker` parity, high-throughput specialization, EP/DeepEP,
PD disaggregation, and a separate plan stream. No independent EAGLE semantic
accuracy, attention-backend matrix, distributed DP/EP, PD-disaggregation, or
LoRA serving run was possible from the supplied environment and weights. The
MI355X evidence must not be generalized to unavailable NVIDIA architectures or
distributed topologies.
