# Overlap speculative decoding correction review

Upstream issue: https://github.com/sgl-project/sglang/issues/11762

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3204

Candidate parent: https://github.com/amdpilot-org/sglang/pull/3164 at exact
commit `476b730e789427a42b363114e5523c1dec4264cb`

Independent-review parent: https://github.com/amdpilot-org/sglang/pull/3201

## Outcome

The candidate is rejected as an implementation fix. Its diff from prepared base
`358c163250ad3b1f62939b01ce1314a0a31a0365` adds only a regression test and
reports. The `ngram_worker.py` blob is identical at base and candidate
(`826895b28eacd49f7d70e604ceb3950d19bc13fb`), and the candidate's exact four
relay cases pass on the base. There is therefore no implementation defect with
failing-before/passing-after evidence to correct in this generation.

The useful relay regression is preserved and extended with adversarial cases
for mixed zero/full-width accepted rows, trie-depth tail truncation, and an
empty batch. All six cases pass.

The pinned interpreter loads `sglang-kernel 0.4.6.post1` from
`/opt/venv/lib/python3.12/site-packages/sglang_kernel-0.4.6.post1-py3.12-linux-x86_64.egg`.
Its Torch namespace registers neither
`reconstruct_indices_from_tree_mask_cpu` nor
`reconstruct_indices_from_tree_mask`. Direct CPU and assigned-GPU executions
both fail with `AttributeError`, preventing the NGRAM path from reaching an
end-to-end serving assertion. Repository source contains these operators, but
the prepared environment supplies no native rebuild command or replacement
artifact. Replacing the pinned dependency or adding an unqualified Python
fallback would not be a justified correction.

The assigned GPU did execute the neighboring speculative KV-index grid, which
passed all three cases including top-k 4/page-size 16 against its independent
Python reference. This is adjacent numerical coverage only and is not evidence
for NGRAM serving or the full roadmap.

## Before/after evidence

| Check | Prepared base / candidate | Delivered branch | Meaning |
| --- | --- | --- | --- |
| Candidate relay regression | `4 passed` on the prepared base; implementation blob is identical in the candidate | `6 passed` | Test hardening only; no base defect |
| CPU reconstruction | Fails: missing `reconstruct_indices_from_tree_mask_cpu` | Unchanged | Pinned native dependency blocker |
| GPU reconstruction | Fails: missing `reconstruct_indices_from_tree_mask` | Unchanged | Pinned native dependency blocker |
| GPU speculative KV-index grid | `3 passed` | Unchanged | Adjacent kernel coverage only |

Raw output is retained in `reports/j-7fd89a45efb2/evidence/`.

## Reproduction

```bash
PYTHONPATH=/job/repo/python \
  /tmp/amdpilot-repo-j-7fd89a45efb2/venv/bin/python -m pytest -q \
  test/registered/cpu/test_ngram_overlap_relay.py

PYTHONPATH=/job/repo/python \
  /tmp/amdpilot-repo-j-7fd89a45efb2/venv/bin/python -m pytest -q \
  test/registered/cpu/test_spec_kernels.py::TestReconstructIndicesFromTreeMask::test_tree_and_chain

HIP_VISIBLE_DEVICES=0 PYTHONPATH=/job/repo/python \
  /tmp/amdpilot-repo-j-7fd89a45efb2/venv/bin/python \
  python/sglang/kernels/aot/tests/speculative/test_ngram_utils.py

HIP_VISIBLE_DEVICES=0 PYTHONPATH=/job/repo/python \
  /tmp/amdpilot-repo-j-7fd89a45efb2/venv/bin/python -m pytest -q \
  test/registered/kernel/speculative/test_spec_kv_indices_grid.py
```

## Remaining limitations

End-to-end overlap NGRAM remains blocked by the missing loaded native operator.
No qualified EAGLE or NGRAM semantic model weights, attention-backend matrix,
distributed DP/EP or DeepEP topology, PD-disaggregation setup, or LoRA serving
model was available. The deterministic tiny Llama fixture cannot qualify those
semantics or bypass the missing operator, so an unrelated startup smoke was not
used as evidence.

The candidate neither implements nor qualifies combined top-k/page-size
serving, memory over-allocation optimization, optional verify synchronization,
penalty support, universal `SpecTpWorker` parity, the high-throughput
specialization, EP/DeepEP, PD disaggregation, or a separate plan stream.
