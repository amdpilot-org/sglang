# Overlap-spec candidate correction review

Upstream issue: https://github.com/sgl-project/sglang/issues/11762

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3111

Candidate: https://github.com/amdpilot-org/sglang/pull/2980 at
`b6b86516be14b5c95237cf20e517bef9d4566e63`

Independent review: https://github.com/amdpilot-org/sglang/pull/3088

## Outcome

The candidate is rejected as a fix, while its valid CPU relay regression is
preserved as test hardening.  Its diff from the prepared base contains only
that test and reports; there is no implementation or native change.  Running
the exact candidate test against the recorded base passed all four cases, so
there is no failing-before/passing-after correction to preserve:

```text
prepared base 358c1632...: 4 passed
candidate b6b86516...:     same implementation; test/report-only diff
```

The pinned `sglang-kernel 0.4.6.post1` wrapper imports, but its loaded Torch
namespace registers neither `reconstruct_indices_from_tree_mask_cpu` nor
`reconstruct_indices_from_tree_mask`.  Direct CPU and MI355X GPU invocations
both fail with `AttributeError`.  This blocks end-to-end overlap NGRAM before
the serving contract can be tested.  `repository-environment.json` provides no
native rebuild command or replacement wheel, so no dependency was replaced
and no speculative source fallback was introduced.

The real-GPU speculative KV-index grid passed all three tests, including an
independent Python reference for top-k 4/page-size 16.  This is adjacent kernel
coverage only; it does not invoke the missing reconstruction operator or prove
NGRAM HTTP/serving behavior.

## Reproduction

Use the required interpreter and source checkout:

```bash
PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-60e55b1e6fb4/venv/bin/python \
  -m pytest -q test/registered/cpu/test_ngram_overlap_relay.py

PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-60e55b1e6fb4/venv/bin/python \
  -m pytest -q test/registered/cpu/test_spec_kernels.py::TestReconstructIndicesFromTreeMask::test_tree_and_chain

HIP_VISIBLE_DEVICES=0 PYTHONPATH=/job/repo/python \
  /tmp/amdpilot-repo-j-60e55b1e6fb4/venv/bin/python \
  python/sglang/kernels/aot/tests/speculative/test_ngram_utils.py

HIP_VISIBLE_DEVICES=0 PYTHONPATH=/job/repo/python \
  /tmp/amdpilot-repo-j-60e55b1e6fb4/venv/bin/python -m pytest -q \
  test/registered/kernel/speculative/test_spec_kv_indices_grid.py
```

Raw logs are retained under
`/tmp/amdpilot-repo-j-60e55b1e6fb4/evidence/`.  Source native bindings are at
`python/sglang/kernels/aot/`; the loaded binary package is
`/opt/venv/lib/python3.12/site-packages/sglang_kernel-0.4.6.post1-py3.12-linux-x86_64.egg`.

## Remaining limitations

No qualified EAGLE weights, attention-backend matrix, distributed DP/EP or
DeepEP topology, PD-disaggregation setup, or LoRA serving model was available.
The candidate also neither implements nor qualifies unchecked top-k/page-size
combinations at the serving level, memory over-allocation work, optional verify
synchronization, penalties, universal worker parity, high-throughput
specialization, EP/DeepEP, PD disaggregation, or a separate plan stream.  The
qualified tiny Llama fixture was inspected but not used as semantic evidence:
random Llama weights can validate transport/engine execution only, and the
missing native operator blocks the requested NGRAM path first.
