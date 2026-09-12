# Independent review of PR 762 at `26f1a21`

Upstream issue: https://github.com/sgl-project/sglang/issues/22935

Mirror issue: https://github.com/amdpilot-org/sglang/issues/789

Candidate: https://github.com/amdpilot-org/sglang/pull/762 at
`26f1a21ab632cea2bb9001ce6bd7277dcd464958`.

## Recommendation

Reject as a fix for the original open issue. The candidate is test-only
hardening that documents checkpoint ownership; it does not modify runtime or
native source and does not make the reported fresh-prefill prefix reusable.

## Independent findings

The prepared base first reproduced the original sequence using the actual
`MambaRadixCache`. Inserting `[10,20,30]` created one compressed edge with its
Mamba state owned at depth 3. Matching `[10,20]` split that edge into a depth-2
parent and depth-3 suffix. The parent had `mamba_value=None`, the suffix retained
the depth-3 state, and the returned `device_indices` length was 0.

At the exact candidate commit, the same independent probe produced the same
result. The candidate's focused suite passed 20 tests, but its new regression
explicitly asserts the zero result. A passing test therefore proves the current
ownership rule, not resolution of the original issue.

Positive controls confirmed that exact state ownership is reusable: a real
depth-2 checkpoint returned two indices, and an arbitrary depth-37 checkpoint
returned 37. Both GPU tensors equaled independent `torch.arange` references.
Conversely, divergence before the first depth-64 checkpoint returned zero and
selected the root as the last reusable state node.

These controls also show why copying the depth-3 state onto the new depth-2
split node would be invalid: recurrent and convolution state belongs to one
exact token depth. A full solution still needs either complete checkpoint
production at the desired arbitrary depth, or execution that can retain deeper
attention KV while separately replaying recurrent layers.

## Classification

- Full original-issue fix: no.
- Partial runtime fix: no.
- Test-only hardening: yes.
- Unverified runtime claim: the candidate correctly avoids claiming a runtime
  fix, but its PR cannot be accepted as resolution of the original issue.

## Environment and paths

- Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`.
- Interpreter: `/tmp/amdpilot-repo-j-d925e51cded9/venv/bin/python`.
- Imported package: `/job/repo/python/sglang/__init__.py`.
- Imported cache source:
  `/job/repo/python/sglang/srt/mem_cache/mamba_radix_cache.py`.
- Torch/ROCm: `2.11.0+rocm7.2`.
- GPU: AMD Instinct MI355X.
- Candidate changed only reports and Python tests. No native source changed, so
  a native rebuild was not applicable.

No hybrid Mamba weights were available. Thus the GPU evidence covers the real
cache implementation and device-resident indices, but not end-to-end logits,
logprobs, latency, convolution state capture, or a recurrent replay execution
path. A tiny Llama transport fixture would not qualify this architecture and
was intentionally not substituted.

## Commands

```bash
PYTHONPATH=/job/repo/python \
  /tmp/amdpilot-repo-j-d925e51cded9/venv/bin/python \
  /job/review-evidence-j-d925e51cded9/independent_probe.py

PYTHONPATH=/job/repo/python \
  /tmp/amdpilot-repo-j-d925e51cded9/venv/bin/python -m pytest -q \
  test/registered/unit/mem_cache/test_mamba_split_checkpoint_ownership.py \
  test/registered/unit/mem_cache/test_mamba_unittest.py
```

Raw measured output is retained in `evidence.txt`; the complete revision-switch
evidence was also preserved outside the checkout under
`/job/review-evidence-j-d925e51cded9/`.
