# Independent review of overlap-spec candidate

Upstream issue: https://github.com/sgl-project/sglang/issues/11762

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3250

Candidate: https://github.com/amdpilot-org/sglang/pull/3245 at exact commit
`2c09dabec1bf3c6ebc90b2747f62075c6ee14927`

Candidate parent: https://github.com/amdpilot-org/sglang/pull/3164 at exact
commit `476b730e789427a42b363114e5523c1dec4264cb`

Independent-review parent: https://github.com/amdpilot-org/sglang/pull/3201

## Recommendation

**Accept as test-only hardening and an accurate limitation report.** The
candidate does **not** fully resolve the original issue. Its diff from recorded
base `358c163250ad3b1f62939b01ce1314a0a31a0365` contains no implementation or
native change: it adds a mocked CPU relay regression and review artifacts.

The candidate does not claim that this is a full implementation. Instead, it
reports that no justified correction can be validated in the pinned prepared
environment. That statement is independently reproduced. The exact six-case
candidate relay test passes on both the recorded base and candidate, so it is
regression coverage rather than failing-before/passing-after defect repair.

## Independent evidence

| Check | Recorded base | Exact candidate | Interpretation |
| --- | --- | --- | --- |
| Candidate relay regression | 6 passed | 6 passed | Test-only hardening; no exposed base defect |
| CPU NGRAM reconstruction | Missing `reconstruct_indices_from_tree_mask_cpu` | No native change | End-to-end prerequisite blocked |
| GPU NGRAM reconstruction | Missing `reconstruct_indices_from_tree_mask` | No native change | End-to-end prerequisite blocked |
| GPU speculative KV-index grid | 3 passed | No relevant implementation change | Adjacent numerical coverage only |

The source import is
`/job/repo/python/sglang/srt/speculative/ngram_worker.py`. The loaded native
package is the pinned
`/opt/venv/lib/python3.12/site-packages/sglang_kernel-0.4.6.post1-py3.12-linux-x86_64.egg`.
It registers neither reconstruction operator. Candidate and base use the same
repository implementation; candidate changes no native source, so a native
rebuild is not applicable to evaluating its diff. The prepared environment
also supplies no native build artifact or rebuild command.

The available accelerator was one AMD Instinct MI350X with Torch
`2.11.0+rocm7.2` and HIP `7.2.26015`. Direct assigned-GPU dispatch reached the
missing operator. The independent neighboring KV-index grid passed all three
cases, including combined top-k/page-size indexing against its Python
reference, but it does not execute NGRAM reconstruction or serving.

## Classification against the original contract

This candidate is **test-only hardening**, not a full or partial implementation
of the overlap-spec roadmap. Its limitation claim is verified, not merely
prose: the added test already passes on base, and the actual CPU/GPU native
entry points fail before an end-to-end NGRAM assertion can run.

The deterministic tiny Llama fixture was not used as substitute evidence. A
transport/startup smoke cannot bypass the missing reconstruction operator or
qualify NGRAM semantic accuracy, another model architecture, distributed
execution, or the remaining roadmap features.

## Commands

```bash
PYTHONPATH=/job/repo/python \
  /tmp/amdpilot-repo-j-acf754ac0075/venv/bin/python -m pytest -q \
  test/registered/cpu/test_ngram_overlap_relay.py

PYTHONPATH=/job/repo/python \
  /tmp/amdpilot-repo-j-acf754ac0075/venv/bin/python -m pytest -q \
  /job/review-evidence-j-acf754ac0075/candidate-test_ngram_overlap_relay.py

PYTHONPATH=/job/repo/python \
  /tmp/amdpilot-repo-j-acf754ac0075/venv/bin/python -m pytest -q \
  test/registered/cpu/test_spec_kernels.py::TestReconstructIndicesFromTreeMask::test_tree_and_chain

HIP_VISIBLE_DEVICES=0 PYTHONPATH=/job/repo/python \
  /tmp/amdpilot-repo-j-acf754ac0075/venv/bin/python \
  python/sglang/kernels/aot/tests/speculative/test_ngram_utils.py

HIP_VISIBLE_DEVICES=0 PYTHONPATH=/job/repo/python \
  /tmp/amdpilot-repo-j-acf754ac0075/venv/bin/python -m pytest -q \
  test/registered/kernel/speculative/test_spec_kv_indices_grid.py
```

Raw logs are retained under `reports/j-acf754ac0075/evidence/`.

## Remaining counterexamples and limitations

- The candidate provides no failing-before/passing-after implementation fix;
  its complete relay regression passes unchanged on the recorded base.
- End-to-end overlap NGRAM serving remains unverified because both pinned
  native reconstruction operators are absent.
- Combined top-k/page-size **serving**, memory over-allocation optimization,
  optional verify synchronization, penalty support, universal worker parity,
  high-throughput specialization, EP/DeepEP, PD disaggregation, and a separate
  plan stream remain unimplemented or unqualified by this candidate.
- No independent EAGLE/NGRAM model-accuracy, attention-backend matrix,
  distributed DP/EP/DeepEP, PD-disaggregation, or LoRA serving result exists.
- Only one AMD MI350X was available. NVIDIA/CUDA backends, multi-GPU and
  multi-node architectures, qualified model weights, and disaggregated serving
  infrastructure were unavailable.
