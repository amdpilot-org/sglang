# Independent review of PR 1301

Upstream issue: https://github.com/sgl-project/sglang/issues/37755

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1351

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1301

Exact candidate: `ee046e734a4675b712b03b76695542555cf8f1fc`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

Reject as a fix for the original issue. The candidate is test-only hardening: it adds one synthetic six-test file and prior review reports, with no production or native-code change. All six candidate tests pass unchanged on the recorded base and on the exact candidate, so they provide no failing-before/passing-after evidence attributable to this candidate.

The tests are useful coverage of existing fused-QKV shard merging, incompatible-TP rejection, deinterleaving/requantization order, and projection-layout parsing. An independent adversarial check also passed for every runtime rank when selecting TP4 shards from a synthetic TP32 checkpoint, and rejected a malformed checkpoint shape. That check ran on the assigned AMD Instinct MI355X. These results validate only existing Python tensor logic.

The original failure could not be reproduced or resolved in this environment. The reported `/home/weights/MiMo-V2.5-Pro-W8A8` checkpoint is absent, and the host has no Ascend/CANN installation, `torch_npu`, `/dev/davinci*` devices, or two-node TP32/DP4 topology. Therefore actual ModelSlim tensor names, shapes, ordering, scale metadata, Ascend loading/serving, generated-token equivalence, and semantic accuracy remain unverified.

## Evidence

- Recorded base with the exact candidate test copied outside the checkout: `6 passed`, exit 0.
- Exact candidate implementation and test: `6 passed`, exit 0.
- Independent TP32-to-TP4 all-rank selection and malformed-shape case: pass on AMD Instinct MI355X, exit 0.
- Candidate versus base production/native diff: empty. No native rebuild was applicable.
- Base imports resolved to `/job/repo/python/sglang/srt/models/mimo_v2.py`; candidate imports resolved to the same source checkout path while detached at the exact candidate.

Commands and concise raw outputs are retained under `raw/`; the full preserved checkout-switch evidence is outside the checkout at `/job/review-evidence-j-b95575c4689c/`.
