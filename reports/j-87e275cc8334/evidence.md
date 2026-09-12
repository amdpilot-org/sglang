# Independent review of amdpilot-org/sglang PR 1176

Reviewed exact candidate commit `d06735d4c13c4b56b17717aefbbc1203af5ee06b`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

- Upstream issue: https://github.com/sgl-project/sglang/issues/36616
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1209
- Candidate: https://github.com/amdpilot-org/sglang/pull/1176

## Finding

Recommendation: **accept**, as test-only hardening. The candidate changes no
production source or native code. It adds a focused regression for the exact
reported tensor name and scalar, plus useful shape and unrelated-name
boundaries. The recorded base already has the relevant source behavior in
`python/sglang/srt/models/qwen4_exp.py`: `_load_qwen4_exp_ple_buffer` recognizes
`weight_scale`, copies it into the registered persistent buffer, and runs before
the generic unknown-scale assertion.

This review does **not** classify the candidate as a fully verified resolution
of the original report. The unmodified recorded base did not reproduce the
failure, and the reported Qwen 3.8 checkpoint/container was unavailable. The
candidate demonstrates the proximate loader contract and protects it against
regression, but cannot prove a complete model load in the reporter's artifact.

## Reproduction and sensitivity

On the exact candidate, the added file passed:

```text
3 passed, 17 warnings in 8.95s
```

For sensitivity, `"weight_scale"` was temporarily removed from the existing PLE
buffer allowlist, without changing the test. The reported-value test then failed
through the actual `load_weights` method with:

```text
AssertionError: Expected 1.0, got 0.00019931793212890625 in skipped model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale
1 failed, 17 warnings in 8.65s
```

The temporary edit was removed before returning to the prepared branch. This is
a counterfactual sensitivity check, not a claim that the unmodified recorded
base failed.

## Independent adversarial and GPU checks

Using the prepared interpreter and `PYTHONPATH=python` loaded:

```text
/job/repo/python/sglang/srt/models/qwen4_exp.py
```

On the assigned `AMD Instinct MI355X`,
`gfx950:sramecc+:xnack-`, the actual helper/load path copied the reported FP32
value into a GPU BF16 buffer. The observed GPU value
`0.00019931793212890625` exactly matched an independently converted CPU BF16
reference, and the returned loaded set contained the exact checkpoint name.

Two independent negative cases exercised the complete `load_weights` path:

- an unrelated `model.layers.1.mlp.weight_scale` value of `0.25` was rejected by
  the generic assertion rather than incorrectly claimed by the PLE helper;
- the reported PLE name with no matching registered destination was likewise
  rejected rather than incorrectly marked loaded.

Raw command output was retained outside the checkout in
`/job/review-evidence-j-87e275cc8334/` while revisions were switched.

## Source, native, and limitations

The prepared interpreter imported SGLang and `qwen4_exp.py` from this checkout,
not from an installed SGLang wheel. The candidate changes only Python tests and
reports; no C++, FlyDSL, or other native source changed, so a native rebuild was
not applicable. Torch remained `2.11.0+rocm7.2` with HIP `7.2.26015`.

The full reported Qwen 3.8 weights and original container were not available.
Therefore full model construction, all checkpoint tensors, HTTP serving,
semantic accuracy, and distributed execution remain unverified. The tiny Llama
fixture was not substituted because it cannot qualify this Qwen-specific loader
contract.
