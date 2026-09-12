# Independent review of amdpilot-org/sglang PR 1802

Candidate reviewed: exact commit `4834828b12073e0166bf3e154d5e89bba1d836ab`.

Upstream issue: https://github.com/sgl-project/sglang/issues/33695

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1839

Recommendation: **accept**. The candidate fully resolves the original issue's
narrow contract in the directly affected PyTorch sampler.

On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the issue's exact
128-row call imported `/job/repo/python/sglang/srt/layers/sampler.py` and failed
at its seeded-min-p assertion before sampling. At the exact candidate commit,
the candidate's three GPU regressions passed. An independent 4,096-row case
combined top-k, top-p, min-p, per-row deterministic seeds, nonconstant
positions, and vocabulary reordering. Candidate output replayed exactly,
matched a separately constructed and explicitly normalized filtered-weight
reference, and never selected the min-p-filtered token. The original issue
fixture selected only original vocabulary IDs 1 and 3, with counts 76 and 52.

The source change only removes the four-line assertion/TODO block. This is
sufficient because the existing path converts zeroed weights to `-inf`, adds
deterministic Gumbel noise, takes `argmax`, and gathers through the sorting
permutation. Per-row normalization adds one constant to all finite log-weights
and therefore cannot change that argmax. The independent numerical comparison
confirmed the reasoning on the assigned GPU rather than accepting PR prose.

No native C++, HIP, or FlyDSL source changed, and the prepared environment has
no separate native artifact configured. A native rebuild was therefore not
applicable. The measured import path was the checked-out repository source,
not an installed sampler module.

## Limitations

- Execution used one AMD Instinct MI350X (`gfx950`, capability `(9, 5)`) with
  ROCm 7.2 and PyTorch `2.11.0+rocm7.2`; no other architecture was tested.
- No model weights, HTTP server, multi-GPU, or multi-node run was needed or
  used. This review establishes the original internal sampling-function
  contract, not full-model semantic accuracy or distributed behavior.
- Torch emitted a non-fatal Dynamo metrics-logging traceback (`Object of type
  function is not JSON serializable`) during the independent run. The process
  exited zero after all assertions and printed its results.
- Current mirror `main` had advanced to
  `a207786205bff0919eb2c8c9126c67f302ccff34` when reviewed; the required
  failing-before comparison remained the recorded base above.
