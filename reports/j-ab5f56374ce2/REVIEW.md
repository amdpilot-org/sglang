# Independent review of PR 3107

Candidate: https://github.com/amdpilot-org/sglang/pull/3107

Exact commit: `e866f0bc4107875d64c299bef69e78d94e3507a2`

Upstream issue: https://github.com/sgl-project/sglang/issues/18891

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3130

## Recommendation

Request changes. The candidate corrects both concrete false-success cases from
the prior review: checkpoint-present buffers participate in the comparison and
an explicit `module_names=[]` request is rejected. Its own focused regression
suite passes, and independent execution on an AMD Instinct MI355X confirmed
both corrections.

However, the comparison cannot replay a production-style diffusion parameter
loader that relies on the specialized parameter class. The candidate's
`_make_checksum_scratch` creates a plain `torch.nn.Parameter` and copies only
the live parameter's instance dictionary. Methods supplied by subclasses such
as `BasevLLMParameter.load_column_parallel_weight` are therefore absent.
Diffusion parallel-linear loaders call those methods (for example,
`runtime/layers/linear.py`), so a normal update can load a checkpoint while the
new comparison raises `AttributeError` for that same checkpoint.

Independent reproducer output:

```
loaded_live [5.0, 6.0]
comparison_error AttributeError 'Parameter' object has no attribute 'load_column_parallel_weight'
```

This is tied directly to the original text-encoder checksum contract, rather
than an unrelated smoke test. The candidate therefore partially fixes the
feature but does not fully resolve the original issue.

## Evidence summary

- Prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365`: the compare endpoint
  and `compare_module_weights_with_disk` are absent, reproducing the original
  missing feature.
- Parent candidate `fe932bb414defb11ff95d456848455b5b2c3fd99`: a changed live
  `running_mean` still reports `match: true`; `module_names=[]` reports
  `success: true` with an empty result.
- Reviewed candidate `e866f0bc4107875d64c299bef69e78d94e3507a2`: the same GPU case
  reports the buffer mismatch and rejects the empty selection.
- Reviewed candidate focused tests: 6 passed, including its five checksum
  tests and the selected layerwise-offload checksum test.
- Reviewed candidate adversarial production-parameter test: the real update
  path succeeds, then comparison fails with the subclass-method error above.

## Environment and limitations

- Source imports resolved from `/job/repo/python`; PyTorch resolved from the
  prepared environment and reported `2.11.0+rocm7.2` with HIP `7.2.26015`.
- One AMD Instinct MI355X was visible and used for the independent buffer
  comparison. Multi-rank TP/SP/DTensor behavior was not exercised.
- FLUX.2-klein and Qwen-Image checkpoints were unavailable, so full
  model-specific serving/API validation remains unverified.
- No native source changed in the candidate. No native rebuild was applicable;
  the loaded AIter extension came from the prepared private cache.
- The tiny Llama fixture cannot qualify diffusion VAE/text-encoder semantics
  and was not used.
