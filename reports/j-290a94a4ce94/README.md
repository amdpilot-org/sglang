# Investigation of sglang#34367

Upstream issue: https://github.com/sgl-project/sglang/issues/34367

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1614

## Result

The reported pre-denoising frame-divisibility failure is not reproducible in the
prepared source at base commit `358c163250ad3b1f62939b01ce1314a0a31a0365`.
The source already contains the model-specific correction and regression coverage
introduced with LongLive2 support in commit
`cfe4eefabbc1b8d3703cf134f76e3387c76caa72`.

`LongLive2T2VConfig.adjust_num_frames()` first applies Wan temporal alignment and
then ensures that the resulting latent frame count is a multiple of
`num_frames_per_block`. The checked-in `longlive2_i2v` CI request is 61 output
frames. With temporal VAE scale 4 it maps to 16 latent frames, exactly two causal
blocks of 8, so it cannot reach the reported divisibility exception.

The focused evidence test also reconstructs the faulty condition without the
LongLive2-specific correction: a 17-frame request remains 17 under the inherited
Wan-only adjustment and maps to 5 latent frames, which is not divisible by 8.
The actual LongLive2 implementation instead adjusts it to 29 output frames and 8
latent frames. Independent cases cover the first valid boundary, rounding on both
sides of the checked-in request, an incompatible 17-latent-frame request, and a
later exact block boundary.

No production source correction is justified because the requested solution is
already present. This PR retains the reproducible investigation and machine-readable
result only.

## Reproduction

From the repository root:

```bash
PYTHONPATH=/job/repo/python \
XDG_CACHE_HOME=/tmp/amdpilot-repo-j-290a94a4ce94/cache \
/tmp/amdpilot-repo-j-290a94a4ce94/venv/bin/python -m pytest \
  reports/j-290a94a4ce94/test_issue_34367.py \
  python/sglang/multimodal_gen/test/unit/test_longlive2_pipeline_config.py \
  -q --junitxml=reports/j-290a94a4ce94/evidence/regression.xml
```

Observed: `12 passed`.

## Limitations

The `Rabinovich/LongLive-2.0-5B-Diffusers` weights were not present in the prepared
runtime, so a full HTTP server/model execution was not performed. The assigned
single AMD Instinct MI355X (`gfx950`) was detected, but this issue is resolved in
CPU-side request normalization before denoising, so no GPU execution is claimed.
This investigation does not establish model output quality or semantic accuracy.
