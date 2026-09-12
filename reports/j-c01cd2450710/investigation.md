# Investigation: fastsafetensors multi-node device selection and GDS fallback

Upstream issue: https://github.com/sgl-project/sglang/issues/29272

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2435

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Findings

The prepared base still used `pg.rank()` (the global distributed rank) as the
CUDA device index in `fastsafetensors_weights_iterator`. A deterministic
regression with global rank 11 and local CUDA device 0 showed that the base
selected `cuda:11` instead of `cuda:0`.

The base already exposed `enable_gds` through `--model-loader-extra-config`,
but GDS remained enabled by default and an `is_gds_supported(0) failed`
exception from `SafeTensorsFileLoader` propagated without retry. The regression
captured in `raw/failing_before.log` failed for both reported mechanisms.

Related upstream PRs #26597, #29717, and #32199 were inspected. They remained
open at investigation time and their complete proposed solutions were not in
the prepared base. The base contained the explicit `enable_gds` configuration,
but not local-device selection or automatic fallback after a GDS-specific
failure.

## Correction

The iterator now uses `torch.cuda.current_device()`, which reflects the local
device already selected by the model worker. When GDS is enabled, loader
construction or the initial file copy is retried once with `nogds=True` only if
the exception identifies GDS. Explicitly disabled GDS is unchanged, and
unrelated exceptions continue to propagate.

## Limitations

The assigned environment contains one AMD Instinct MI355X (`gfx950`) GPU, not a
multi-node deployment, and does not contain GLM-5.2 weights or the
`fastsafetensors` package. Therefore the original 4-node model launch was not
performed. The regression validates the exact rank/device mapping and fallback
control flow with test doubles. The GPU check only establishes real execution
and numerical agreement on the assigned device; it does not qualify
fastsafetensors, GLM/DeepSeek-V2 loading, semantic accuracy, or distributed
execution.
