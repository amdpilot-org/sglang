# Investigation result: issue already fixed on prepared `main`

Source issue: https://github.com/sgl-project/sglang/issues/33142

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1946

The prepared base (`358c163250ad3b1f62939b01ce1314a0a31a0365`) already contains the
stable-public-API rewrite requested by the source issue. The current kernel and
registered regression test are byte-for-byte identical to the versions in the
merge commit for upstream PR #36349 (`1fb85053e74c3263c79457a14f589002e3cb8c31`),
"Migrate FlyDSL fused norm kernels to the v0.3.0 stable API." No additional
kernel change was made because the issue-specific defect is no longer present.

## Reproduction and verification

The historical file was downloaded from the exact revision named by the issue,
`2573190b9377debd135417f25b0c4185b0a3c48a`, into the private runtime under
`/tmp`. Executing it with the prepared interpreter and installed FlyDSL 0.3.2
fails at import exactly as reported:

```text
ImportError: cannot import name 'buffer_ops' from 'flydsl.expr'
```

The current replacement imports from the installed wheel successfully. A grep
for the issue's private/deprecated dependencies (`flydsl._mlir`, `buffer_ops`,
`index_cast`, `CompilationContext`, and `_unwrap_value`) returns no matches in
the current kernel. It also has an import-time capability guard which converts
missing v0.3 stable APIs into `ImportError`, preserving the layer's native
fallback contract on older FlyDSL installations.

The complete registered test file passed on the assigned single AMD Instinct
MI350X (`gfx950:sramecc+:xnack-`) with ROCm 7.2, Torch 2.11.0, and FlyDSL 0.3.2:

```text
22 passed, 1 warning in 11.45s
```

Those tests compare GPU output with independent fp32 PyTorch reference chains
and cover RMSNorm and LayerNorm, fused and non-fused entry points, gate/no-gate,
affine/no-affine, broadcast/per-row/4D scale-shift layouts, `D=10240` multi-tile
execution, row-count/layout compile-cache reuse, and wheel-only import without
a FlyDSL source tree.

## Evidence

Raw logs are retained in `reports/j-2f93e5f9c583/evidence/`:

- `historical-import.log` and `.exit`: failing-before import at the issue revision.
- `current-import.log`: passing current import from the installed wheel.
- `full-gpu-tests.log` and `.exit`: all 22 registered GPU regressions.
- `focused-gpu-tests.log` and `.exit`: independently selected boundary cases.
- `environment.log`: interpreter, ROCm, FlyDSL, and assigned GPU details.
- `fix-content-diff.numstat`: empty because both fixed source/test files match
  PR #36349's merge commit exactly.
- `current-private-api-grep.txt`: empty private/deprecated dependency audit.

## Limitations

No diffusion model weights were available or needed for this import-time kernel
defect, so no full model/server or semantic-quality reproduction was performed.
Testing is limited to one gfx950 GPU; no gfx942 or multi-GPU claim is made. The
historical kernel could not execute numerically because its reported import
failure occurs first. No native C++ component was changed or rebuilt.
