# Independent review of PR 3308

Candidate: https://github.com/amdpilot-org/sglang/pull/3308

Exact commit: `4b304308c20d3e8b96932ea40d156cc452c93ea2`

Upstream issue: https://github.com/sgl-project/sglang/issues/38980

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3309

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Finding

Recommendation: **unverified**. The candidate fixes the independently testable
Python API contract, but this environment cannot verify the native part of the
original SM89 launch failure. It should not be described as a fully verified
resolution of the original issue until the pinned CUDA source is rebuilt and
the reported head-dimension-256 split/local cases are executed numerically on
SM89.

On the recorded base, both public wrappers expose `ver=3`; passing `ver=2` to
`flash_attn_with_kvcache` reaches the same `sgl_kernel::fwd` boundary. The base
also reports SM89 supported, and its lexicographic CUDA version comparison
incorrectly rejects `12.10`. The candidate removes `ver` from both wrappers,
uses numeric CUDA version comparison, passes each input tensor's device to the
support check, rejects unsupported devices before the native boundary, and
retains SM8x support.

Removing `ver` is one of the original issue's explicit acceptable outcomes.
Source inspection confirms that these wrappers have only one registered native
operation (`sgl_kernel::fwd`); there is no separately wired FA2 operator to
which `ver=2` could honestly dispatch.

The candidate also updates the pinned `sgl-attn` archive from
`f89bc2306632d1ec5f97b014dded4254f5b4a907` to
`149a54916c6a1b940f7ff1afd95f078349a9c471`. Independently downloading that
archive produced the declared SHA256
`98cbb200b5057db2b4a40fdd80fb6eae65e784c751ce6d05a9b220f70b797879`.
Inspection confirms changes to SM8x shared-memory reuse and local KV block
coordinates, plus a CUDA numerical regression covering SM86/SM89, fp16/bf16,
head dimensions 192/256, split/local/paged cases, persistent work, and CUDA
Graphs. This is source evidence only; the test could not execute locally.

## Evidence

- Recorded-base instrumented entrypoint: both signatures contained `ver`; SM89
  returned true; CUDA 12.10/SM90 returned false; `ver=2` invoked the same mocked
  native boundary once. Exit 0. Raw output:
  `/tmp/amdpilot-review-j-93db523e13fe/base_contract.txt`.
- Exact-candidate regression:
  `/tmp/amdpilot-repo-j-93db523e13fe/venv/bin/python -m pytest -q python/sglang/kernels/aot/tests/test_flash_attn_public_contract.py`
  returned 0 with `16 passed`.
- Independent exact-candidate adversarial script loaded
  `/job/repo/python/sglang/kernels/aot/python/sgl_kernel/flash_attn.py` while at
  the candidate commit. It checked CUDA 12.2, 12.3, 12.10 and 13.0; SM80,
  SM89, SM90, SM91 and SM100; both wrapper device-routing paths; rejection
  before the backend; and removal of `ver=2`. Exit 0. Raw output:
  `/tmp/amdpilot-review-j-93db523e13fe/candidate_adversarial.txt`.
- Candidate Python compilation and `git diff --check` returned 0.
- Installed-path inspection found the prepared package at
  `/opt/venv/lib/python3.12/site-packages/sglang_kernel-0.4.6.post1-py3.12-linux-x86_64.egg/sgl_kernel`.
  It contains `common_ops` but no importable `flash_ops`, so it is not a usable
  candidate native library. Candidate tests therefore loaded the checked-out
  source explicitly and instrumented the native boundary.

## Native rebuild and architecture limitation

The assigned GPU is AMD Instinct MI350X under Torch `2.11.0+rocm7.2`/HIP 7.2.
There is no NVIDIA GPU, `nvcc`, or `cuobjdump`. A wheel build stopped before
configuration because the package requires Torch 2.13.0. Direct CMake
configuration with the prepared Torch prefix then failed with `Failed to find
nvcc`. No candidate CUDA library was produced, no cubin was inspected, and no
SM86 cubin was executed on SM89.

Consequently these original-issue cases remain counterexamples requiring
verification rather than observed candidate failures:

1. SM89, fp16 and bf16, head dimension 256, `num_splits=2` (the reported launch
   failure), checked against an independent attention reference.
2. SM89 local split and shuffled paged-KV cases, including CUDA Graph replay
   after changing V.
3. Binary inspection proving the rebuilt `flash_ops` contains the intended
   SM8x code, followed by actual SM86-cubin execution on SM89.

The Python/API portion is verified; the native original failure is not.
