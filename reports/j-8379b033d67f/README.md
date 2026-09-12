# Investigation: Triton version mismatch on gfx1150

Upstream issue: https://github.com/sgl-project/sglang/issues/35785

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3334

The reported warning is an expected AITER compatibility guard: the AITER
revision selected by the current Docker sources warns when
`AITER_USE_SYSTEM_TRITON=1` is combined with Triton 3.4, and accepts Triton
3.6 and newer. The general ROCm Dockerfile's supported ROCm 7.2 flavors now
install a pinned Triton 3.7 wheel.

The report names `gfx1150`, but current SGLang Docker sources do not contain a
gfx1150 build stage or map that string to another architecture. Current source
does contain a separate `docker/rocm-gfx1151.Dockerfile` for Strix Halo. That
image was added by upstream PR 33939, pins an AMD ROCm/PyTorch base image,
disables AITER runtime dispatch, and documents explicit Triton attention. It is
not evidence that gfx1150 is supported, nor is gfx1151 interchangeable with
gfx1150.

No SGLang source correction is justified from the available evidence. The
original gfx1150 container/model run could not be reproduced because the
assigned device is gfx950 and no model weights were supplied. A full Docker
image build was also unavailable because this environment has no Docker or
Podman executable. The GPU check in `gpu_check.log` only validates the assigned
gfx950 ROCm execution path and is not presented as gfx1150 or model-serving
validation.

## Reproduction

The exact AITER files used for the check were retained outside the worktree at
`/tmp/amdpilot-repo-j-8379b033d67f/aiter`, checked out from the commits pinned
by the two current ROCm Dockerfiles. Run:

```bash
/tmp/amdpilot-repo-j-8379b033d67f/venv/bin/python \
  reports/j-8379b033d67f/check_issue.py \
  --aiter-source /tmp/amdpilot-repo-j-8379b033d67f/aiter/aiter/ops/triton/gluon/__init__.py
```

Raw outputs are in `check_issue.log`, `gpu_check.log`, and
`source_evidence.log`.

The limited assigned-device check is reproducible with:

```bash
/tmp/amdpilot-repo-j-8379b033d67f/venv/bin/python \
  reports/j-8379b033d67f/gpu_check.py
```
