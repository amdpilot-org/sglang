# Independent review of PR 1531

Upstream issue: https://github.com/sgl-project/sglang/issues/34772

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1568

Candidate: https://github.com/amdpilot-org/sglang/pull/1531 at
`2f7f86127908c8e7c766ecc070976330c45badc9`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`. The image-prepared
checkout was exactly this commit, with no revision difference.

## Recommendation

Accept. The original `should_offload(..., model_config=None)` implementation
described by the report has been replaced by a residency-policy refactor on the
recorded base, but the issue's observable contract still failed: a non-quantized
native Transformers fallback did not consult `should_start_component_on_cpu`
until after `from_pretrained` returned. That permitted transient GPU
materialization before the common final placement.

The candidate supplies a CPU `device_map` during `from_pretrained` whenever the
effective residency policy says the component starts on CPU. Independent tests
confirmed this for explicit/component offload, a layerwise-offload-equivalent
decision, and the exact `text_encoder_2` key. Resident loading remains without a
forced device map, and Transformers-managed quantized loading retains its
existing GPU-resident precedence.

On the exact base, all non-quantized cases ignored the CPU-start decision and
passed no `device_map`. At the exact candidate, the CPU-start cases passed
`{"": cpu}` and marked native placement as managed. A generated tiny T5 fixture
then exercised real `from_pretrained` on the assigned AMD Instinct MI350X
(gfx950, ROCm 7.2): all parameters remained on CPU and PyTorch GPU allocation
stayed at zero bytes.

This is a source-only Python change. No C++/HIP/FlyDSL/native source changed, so
there was no native library to rebuild. The imported loader source resolved to
the active checkout under `/job/repo/python/sglang` at both revisions.

## Limits

The reporter's Wan and Sana weights were unavailable. The assigned GPU was a
256 GiB AMD Instinct MI350X under ROCm rather than an 8 GiB NVIDIA RTX 4070
Laptop under CUDA/WSL2. Therefore the exact full diffusion load, fallback
trigger, and OOM were not reproduced. The tiny fixture establishes native
Transformers placement and allocator behavior only; it does not establish
generation semantics, the reported architectures, or CUDA-specific behavior.

No remaining source-level counterexample was found for the reported
non-quantized native Transformers text-encoder fallback contract. Native
Diffusers loading and Transformers-managed quantized components have different
loading/residency contracts and were not treated as evidence for this issue.

Raw command output is retained in `reports/j-ad279a68e0fa/raw/`.
