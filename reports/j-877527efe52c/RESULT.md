# Independent review of PR 3014

Reviewed `https://github.com/amdpilot-org/sglang/pull/3014` at exact commit
`f5be8cecbc8a3315d28b975d40e740623e456825` against
`https://github.com/sgl-project/sglang/issues/31924`.

## Recommendation

**Accept**, as test-only hardening and an honest negative result. The candidate does
not claim or provide a production fix. It preserves useful regression coverage and
accurately records that the original issue remains unresolved.

`fully_resolves_original` is **false**. This is not a full or partial production
fix: the diff from the recorded base contains only tests and reports. The mapped
plain-loader layout mismatch still raises the same empty `AssertionError` on CPU
and the assigned AMD GPU. The BF16-style exact-name `to_q.weight` case follows the
identity path and does not benefit from fused-QKV routing.

## Independent evidence

At the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the
candidate's adversarial reproducer mapped `img_in.weight` to
`x_embedder.weight`, passed the incompatible `(2, 3)` tensor unchanged to a
plain two-argument loader for a `(3, 2)` parameter, and raised
`AssertionError('')` on CPU and gfx950. The same result occurred at the exact
candidate commit. Thus the original contract failure is reproduced before and
remains after.

The candidate's two regression tests passed at both revisions. They verify fused
QKV shard routing and preservation of a matching runtime-layout tensor. They do
not reconcile a mapped diffusers tensor whose layout differs from the runtime
parameter, exercise FSDP text-encoder reload names, replay a real checkpoint, or
re-enable the ROCm standalone test.

Imports were verified from `/job/repo/python/sglang`; Torch was the prepared
`2.11.0+rocm7.2` installation. GPU execution used one AMD Instinct MI350X,
`gfx950:sramecc+:xnack-`. No production or native source differs from the base,
so no native rebuild was applicable.

## Limitations

Qwen-Image and FLUX.2-klein-base-4B weights were not available. Therefore this
review does not claim a full-model HTTP replay, identify the historical real
tensor name/shapes, or establish that transpose/reshape is correct for any real
checkpoint. FSDP/DTensor text-encoder reload, layerwise offload, tensor parallel,
multi-GPU, and multi-node behavior remain unverified. The ROCm
`test_update_weights_from_disk.py` registration remains explicitly skipped.

Raw independently collected summaries are in `raw/base.log` and
`raw/candidate.log`; full preserved console logs also remain outside the checkout
at `/job/review-evidence-j-877527efe52c/raw/`.
