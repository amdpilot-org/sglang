# Independent review of candidate PR 2766

Upstream issue: https://github.com/sgl-project/sglang/issues/33846

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2720

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2800

Candidate: https://github.com/amdpilot-org/sglang/pull/2766 at exact commit
`3d4404dc8451656c212c758a5086d1ade93c4170`.

## Recommendation

**Accept as test-only hardening.** The candidate adds a focused regression for
an implementation fix that is already present in the recorded base. It does
not itself implement or fully revalidate the original issue fix.

The exact candidate changes only a unit test and investigation artifacts. It
does not change Python runtime implementation, Triton kernels, C++, or another
native source. Consequently no native rebuild is applicable. Imports during
review resolved to the prepared checkout, including
`/job/repo/python/sglang/srt/layers/attention/linear/kda_backend.py` and
`/job/repo/python/sglang/srt/managers/overlap_utils.py`, rather than an installed
copy.

## Original contract and classification

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` already has
`KDAAttnBackend.needs_cpu_seq_lens = False`, originating from merged upstream
PR 32219. The original issue reporter later reported ten c=1 rounds on the
original 8xMI350X Kimi-K3 MXFP4 DSPARK configuration with the isolated flag
change and no hang. That is relevant original-scale evidence, but it was not
produced by candidate PR 2766 or this review.

On the prepared base the reported failure could not be reproduced because the
source fix is already present, and the required TP8 model/weights were not
available. Restoring the old boolean in a negative control reproduces the
contract violation (the scheduler selects the host-mirror path), not the
approximately 3% permanent eight-GPU serving hang. The candidate should
therefore be classified as **test-only hardening**, while the pre-existing
source change is the probable original-issue fix.

## Independent evidence

- The candidate's four regression cases pass at the exact reviewed commit.
- An independent MI350X/gfx950 exercise of the production
  `FutureMap.resolve_seq_lens_cpu` method showed `needs_cpu_seq_lens=False`
  leaves `seq_lens_cpu` and `seq_lens_sum` unset while retaining the GPU
  sequence length. The adversarial `True` control created the CPU tensor and
  sum, directly demonstrating the D2H-gating behavior beyond checking a class
  attribute.
- The production KDA Triton correctness fixture passed 10 subcases against its
  PyTorch reference on one MI350X/gfx950. This checks real kernel execution and
  numerical boundaries, but it is not proof of the scheduler/serving hang.
- The candidate's retained run identifies an MI355X/gfx950, whereas this review
  was assigned an MI350X/gfx950. Both are gfx950, but neither single-GPU run
  reproduces the original TP8 deployment.

## Remaining counterexamples and limitations

- No 8xMI350X TP8 Kimi-K3 MXFP4 DSPARK serving run was possible; the original
  low-frequency permanent hang was not independently reproduced or stress
  tested after the fix.
- The unit regression verifies scheduler selection, not absence of every
  blocking D2H elsewhere in the complete Kimi-K3/DSPARK serving path.
- The one-GPU Triton fixture does not qualify distributed stream ordering,
  DSPARK integration, model architecture behavior, or long-run hang frequency.
- No compiler or ISA claim is made, and no native source changed.
