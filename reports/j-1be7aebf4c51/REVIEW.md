# Independent review of amdpilot-org/sglang#2956

Candidate: `c356b25345ae137a2d1d233f3bd473e5067e350d`

Upstream issue: https://github.com/sgl-project/sglang/issues/12562

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2918

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2996

## Recommendation

`unverified`

The source patch addresses both requested themes: it removes `draft_probs` from
the target-only Python/native ABI and replaces the dense scratch tensor with
on-the-fly masking of rejected sibling token IDs. For top-k=1, the masking is a
single token-ID comparison per vocabulary element. However, the changed native
code could not be built or executed on the prepared machine. Therefore this
review cannot establish that the candidate fully resolves the original issue,
nor can it substantiate the performance part of the request.

## Evidence

- The prepared checkout exactly matched the recorded base
  `358c163250ad3b1f62939b01ce1314a0a31a0365`; no discrepancy was observed.
- On the base, the candidate's new regression failed because the installed
  native wrapper still required `draft_probs`. This reproduces the old API.
- At the exact candidate commit, source imports for SGLang resolved to
  `/job/repo/python/sglang/...`, but `sgl_kernel` resolved to the prepared wheel
  at `/opt/venv/lib/python3.12/site-packages/...`, whose wrapper retained the
  old ABI. Thus running the candidate tests against that wheel is not candidate
  validation: all three tests fail before native dispatch.
- A real supported native build was attempted with the pinned interpreter. It
  failed during CMake configuration because no CUDA toolkit/nvcc is installed.
  The assigned GPU is an AMD Instinct MI350X (`gfx950`) with ROCm 7.2. The ROCm
  build source list does not include `speculative_sampling.cu`, so it cannot
  produce the changed operator. No NVIDIA or MUSA device/toolchain was present.
- Independent AST checks confirmed the source wrapper has no `draft_probs`
  argument and dispatches the expected ten tensors plus three scalar controls.
  A 10,000-case CPU algebra check confirmed that the old scratch operation
  `relu(q - scratch)`, where rejected IDs store their corresponding `q`, equals
  masking those rejected IDs as implemented by the candidate. This is useful
  source-level evidence, but it does not validate compilation, device behavior,
  numerical edge cases in the sampling primitive, or performance.

Raw logs and the exact candidate diff were retained outside the checkout at
`/job/review-evidence-j-1be7aebf4c51/` while revisions were switched.

## Remaining counterexamples and gaps

- No candidate native kernel executed, including the added forced-rejection
  regression.
- No adversarial device test covered multiple rejected siblings, duplicate
  candidate token IDs, all-drafts-accepted bonus sampling, boundary coins, or
  multiple batches against an independent numerical reference.
- No benchmark demonstrates that the new top-k=1 path is more efficient. Static
  inspection only shows removal of the dense scratch allocation and one ID
  comparison per vocabulary item for a one-candidate sibling chain.
- CUDA and MUSA compilation/ABI compatibility remain unverified; ROCm does not
  build this operator in the prepared repository.

