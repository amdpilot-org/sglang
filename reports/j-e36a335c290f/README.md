# Independent review of PR 2846

- Upstream issue: https://github.com/sgl-project/sglang/issues/31248
- Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2785
- Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2882
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Candidate: `b1b2f71e8032aff28d638ef356fb62ee61f249c1`

## Recommendation

`reject`. The candidate accurately states in its own report that it does not
implement `CompressedTensorsW4A16Sparse24`. Its source changes only improve two
exception messages. Its added tests cover the current public checkpoint config
and error classification, not execution of the requested W4A16 2:4 scheme.

On the recorded base, a compressed-tensors configuration with INT4 groupwise
weights and `sparse-24-bitmask`/`2:4` metadata reproduced the original exception:

```text
ImportError: Other method (CompressedTensorsW4A16Sparse24) is not supported now
```

At the exact candidate commit, the same configuration still raises
`ImportError`; only its text changes. An adversarial representation with the
per-group format `pack-quantized` and the same 2:4 sparsity metadata returns
`CompressedTensorsWNA16`, a dense weight-only scheme, rather than a sparse
scheme. An unquantized 2:4 configuration reaches the candidate's other changed
branch, which explicitly says that 2:4 is unsupported.

## Environment and evidence

The pinned interpreter was
`/tmp/amdpilot-repo-j-e36a335c290f/venv/bin/python`. Both `sglang` and the
reviewed compressed-tensors module imported from `/job/repo/python`, proving
that tests exercised the checked-out source. The assigned accelerator was an
AMD Instinct MI355X (`gfx950`) with ROCm 7.2 and Torch 2.11.0+rocm7.2. The
requested sparse path is NVIDIA-specific, so no qualifying GPU numerical test
or compiler/ISA validation was possible. The candidate changes no native code;
there was therefore no native component to rebuild.

Raw evidence was retained outside the checkout under
`/job/review-evidence/`, including the base failure, candidate import paths,
the candidate test log, adversarial results, GPU inventory, issue snapshots,
and the exact candidate diff at `/job/candidate.patch`.

The 26B Gemma model weights were not available and were not served. This does
not weaken the rejection: the candidate contains neither a sparse scheme nor a
kernel and deterministically rejects the requested configuration before model
execution. The tiny Llama fixture is not architecturally relevant to Gemma 4,
NVFP4/W4A16, or 2:4 sparsity and was not used as substitute evidence.
