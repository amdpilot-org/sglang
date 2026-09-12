# Independent review of PR 2187

Upstream issue: https://github.com/sgl-project/sglang/issues/32169

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2116

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2223

Candidate reviewed exactly at `7b1b516e688f0832e363c86dc34eb8f29ed54528`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

## Verdict

Recommendation: **accept**, with `fully_resolves_original: false` because the
prepared environment cannot complete the reported real-model startup.

The base independently reproduced the issue: Hugging Face resolved
`OpenGVLab/InternVL2_5-2B` to a bare `CLIPImageProcessor`, and SGLang raised
the reported unguarded `.tokenizer` `AttributeError`. At the candidate commit,
the same real-model run bypassed that exception and invoked SGLang's separate
tokenizer loader. It then failed later because SentencePiece 0.2.2 rejected a
literal-NUL vocabulary piece. Thus the candidate is verified for the exact
processor/tokenizer-resolution defect, but full server readiness and inference
remain unverified rather than being counted as solved.

The candidate's complete targeted unit file passed: 69 tests and 2 subtests.
Independent adversarial checks passed using the real downloaded image
processor with a controlled tokenizer, a distinct model/tokenizer repository
and tokenizer revision, a combined processor, and a processor that is itself a
tokenizer. Imports resolved to `/job/repo/python`, so validation exercised the
checked-out candidate rather than an installed SGLang wheel.

## Scope and environment

This is a Python-only change. No C++/HIP/native source changed, so no native
rebuild was applicable. The assigned device is one AMD Instinct MI350X
(`gfx950`) with ROCm 7.2 and Torch 2.11.0+rocm7.2, but the reported defect is
device-independent and GPU execution was neither necessary nor performed.
No model weights were downloaded, and this review does not claim model
inference, semantic accuracy, full HTTP serving startup, or distributed
behavior.

Raw commands and outputs are under `raw/`.
