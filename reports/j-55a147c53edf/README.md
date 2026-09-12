# Request logger truncation correction

Upstream issue: https://github.com/sgl-project/sglang/issues/33164

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2134

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2001 at exact commit `90b5ae30848c43a54f1ad293ab3b4dd2ce4bf478`

Independent review PR: https://github.com/amdpilot-org/sglang/pull/2098

The exact candidate was checked out in `/job/repo` before modification. Its four
regressions and recursive list/tuple formatting are valid, but the review's
`max_length=1` counterexamples reproduce: a zero half-length makes `[-0:]`
select the complete tail in string and collection slicing.

The consolidated correction retains the candidate's recursive element
truncation and tuple formatting, and uses an empty same-type slice when the
half-length is zero. The focused suite includes the candidate's ordinary,
nested, boundary, and tuple cases plus JSON, text, and direct-string
`max_length=1` regressions.

No GPU, server, model weights, or native rebuild were used. These helpers are
pure Python serialization code with no hardware-specific execution path.

