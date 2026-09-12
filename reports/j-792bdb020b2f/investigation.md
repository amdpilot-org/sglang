# GLM-5.3 vision processor correction review

Upstream issue: https://github.com/sgl-project/sglang/issues/38821

Mirror issue: https://github.com/amdpilot-org/sglang/issues/925

Candidate parent: https://github.com/amdpilot-org/sglang/pull/808 at
`436f0c1f6d0b64fa2947441bdd0add9611b4778b`

Independent review parent: https://github.com/amdpilot-org/sglang/pull/893

## Finding and correction

The candidate's processor registration and image preprocessing are retained.
On the exact candidate, an independent integration-style regression reproduced
the review counterexample: one processor-expanded image span
`[99, 99, 99, 99]` was forwarded unchanged to `load_mm_data`, although that
stage expects one placeholder per supplied image. The assertion observed the
full prompt `[1, 10, 99, 99, 99, 99, 11, 2]` instead of the required
`[1, 10, 99, 11, 2]`.

The correction collapses consecutive image-token runs only for list-valued
`glm5_next` inputs, before media loading. Existing GLM-4V behavior is unchanged.
Coverage includes sequence boundaries, two separately delimited image spans,
the full pre-load call path, and the non-GLM boundary.

## Evidence and limitations

- Exact-candidate failure: `raw/candidate_failure.txt` (pytest exit 1).
- Corrected regression plus candidate processor tests: `raw/focused_tests.txt`
  (10 passed) and `raw/unit_tests.txt` (9 passed).
- Formatting/lint result: `raw/precommit.txt`.
- Hardware inventory: `raw/gpu_inventory.txt`; one AMD Instinct MI350X/gfx950
  with ROCm 7.2 was visible.

The exact Statue of Liberty JPEG with SHA-256
`ff13fd6f991b37253d3745dc6b9ef8e7a92f17cee7c4c8bc84735000a668fcd7`, the
private `GLM-5.3-Flash-sglang0907-128k-0907t3` weights, eight NVIDIA H20 GPUs,
and the reported deployment were unavailable. Therefore no OpenAI endpoint
semantic inference was run, and this change does not claim to explain the
reported deployment's already-processed but incorrect bird semantics. The
text-only tiny Llama fixture cannot qualify GLM-5.3 vision and was not used as
a substitute.
