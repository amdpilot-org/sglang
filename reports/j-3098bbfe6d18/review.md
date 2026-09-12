# Independent review of amdpilot-org/sglang PR 2163

Reviewed candidate: `4faa68870a503cd20c7b5473f0133fe6c63fc0a2`

Upstream issue: https://github.com/sgl-project/sglang/issues/32290

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2097

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2198

## Verdict

Recommendation: **accept**, as a narrow serving-path fix. The candidate is not a
complete fix for every representation discussed in the report, so
`fully_resolves_original` is false.

The recorded base commit reproduced the exact collision: three candidates became
two and the U+FFFD entry incorrectly retained `-2.5` rather than `-0.5`. At the
candidate commit, Chat Completions and Responses preserve all three ordered list
entries and both duplicate logprobs. The legacy/direct `LogProbs.top_logprobs`
dictionary still contains only two keys, but now keeps the first, higher-ranked
candidate for ranked engine output.

## Evidence

- Prepared checkout initially matched recorded base
  `358c163250ad3b1f62939b01ce1314a0a31a0365` exactly.
- Base reproducer imported
  `/job/repo/python/sglang/srt/entrypoints/openai/utils.py`, printed
  `count=2`, `replacement_logprob=-2.5`, and exited 1 on the original expected
  assertions.
- Candidate source imports were all under `/job/repo/python/sglang/...`; no
  installed SGLang copy was tested.
- Candidate regression: 4 passed.
- Adjacent OpenAI serving suites: 189 passed and 72 subtests passed.
- Independent serialized Chat result and Responses result both contained
  `[('A', -0.1), ('�', -0.5), ('�', -2.5)]`.
- Independent mixed prompt/output case retained three candidates at both
  positions, including duplicate decoded strings.
- `git diff --check` found trailing whitespace in the candidate's archived
  `reports/j-7c3ec2d6e068/pytest.log`; product source and test behavior passed.

Raw command output and fetched issue/PR metadata are retained outside the
revision-switching checkout in `/job/review-evidence-j-3098bbfe6d18/`.

## Remaining counterexamples and limitations

1. The issue's direct utility fixture still observes
   `len(out.top_logprobs[0]) == 2`, not 3, because that public/internal field
   remains `Dict[str, float]`. Legacy Completions therefore cannot expose both
   same-text candidates.
2. Original token bytes are not carried. A byte-fallback candidate decoded as
   U+FFFD still serializes bytes as `[239, 191, 189]`, reconstructed by UTF-8
   encoding the replacement character.
3. `setdefault` preserves the first encounter, relying on the engine's documented
   ranked order. An intentionally unsorted duplicate list `[-9, -1]` retains
   `-9`, not the mathematically greater logprob.
4. The original DeepSeek-V4-Flash-FP8 TP4 workload on four NVIDIA H200 GPUs was
   not reproduced. This environment has one AMD gfx950-class ROCm GPU and no
   specified model weights; that hardware cannot qualify the original workload.
   The defect and candidate change are deterministic CPU response shaping, so no
   GPU execution was needed or claimed.
5. No native source changed. `repository-environment.json` reports `native: null`;
   consequently no native rebuild was applicable.

Related upstream work was inspected. Closed PR 32387 and open PR 33201 only keep
the highest-ranked dictionary survivor; they do not preserve duplicate list
entries. Candidate PR 2163 goes further for Chat Completions and Responses.
