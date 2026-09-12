# GLM5 bundled NextN / EAGLE3 guard correction

Candidate PR: https://github.com/amdpilot-org/sglang/pull/3361
Independent review PR: https://github.com/amdpilot-org/sglang/pull/3395
Candidate commit: `ea0cfb324846ce69627ceb74aa0488a9a0502d94`
Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Candidate reproduction

The candidate was applied unchanged before correction. Its own five focused tests
passed. Direct calls to its actual validator then showed:

- exact local path: rejected;
- the same existing directory with a trailing slash: allowed;
- the same existing directory through a literal dot component: allowed;
- a symlink to the same existing directory: allowed.

This reproduces the concrete local-checkpoint bypass. After a bypass, the
candidate does not alter the EAGLE3 capture path, so the candidate's previously
recorded `[N, 3 * hc_mult * H]` versus bundled-head `H` mismatch remains
reachable.

The review's raw `repo@revision` spelling is not a valid Hugging Face repository
identifier (`validate_repo_id` raises `HFValidationError`). Revisions are
represented by the existing `--revision` and
`--speculative-draft-model-revision` arguments. The correction therefore
compares a Hub checkpoint as `(repo_id, revision)`, treating omitted revision
and `main` as equivalent, without inventing an unsupported `@` parser.

## Correction and passing evidence

The candidate's narrow rejection is preserved. Local paths now use canonical
paths plus `os.path.samefile`, covering separators, dot components, symlinks,
and mounted aliases that identify the same filesystem object. Hub identities
compare repository ID and normalized revision. A genuinely different Hub
revision remains eligible as a potentially distinct EAGLE3 draft.

`PYTHONPATH=python /tmp/amdpilot-repo-j-6f8434582863/venv/bin/python -m pytest -q test/registered/unit/spec/test_glm5_next_spec_algorithm.py test/registered/unit/models/test_glm5_next_dflash_capture.py`
passed with 9 tests and 3 subtests. The independent post-correction probe
rejected trailing-slash, dot-component, symlink, and omitted-vs-`main` Hub
aliases, while allowing the same Hub repository at two distinct revisions.

Raw commands and outputs are retained under `reports/j-6f8434582863/raw/`.

## Limitations

GLM-5.3-Flash weights and the reported 4x H20 TP=4 environment were unavailable.
No full server, CUDA fused kernel, semantic correctness, acceptance-length, or
throughput result is claimed. No distinct GLM5 EAGLE3-trained draft was
available, so allowing a truly distinct path/revision preserves existing
support but does not claim architecture compatibility or successful execution.
The original acceptance approximately 1.0 report remains unresolved.
