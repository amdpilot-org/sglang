# Independent review of candidate PR 1326

Candidate: https://github.com/amdpilot-org/sglang/pull/1326  
Exact commit: `0f6ace775da584a0bfe330ff908470574b14fb7c`  
Upstream issue: https://github.com/sgl-project/sglang/issues/35912  
Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/1275  
Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3215

## Recommendation

`request_changes`.

The candidate correctly changes the one remaining unsafe cookbook sentence on
the prepared base, and its checker rejects that exact one-line regression.
Most originally reported pages were already corrected on the recorded base by
an earlier related change. The candidate is therefore a narrow remaining docs
fix plus incomplete test hardening, not an independent fix for uv's resolver
behavior.

The prevention claim is too broad. The actual checker returned success after
adding each of these unsafe recommendations:

```bash
uv pip install \
  sglang

uv pip install SGLang
```

It also does not inspect `docs/docs`, so an unsafe install-guide or quickstart
regression is unguarded. Package names are case-insensitive, and multiline shell
commands are normal MDX syntax. The current tree after the candidate has no
plain one-line PyPI SGLang install found by a repository search, so these are
hardening counterexamples rather than evidence that its edited sentence is
wrong.

## Independent issue reproduction

Using fresh private virtual environments outside the checkout:

- uv 0.11.16 resolved `sglang<0.5.18` to `sglang==0.5.9` (exit 0).
- uv 0.12.0 resolved the same input to `sglang==0.5.17` and
  `flash-attn-4==4.0.0b19` (exit 0).
- uv 0.11.16 with `--prerelease=allow` resolved to `sglang==0.5.17`
  and `flash-attn-4==4.0.0b19` (exit 0).
- uv 0.11.16 with pinned `sglang==0.5.10.post1` failed (exit 1), stating
  that `flash-attn-4` prereleases were not enabled and suggesting
  `--prerelease=allow`.

This reproduces the original package-resolution failure without relying on the
candidate's report.

## Candidate regression and adversarial checks

- Exact candidate: `node docs/scripts/check_cookbook_configs.mjs` passed.
- With the edited sentence temporarily restored to `uv pip install sglang`,
  the exact candidate checker failed on line 34 as intended.
- With temporary cookbook MDX files containing the multiline and `SGLang`
  forms above, and a temporary `docs/docs` page containing the exact plain
  command, the checker incorrectly passed. All temporary files were removed
  before returning to the review branch.
- `git diff --check` for base-to-candidate passed.

## Source, native, GPU, and architecture evidence

The candidate changes only MDX, a Node documentation checker, and its own
report artifacts. It changes no Python package, C++, HIP, FlyDSL, or native
extension, so a native rebuild is not applicable. With the prescribed
interpreter and `PYTHONPATH=/job/repo/python`, `sglang` imported from
`/job/repo/python/sglang/__init__.py`; Torch imported from
`/opt/venv/lib/python3.12/site-packages/torch/__init__.py` and reported
`2.11.0+rocm7.2`, HIP 7.2.26015, one visible GPU.

No GPU execution was needed to establish this deterministic resolver/docs
contract. The reported Qwen3.8-27B-FP8 TP=2 semantic consequence was not and
cannot be qualified here: only one gfx950 GPU is assigned, and the original
weights/two-GPU topology were not available. A tiny model or one-GPU smoke
would not prove that consequence.

