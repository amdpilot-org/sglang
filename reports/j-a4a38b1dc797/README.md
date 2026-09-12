# Independent review of PR 3006

Reviewed exact commit `9080a9f9b7d091a3f6ea8c7c1717c753dc592349`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **request changes**. The candidate is a partial fix. It fixes the
specific inherited zero-argument `super()` counterexample from the preceding
review, but it does not fully implement reliable source reload for the original
issue.

## Reproduction summary

The prepared base has no `sglang.srt.dev_reload`; the import fails with
`ModuleNotFoundError`. At the candidate, the six focused tests pass, including
the exact retained-instance/zero-argument-`super()` regression.

Independent cases then showed two remaining false-success behaviors:

1. Removing top-level definitions from edited source leaves them in the module
   dictionary after `importlib.reload`, so deleted functions/classes remain live.
2. Changing `Child(BaseA)` to `Child(BaseB)` does not update the preserved
   `Child.__bases__`; calls continue through `BaseA` while reload reports success.

Exact output:

```text
zero_arg_super base-version-two-child-version-two
deleted_names_present True True
deleted_function_result stale-function
preserved_bases ['BaseA']
changed_inheritance_result A-new-and-longer-child-new-and-longer
```

The expected last line after the source edit begins with `B-new...`, and removed
names should not remain exposed. If those edit categories are intentionally
unsupported, the endpoint should detect and reject them instead of returning a
successful reload.

## Environment and paths

- Python: `/tmp/amdpilot-repo-j-a4a38b1dc797/venv/bin/python`
- Imported SGLang: `/job/repo/python/sglang/__init__.py`
- Candidate reload source: `/job/repo/python/sglang/srt/dev_reload.py`
- Torch: `2.11.0+rocm7.2` from `/opt/venv/lib/python3.12/site-packages/torch`
- HIP: `7.2.26015`; `torch.cuda.is_available()` was true
- Native rebuild: not applicable because the candidate changes no native source
- GPU execution: false; no model/graph numerical claim is made

Actual HTTP serving, graph recapture, weight preservation, and distributed
coordination were not independently executed. The result therefore does not use
an unrelated startup smoke as evidence. The pure-Python failures independently
establish that the complete original reload contract is not resolved.

Raw commands and outputs are retained in `evidence/`.
