# Investigation evidence

- Upstream issue and mirror issue were open and had no comments when inspected on 2026-09-12.
- A GitHub PR search found no pull request referencing issue 36427. Current upstream `main` retained the same installation page as the prepared base.
- Before the change, direct assertions for a package-version explanation, graph fallback, and ModelScope instructions all evaluated to `False` and raised `AssertionError`.
- `python/pyproject_npu.toml` includes `modelscope`; `python/sglang/srt/arg_groups/serving_hook.py` gates ModelScope resolution on `SGLANG_USE_MODELSCOPE`; and the focused model-path suite passed 12 tests.
- NPU graph-runner implementations are present, so the documentation retains graph capture as the default and describes eager execution only as a failure fallback.
- `ls -l /dev/davinci*` failed with “No such file or directory”; Ascend runtime reproduction was unavailable.

## Raw test output

```text
...                                                                      [100%]
3 passed, 1 warning in 0.29s
```

```text
............                                                             [100%]
12 passed, 17 warnings in 19.08s
```
