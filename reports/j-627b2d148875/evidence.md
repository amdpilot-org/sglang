# Evidence

## Failing before the change

At base commit `358c163250ad3b1f62939b01ce1314a0a31a0365`, a constructor probe patched
only the expensive model body and recorded the keyword arguments passed by
`BailingMoELinearForCausalLM` to its `ParallelLMHead`. With outer prefix
`language_model`, the recorded arguments were:

```text
{'params_dtype': torch.float32, 'quant_config': <object>,
 'use_attn_tp_group': False}
```

The assertion expecting `prefix == "language_model.lm_head"` failed because
the keyword was absent. `ParallelLMHead` therefore used its default `""`.

## Passing after the change

```text
$ /tmp/amdpilot-repo-j-627b2d148875/venv/bin/python -m pytest -q \
    test/registered/unit/layers/quantization/test_compressed_tensors_lm_head.py
.......... [100%]
10 passed, 18 warnings, 2 subtests passed in 12.31s
```

The two new subtests cover an empty model prefix (`lm_head`) and a nested model
prefix (`language_model.lm_head`). The same test file independently verifies
that compressed-tensors resolves exact and regex head targets, honors ignored
heads, and leaves unnamed heads unquantized.

## Remaining static audit scope

```text
$ /tmp/amdpilot-repo-j-627b2d148875/venv/bin/python \
    reports/j-627b2d148875/reproduce_missing_quant_prefixes.py
missing_prefix_sites=52
```

The complete list is emitted by the retained script. This PR deliberately
removes the demonstrated Bailing site only; the other 52 sites require
model-specific qualified-name validation and remain outside this candidate.

Compilation of all three changed Python files succeeded. A Ruff check could
not run because the prepared interpreter does not contain the `ruff` module.
No GPU execution was used: the defect and regression concern constructor name
routing before weights or kernels execute.
