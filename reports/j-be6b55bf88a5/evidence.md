# Independent review evidence

Reviewed exact candidate `87d106c558fb4472f0821013762ee19d6189b765`
from https://github.com/amdpilot-org/sglang/pull/1240 against recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365` and the original issue contract.
The candidate was not modified.

## Verdict

Recommendation: `request_changes`. The candidate is a valid partial fix, not a
full fix for the original open issue. It repairs all 13 concrete sites named by
the preceding review, but leaves independently demonstrable empty-prefix paths.

## Environment and source selection

The prepared interpreter was
`/tmp/amdpilot-repo-j-be6b55bf88a5/venv/bin/python`. At the candidate it imported
`sglang` from `/job/repo/python/sglang/__init__.py`, not an installed wheel.
PyTorch was `2.11.0+rocm7.2`; ROCm was `7.2`; one AMD Instinct MI350X was visible.

The base-to-candidate file list contains no C, C++, CUDA, HIP, Cython, or header
source. Consequently there was no changed native artifact to rebuild.

## Failing before

Running the original AST contract against the actual base model tree exited 1
and reported `missing_prefix_sites=53`. The list included the reviewed
Persimmon, Voxtral, Falcon-H1, Hunyuan, and GPT-J sites as well as the original
Bailing example.

## Candidate regression

At the exact candidate:

```text
40 passed, 18 warnings, 42 subtests passed in 28.90s
```

The command was:

```bash
/tmp/amdpilot-repo-j-be6b55bf88a5/venv/bin/python -m pytest -q \
  test/registered/unit/models/test_reviewed_quantization_prefixes.py \
  test/registered/unit/models/test_phi_quantization_prefix.py \
  test/registered/unit/layers/quantization/test_compressed_tensors_lm_head.py
```

The candidate's retained 13-site checker also passed:

```text
qualified_review_counterexamples=13/13
```

These results establish that the candidate preserves and extends the earlier
valid corrections. They do not establish the original issue's universal
contract.

## Independent adversarial result

The original scan still exits 1 at the candidate:

```text
missing_prefix_sites=25
```

Seven sites explicitly use `quant_config=None` and are syntactic false
positives. The other 18 accept a live quantization configuration. An independent
constructor probe replaced only the expensive Solar model body and recorded
arguments delivered to `ParallelLMHead`:

```text
quant_config_identity= True
prefix_present= False
prefix_value= <constructor default: empty string>
AssertionError: {'org_num_embeddings': 32, 'padding_size': 64,
                 'quant_config': <object object at ...>}
```

The probe constructed `SolarForCausalLM` with `prefix="nested"`; the contractually
correct head prefix is `nested.lm_head`. This reproduces the original failure
mechanism on the exact candidate without weights or GPU execution.

The remaining live-config syntactic sites are enumerated in `result.json`.
They span nine model files and require architecture-specific validation before
source correction. Their existence, plus the executed Solar counterexample,
means the candidate does not fully resolve the original issue.

## Limitations

No affected-model weights were available. No full-model, semantic, serving, or
distributed workload was claimed. GPU execution was unnecessary for the
constructor/configuration defect, and the available ROCm MI350X differs from
the CUDA RTX 4090 in the issue snapshot. This review adds only evidence and a
verdict; it does not duplicate or amend the candidate patch.
