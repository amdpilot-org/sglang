# Independent review of PR 1995

Upstream issue: https://github.com/sgl-project/sglang/issues/33207

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/1928

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2031

Candidate: https://github.com/amdpilot-org/sglang/pull/1995 at `e0e0ef0817fe8ead8ee19803501c7d7e09c49691`

## Recommendation

**Accept as test-only hardening, not as a candidate-produced fix.** The candidate
changes only a regression test and its investigation report. Its two tests pass
unchanged against the exact parent/base commit
`358c163250ad3b1f62939b01ce1314a0a31a0365`, so there is no failing-before /
passing-after production correction in this PR.

The underlying prepared source already routes the real checkpoint configuration
(`model_type=deepseek_v4`, `architectures=[DeepseekV4ForCausalLM]`) from
`_DeepseekV4ConfigAlias` to SGLang's native
`sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM`. Independently forcing
Hugging Face `AutoModelForCausalLM.from_config` with that alias reproduces the
reported `Unrecognized configuration class` exception, but default SGLang AUTO
dispatch does not take that path.

The candidate's regression correctly protects native AUTO dispatch and the
unknown-architecture fallback boundary. An independent probe used the current
real `DeepSeek-V4-Flash-0731` `config.json`, rather than only the candidate's
minimal synthetic config, and confirmed the same native selection.

## Classification and limits

This is **test-only hardening** around a solution already present in the recorded
base, not a full original-issue fix authored by the candidate. The configuration
dispatch symptom is verified, but `fully_resolves_original` is false because the
reported end-to-end serving contract was not reproduced: the assigned system has
one AMD gfx950 GPU with ROCm 7.2, not two NVIDIA Blackwell GPUs with CUDA 13.3;
the model weights were not downloaded; and TP=2, FlashInfer MXFP4, full startup,
generation accuracy, and NVIDIA compiler/ISA behavior remain unverified.

No production or native sources differ in the candidate, so no native rebuild
was applicable. Import-path evidence confirms SGLang loaded from
`/job/repo/python/sglang` while Transformers 5.12.1 loaded from the prepared
environment.

## Reproduction

```bash
/tmp/amdpilot-repo-j-77e48d67966a/venv/bin/python -m pytest -q \
  test/registered/unit/model_loader/test_deepseek_v4_dispatch.py
```

At the candidate: `2 passed`. The same file copied outside the checkout and run
against its exact parent/base also reports `2 passed`, establishing that this PR
adds coverage without changing behavior.

Raw outputs are retained beside this report in `evidence/`.
