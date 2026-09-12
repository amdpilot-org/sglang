# Investigation of stop sequence after Chinese text

Upstream issue: https://github.com/sgl-project/sglang/issues/36698

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1118

At base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the reported scheduler
behavior was not reproduced with the exact public tokenizer from
`deepseek-ai/DeepSeek-V4-Flash-0731`. The real `Req.update_finish_state` path
matched and trimmed `<END>` after both English and Chinese text. It also passed
two independent boundaries: a multibyte emoji plus Chinese prefix, and a stop
string completed one token per scheduler step.

No production source was changed because the evidence does not identify a
defect in the current matcher. Existing source already includes the relevant
speculative-decoding corrections from upstream PRs
https://github.com/sgl-project/sglang/pull/23802 and
https://github.com/sgl-project/sglang/pull/28802, including coverage of all
newly accepted tokens and trimming when a stop string and EOS occur in one
accepted run.

## Reproduction

The tokenizer snapshot was downloaded outside the worktree. Its resolved
revision and the complete probe output are retained in `tokenizer_revision.txt`
and `stop_sequence_probe.log`.

```bash
HF_HOME=/tmp/amdpilot-repo-j-103714608623/hf \
  /tmp/amdpilot-repo-j-103714608623/venv/bin/python \
  reports/j-103714608623/reproduce_stop_sequence.py \
  /tmp/amdpilot-repo-j-103714608623/hf/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/REVISION
```

The existing focused scheduler regression also passes:

```bash
/tmp/amdpilot-repo-j-103714608623/venv/bin/python \
  test/registered/unit/managers/test_stop_str_speculative.py
```

## Limitations

The assigned device is one AMD gfx950 GPU. The report used H20 and a
DeepSeek-V4-Flash-0731 PD-disaggregated deployment; the 285B model and the
required multi-node/multi-GPU topology were not available. No full-model
generation or PD transport claim is made. The probe qualifies the exact
tokenizer plus SGLang scheduler stop/trim logic only, not model semantic
accuracy or distributed execution. GPU execution is not relevant to this CPU
control-flow path and was not used as substitute evidence.
