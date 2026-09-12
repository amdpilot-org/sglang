# Qwen3.5 MTP compressed-tensors investigation

The pinned source at `358c163250ad3b1f62939b01ce1314a0a31a0365`
contained the earlier Quark normalization but no equivalent handling for
`CompressedTensorsConfig.ignore`. The concrete config implementation stores its
checkpoint exclusions in `self.ignore` and reports the canonical name
`compressed_tensors`.

The added regression uses the issue's representative unfused names and failed
before the production change because `_mtp_quant_config()` returned that config
unchanged. After the narrow branch was added, the same test returns `None`, so
the MTP constructor and its shared-expert fusion gate consistently build the
MTP module unquantized. Independent cases show that unrelated target-model
entries, an empty or missing list, and non-string values do not disable MTP
quantization.

Raw JUnit evidence is retained beside this report. No checkpoint weights were
available, so this work does not claim a full serving-path, acceptance-rate,
throughput, semantic, TP=2, or NVIDIA reproduction. The assigned GPU was an AMD
Instinct MI355X (`gfx950`), observed with `rocm-smi`; GPU execution was not
needed for this pure construction-policy correction.
