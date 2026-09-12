# Investigation evidence

## Finding

The defect reported in the source issue is already fixed in the prepared base
commit `358c163250ad3b1f62939b01ce1314a0a31a0365`. Upstream PR #33140 added an
`official` DeepSeek-V4 reasoning-effort profile and checkpoint-based profile
detection. No product source change was made in this investigation.

Upstream fix: https://github.com/sgl-project/sglang/pull/33140

## Pre-fix behavior

The first parent of upstream's fix merge is
`198a3bc29bbf2ed169d50f5b7ad35c74262ff10f`. Its
`encoding_dsv4.py` has only `REASONING_EFFORT_MAX` and only injects that prompt
when `reasoning_effort == "max"`; it contains no `Beyond maximum` prompt. This
matches the reported shifted mapping.

Command:

```bash
git show 198a3bc29bbf2ed169d50f5b7ad35c74262ff10f:python/sglang/srt/entrypoints/openai/encoding_dsv4.py \
  | rg -n "REASONING_EFFORT|Beyond maximum|reasoning_effort =="
```

Observed:

```text
63:REASONING_EFFORT_MAX = (
299:    if index == 0 and thinking_mode == "thinking" and reasoning_effort == "max":
300:        prompt += REASONING_EFFORT_MAX
```

## Current implementation versus checkpoint reference

The reference encoder was downloaded without model weights to the private
runtime cache at:

```text
/tmp/amdpilot-repo-j-7b8a88b7e30d/hf-cache/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062/encoding/encoding_dsv4.py
```

An AST-based comparison of the checkpoint's `REASONING_EFFORT_PROMPTS` with
SGLang's `REASONING_EFFORT_PROFILES["official"]` produced:

```text
reference_revision=7872f01b1d1fe23eabc4c98b48bffcef5a386062
detected_profile=official
low: equal=True bytes=0 sha256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
high: equal=True bytes=476 sha256=f7a24f3b3050d4a83781e9b755b316aae2ec1d9dc87562541d90bcb04d7350f7
max: equal=True bytes=528 sha256=53fb31b8392ec5b4a926481943efc67636748f137c1a49d493a7af4d4fa55d2c
encoded_low: bytes=76 has_absolute=False has_beyond=False
encoded_high: bytes=552 has_absolute=True has_beyond=False
encoded_max: bytes=604 has_absolute=False has_beyond=True
default_equals_low=True
```

This independently covers the three requested levels and the omitted/default
boundary. The repository regression also covers preview compatibility,
checkpoint detection, explicit override, and invalid-profile rejection.

## Focused regression

Command:

```bash
/tmp/amdpilot-repo-j-7b8a88b7e30d/venv/bin/python -m pytest -q \
  test/registered/unit/entrypoints/openai/test_serving_chat.py \
  -k 'dsv4_reasoning_effort'
```

Result: `4 passed, 131 deselected` (exit code 0).

## Hardware and limitations

The assigned device was visible as one AMD Instinct MI350X (`gfx950`, 252 GiB
VRAM). GPU execution was not used because the defect and existing regression
are deterministic prompt construction and profile-selection logic. The
DeepSeek-V4-Flash-0731 weights were not available/downloaded, so no full-model,
semantic-accuracy, H200, tensor-parallel, or multi-node reproduction was
performed or claimed.
