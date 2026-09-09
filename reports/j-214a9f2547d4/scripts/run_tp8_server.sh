#!/usr/bin/env bash
set -euo pipefail

CONTROL=/tmp/sglang-pr57-control
REQUIRED_CONTROL=484c2286c993d36e862343c390a77439a003d244

test "$(git -C "$CONTROL" rev-parse HEAD)" = "$REQUIRED_CONTROL"
git -C "$CONTROL" diff --exit-code "$REQUIRED_CONTROL" -- python/sglang

export PYTHONPATH="$CONTROL/python"
python - <<'PY'
import sglang
import sglang.srt.layers.quantization.fp8

control = "/tmp/sglang-pr57-control/python"
assert sglang.__file__.startswith(control + "/")
assert sglang.srt.layers.quantization.fp8.__file__.startswith(control + "/")
print(f"sglang={sglang.__file__}", flush=True)
print(f"fp8={sglang.srt.layers.quantization.fp8.__file__}", flush=True)
PY

export SGLANG_USE_AITER=1
export SGLANG_DSV4_FP4_DEQUANT=1
export SGLANG_HACK_FLASHMLA_BACKEND=triton

exec python -m sglang.launch_server \
  --host 127.0.0.1 \
  --port 31322 \
  --model-path /models/DeepSeek-V4-Flash-0731 \
  --tp 8 \
  --cuda-graph-max-bs-decode 8
