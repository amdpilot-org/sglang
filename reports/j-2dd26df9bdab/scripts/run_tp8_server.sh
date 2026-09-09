#!/usr/bin/env bash
set -euo pipefail

ARM=${1:?control or candidate}
case "$ARM" in
  control) SOURCE=/cache/j-2dd26df9bdab/control.git ;;
  candidate) SOURCE=/cache/j-2dd26df9bdab/candidate.git ;;
  *) echo "invalid arm: $ARM" >&2; exit 2 ;;
esac

CACHE_ROOT=/job/caches/$ARM
LOG_DIR=/job/logs/$ARM
EVIDENCE_DIR=/job/evidence/$ARM
mkdir -p "$CACHE_ROOT"/{sglang,triton,torchinductor,deep_gemm,cute_aot,flydsl,ccache,numba,xdg} "$LOG_DIR" "$EVIDENCE_DIR"

if find "$CACHE_ROOT" -type f -print -quit | grep -q .; then
  echo "private cache root is not initially empty: $CACHE_ROOT" >&2
  exit 3
fi

/cache/j-2dd26df9bdab/delivery.git/reports/j-2dd26df9bdab/scripts/assert_identities.sh \
  > "$EVIDENCE_DIR/prelaunch-identity.txt"

export SGLANG_USE_AITER=1
export SGLANG_DSV4_FP4_DEQUANT=1
export SGLANG_HACK_FLASHMLA_BACKEND=triton
export PYTHONPATH="$SOURCE/python"
export SGLANG_CACHE_DIR="$CACHE_ROOT/sglang"
export SGLANG_JIT_CACHE_DIR="$CACHE_ROOT/sglang/jit"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export DG_JIT_CACHE_DIR="$CACHE_ROOT/deep_gemm"
export SGLANG_DG_CACHE_DIR="$CACHE_ROOT/deep_gemm"
export SGLANG_CUTE_AOT_CACHE_DIR="$CACHE_ROOT/cute_aot"
export FLYDSL_RUNTIME_CACHE_DIR="$CACHE_ROOT/flydsl"
export CCACHE_DIR="$CACHE_ROOT/ccache"
export NUMBA_CACHE_DIR="$CACHE_ROOT/numba"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

python - "$ARM" "$SOURCE" "$CACHE_ROOT" > "$EVIDENCE_DIR/prelaunch-imports.txt" <<'PY'
import hashlib
import json
import os
import platform
import sys

arm, source, cache_root = sys.argv[1:]
header = os.path.join(source, "python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/fp8_utils.cuh")
import sglang
import sglang.srt.layers.quantization.fp8 as fp8
from sglang.srt.entrypoints.openai import encoding_dsv4
import torch
import triton
import transformers

record = {
    "arm": arm,
    "source": source,
    "cache_root": cache_root,
    "python": sys.executable,
    "python_version": platform.python_version(),
    "sglang_file": sglang.__file__,
    "quantization_fp8_file": fp8.__file__,
    "encoding_dsv4_file": encoding_dsv4.__file__,
    "header_sha256": hashlib.sha256(open(header, "rb").read()).hexdigest(),
    "torch_version": torch.__version__,
    "hip_version": torch.version.hip,
    "triton_version": triton.__version__,
    "transformers_version": transformers.__version__,
    "environment": {
        key: os.environ.get(key)
        for key in [
            "SGLANG_USE_AITER",
            "SGLANG_DSV4_FP4_DEQUANT",
            "SGLANG_HACK_FLASHMLA_BACKEND",
            "PYTHONPATH",
            "SGLANG_CACHE_DIR",
            "SGLANG_JIT_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "DG_JIT_CACHE_DIR",
            "SGLANG_DG_CACHE_DIR",
            "SGLANG_CUTE_AOT_CACHE_DIR",
            "FLYDSL_RUNTIME_CACHE_DIR",
            "CCACHE_DIR",
            "NUMBA_CACHE_DIR",
            "XDG_CACHE_HOME",
        ]
    },
}
assert record["sglang_file"].startswith(source + "/")
assert record["quantization_fp8_file"].startswith(source + "/")
assert record["encoding_dsv4_file"].startswith(source + "/")
print(json.dumps(record, indent=2, sort_keys=True))
PY

printf '%s\n' \
  python -m sglang.launch_server \
  --host 127.0.0.1 \
  --port 31322 \
  --model-path /models/DeepSeek-V4-Flash-0731 \
  --tp 8 \
  --cuda-graph-max-bs-decode 8 \
  --random-seed 12345 \
  > "$EVIDENCE_DIR/server-argv.txt"

printf '%s\n' $$ > "$EVIDENCE_DIR/server.pid"
exec python -m sglang.launch_server \
  --host 127.0.0.1 \
  --port 31322 \
  --model-path /models/DeepSeek-V4-Flash-0731 \
  --tp 8 \
  --cuda-graph-max-bs-decode 8 \
  --random-seed 12345
