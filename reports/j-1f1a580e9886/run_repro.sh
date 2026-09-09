#!/usr/bin/env bash
set -u

repo_root=${1:?repository root}
log_root=${2:?log root}
cache_root=${3:?Triton cache root}
mkdir -p "$log_root" "$cache_root"

export PYTHONPATH="$repo_root/python${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="$cache_root"
export TRITON_ALWAYS_COMPILE=1

for helper in mul_rn_f32 div_rn_f32 rsqrt_approx_f32 cuda_rsqrtf _rcp4; do
  log="$log_root/$helper.log"
  echo "=== $helper ===" | tee "$log"
  timeout 120 /opt/venv/bin/python "$repo_root/reports/j-1f1a580e9886/repro_ptx.py" "$helper" >>"$log" 2>&1
  status=$?
  echo "exit_status=$status" | tee -a "$log"
  cat "$log"
done
