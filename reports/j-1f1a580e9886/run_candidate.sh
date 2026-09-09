#!/usr/bin/env bash
set -u

harness_root=${1:?harness repository root}
candidate_root=${2:?candidate repository root}
label=${3:?candidate label}
log_root=${4:?log root}
cache_root=${5:?Triton cache root}
mkdir -p "$log_root" "$cache_root"

export PYTHONPATH="$candidate_root/python${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="$cache_root"
export TRITON_ALWAYS_COMPILE=1

for mode in semantics round; do
  log="$log_root/$mode.log"
  echo "=== $label $mode ===" | tee "$log"
  timeout 120 /opt/venv/bin/python "$harness_root/reports/j-1f1a580e9886/repro_norm.py" "$mode" >>"$log" 2>&1
  status=$?
  echo "exit_status=$status" | tee -a "$log"
  cat "$log"
done

for helper in mul_rn_f32 div_rn_f32 rsqrt_approx_f32 cuda_rsqrtf _rcp4; do
  log="$log_root/ptx-$helper.log"
  echo "=== $label ptx $helper ===" | tee "$log"
  timeout 120 /opt/venv/bin/python "$harness_root/reports/j-1f1a580e9886/repro_ptx.py" "$helper" >>"$log" 2>&1
  status=$?
  echo "exit_status=$status" | tee -a "$log"
  cat "$log"
done
