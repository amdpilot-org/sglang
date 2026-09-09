#!/usr/bin/env bash
set -u

repo_root=${1:?repository root}
log_root=${2:?log root}
cache_root=${3:?Triton cache root}
mkdir -p "$log_root" "$cache_root"

export PYTHONPATH="$repo_root/python${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="$cache_root"
export TRITON_ALWAYS_COMPILE=1

for mode in direct-ln direct-qk; do
  log="$log_root/$mode.log"
  echo "=== $mode ===" | tee "$log"
  timeout 120 /opt/venv/bin/python "$repo_root/reports/j-1f1a580e9886/repro_norm.py" "$mode" >>"$log" 2>&1
  status=$?
  echo "exit_status=$status" | tee -a "$log"
  cat "$log"
done

for mode in semantics round; do
  log="$log_root/$mode.log"
  echo "=== $mode ===" | tee "$log"
  timeout 120 /opt/venv/bin/python "$repo_root/reports/j-1f1a580e9886/repro_norm.py" "$mode" >>"$log" 2>&1
  status=$?
  echo "exit_status=$status" | tee -a "$log"
  cat "$log"
done
