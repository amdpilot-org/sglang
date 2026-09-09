#!/usr/bin/env bash
set -euo pipefail

CONTROL=/cache/j-2dd26df9bdab/control.git
CANDIDATE=/cache/j-2dd26df9bdab/candidate.git
EXPECTED_COMMIT=484c2286c993d36e862343c390a77439a003d244
HEADER=python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/fp8_utils.cuh
EXPECTED_CONTROL_HEADER=6b686494e4ee7ac6f12f972e1033837ff020d1629e833e06f57f304ed0c5f022
EXPECTED_CANDIDATE_HEADER=16ee2f59330081e0015390d4f895cbd7aa093b55ace86723a668699b130ff375

test "$(git -C "$CONTROL" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git -C "$CANDIDATE" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git -C "$CONTROL" status --porcelain | wc -l)" = 0
test "$(git -C "$CANDIDATE" diff --name-only)" = "$HEADER"
test "$(git -C "$CANDIDATE" status --porcelain | wc -l)" = 1
test "$(sha256sum "$CONTROL/$HEADER" | cut -d' ' -f1)" = "$EXPECTED_CONTROL_HEADER"
test "$(sha256sum "$CANDIDATE/$HEADER" | cut -d' ' -f1)" = "$EXPECTED_CANDIDATE_HEADER"

printf 'control_commit=%s\n' "$(git -C "$CONTROL" rev-parse HEAD)"
printf 'control_status_clean=%s\n' "$(git -C "$CONTROL" status --porcelain | wc -l)"
printf 'control_header_sha256=%s\n' "$(sha256sum "$CONTROL/$HEADER" | cut -d' ' -f1)"
printf 'candidate_commit=%s\n' "$(git -C "$CANDIDATE" rev-parse HEAD)"
printf 'candidate_changed_files=%s\n' "$(git -C "$CANDIDATE" diff --name-only | tr '\n' ' ')"
printf 'candidate_header_sha256=%s\n' "$(sha256sum "$CANDIDATE/$HEADER" | cut -d' ' -f1)"
