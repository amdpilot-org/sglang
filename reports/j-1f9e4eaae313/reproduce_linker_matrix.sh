#!/usr/bin/env bash
set -euo pipefail

runtime_dir=${1:?usage: reproduce_linker_matrix.sh RUNTIME_DIR}
fixture_dir="$runtime_dir/linker-fixture"
rm -rf "$fixture_dir"
mkdir -p "$fixture_dir/cuda/lib64/stubs" "$fixture_dir/system-driver"

printf 'int fixture(void) { return 0; }\n' > "$fixture_dir/fixture.c"
cc -fPIC -c "$fixture_dir/fixture.c" -o "$fixture_dir/fixture.o"

# FlashInfer 0.6.17 and 0.6.18 both put these paths before -lcuda.
flashinfer_link() {
  c++ -shared "$fixture_dir/fixture.o" \
    -L"$fixture_dir/cuda/lib64" \
    -L"$fixture_dir/cuda/lib64/stubs" \
    "$@" -lcuda -o "$fixture_dir/fixture.so"
}

expect_failure() {
  local name=$1
  shift
  rm -f "$fixture_dir/fixture.so"
  if "$@" >"$fixture_dir/$name.stdout" 2>"$fixture_dir/$name.stderr"; then
    echo "FAIL $name: unexpectedly linked"
    return 1
  fi
  grep -E 'cannot find -lcuda|library not found for -lcuda' "$fixture_dir/$name.stderr"
  echo "PASS $name: missing unversioned libcuda fails as expected"
}

expect_success() {
  local name=$1
  shift
  rm -f "$fixture_dir/fixture.so"
  "$@" >"$fixture_dir/$name.stdout" 2>"$fixture_dir/$name.stderr"
  test -f "$fixture_dir/fixture.so"
  echo "PASS $name: linked"
}

expect_failure missing_everywhere flashinfer_link

# A driver installation commonly exposes libcuda.so.1 for runtime loading.
# The linker's -lcuda lookup specifically requires an unversioned libcuda.so.
cc -shared "$fixture_dir/fixture.o" -o "$fixture_dir/system-driver/libcuda.so.1"
expect_failure versioned_runtime_only flashinfer_link -L"$fixture_dir/system-driver"

# A complete toolkit supplies this unversioned linker stub.
cc -shared "$fixture_dir/fixture.o" -o "$fixture_dir/cuda/lib64/stubs/libcuda.so"
expect_success toolkit_stub flashinfer_link
rm "$fixture_dir/cuda/lib64/stubs/libcuda.so"

# An unversioned driver library in a system directory also works when that
# directory is explicitly supplied (e.g. FLASHINFER_EXTRA_LDFLAGS=-L...).
ln -s libcuda.so.1 "$fixture_dir/system-driver/libcuda.so"
expect_success explicit_driver_search flashinfer_link -L"$fixture_dir/system-driver"

echo "All linker boundary cases passed."
