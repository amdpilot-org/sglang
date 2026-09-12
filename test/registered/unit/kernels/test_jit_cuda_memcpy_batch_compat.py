import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


class TestJitCudaMemcpyBatchCompat(unittest.TestCase):
    def test_dispatches_using_runtime_not_compile_time_cuda_version(self):
        repo_root = Path(__file__).resolve().parents[4]
        include_dir = repo_root / "python/sglang/kernels/jit/csrc/kvcacheio"
        source = textwrap.dedent(
            r"""
            #include <cstddef>
            #include <cstdint>
            #define CUDA_VERSION 12080
            using cudaError_t = int;
            using cudaStream_t = void*;
            struct cudaMemcpyAttributes {};
            #include "cuda_memcpy_batch_compat.cuh"

            static int called = 0;
            static cudaError_t v13(void* const*, const void* const*, const size_t*, size_t,
                                   cudaMemcpyAttributes*, size_t*, size_t, cudaStream_t stream) {
              called = stream == reinterpret_cast<void*>(13) ? 13 : -13;
              return 113;
            }
            static cudaError_t v12(void**, void**, size_t*, size_t, cudaMemcpyAttributes*, size_t*, size_t,
                                   size_t* fail_idx, cudaStream_t stream) {
              called = stream == reinterpret_cast<void*>(12) && *fail_idx == SIZE_MAX ? 12 : -12;
              return 112;
            }
            int main() {
              void* ptrs[] = {nullptr};
              size_t sizes[] = {1};
              size_t attrs_idxs[] = {0};
              cudaMemcpyAttributes attrs{};
              auto err = sglang::call_cuda_memcpy_batch_async(
                  reinterpret_cast<void*>(&v13), 13000, ptrs, ptrs, sizes, 1, &attrs, attrs_idxs, 1,
                  reinterpret_cast<void*>(13));
              if (err != 113 || called != 13) return 1;
              err = sglang::call_cuda_memcpy_batch_async(
                  reinterpret_cast<void*>(&v12), 12999, ptrs, ptrs, sizes, 1, &attrs, attrs_idxs, 1,
                  reinterpret_cast<void*>(12));
              return err == 112 && called == 12 ? 0 : 2;
            }
            """
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_path = tmp_path / "dispatch_test.cpp"
            binary_path = tmp_path / "dispatch_test"
            source_path.write_text(source)
            subprocess.run(
                [
                    "c++",
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(include_dir),
                    str(source_path),
                    "-o",
                    str(binary_path),
                ],
                check=True,
            )
            subprocess.run([str(binary_path)], check=True)


if __name__ == "__main__":
    unittest.main()
