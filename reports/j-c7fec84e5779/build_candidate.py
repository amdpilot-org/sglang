import os
import shutil
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "python/sglang/kernels/aot/csrc/speculative/eagle_utils.cu"
BINDING = Path(__file__).resolve().parent / "candidate_binding.cc"
BUILD_DIR = Path(os.environ.get("SGLANG_JOB_CACHE", "/tmp/sglang-cache-j-c7fec84e5779"))


def load_candidate():
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    extension_dir = BUILD_DIR / "extension"
    extension_dir.mkdir(parents=True, exist_ok=True)
    source_copy = BUILD_DIR / "eagle_utils.cu"
    binding_copy = BUILD_DIR / "candidate_binding.cc"
    shutil.copy2(SOURCE, source_copy)
    shutil.copy2(BINDING, binding_copy)
    return load(
        name="eagle_tree_candidate",
        sources=[str(source_copy), str(binding_copy)],
        extra_include_paths=[
            str(REPO / "python/sglang/kernels/aot/include"),
            str(REPO / "python/sglang/kernels/aot/csrc"),
        ],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-DUSE_ROCM",
            "--amdgpu-target=gfx942",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        build_directory=str(extension_dir),
        verbose=True,
    )


if __name__ == "__main__":
    module = load_candidate()
    print(module.__file__)
    print(module.build_tree_kernel_efficient)
