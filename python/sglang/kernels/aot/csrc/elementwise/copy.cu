#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <algorithm>

template <int N>
struct InputArray {
  int values[N];
};

// Keep each fallback launch's by-value argument at 2 KiB, leaving room under
// CUDA's 4 KiB kernel-parameter limit for the element count and output pointer.
constexpr int kCopyFallbackChunkN = 512;

template <int N>
__global__ void copy_to_gpu_no_ce_kernel(const InputArray<N> input_array, int* output) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    output[idx] = input_array.values[idx];
  }
}

template <int N>
__global__ void copy_to_gpu_no_ce_fallback_kernel(const InputArray<N> input_array, int n, int* output) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    output[idx] = input_array.values[idx];
  }
}

void check_copy_to_gpu_no_ce_inputs(const at::Tensor& input, const at::Tensor& output) {
  TORCH_CHECK(input.dim() == 1, "input must be 1-D");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.dtype() == torch::kInt32, "input dtype must be int32");

  TORCH_CHECK(output.dim() == 1, "output must be 1-D");
  TORCH_CHECK(input.numel() == output.numel(), "input and output must have the same size");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(output.dtype() == torch::kInt32, "output dtype must be int32");

  TORCH_CHECK(input.device().is_cpu(), "input must be a CPU tensor");
  TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");
}

template <int N>
void copy_to_gpu_no_ce_impl(const at::Tensor& input, at::Tensor& output) {
  InputArray<N> input_array;
  const int* input_ptr = input.data_ptr<int>();
  for (int i = 0; i < N; ++i)
    input_array.values[i] = input_ptr[i];

  // may use multi thread blocks if performance bottleneck
  dim3 grid(1);
  dim3 block(static_cast<int>(input.numel()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  copy_to_gpu_no_ce_kernel<<<grid, block, 0, stream>>>(input_array, output.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int CHUNK_N>
void copy_to_gpu_no_ce_fallback_impl(const at::Tensor& input, at::Tensor& output) {
  const int64_t n = input.numel();
  const int* input_ptr = input.data_ptr<int>();
  constexpr int kThreads = 256;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  for (int64_t offset = 0; offset < n; offset += CHUNK_N) {
    const int chunk_size = static_cast<int>(std::min<int64_t>(CHUNK_N, n - offset));
    InputArray<CHUNK_N> input_array{};
    for (int i = 0; i < chunk_size; ++i)
      input_array.values[i] = input_ptr[offset + i];

    dim3 grid((chunk_size + kThreads - 1) / kThreads);
    dim3 block(kThreads);
    copy_to_gpu_no_ce_fallback_kernel<<<grid, block, 0, stream>>>(
        input_array, chunk_size, output.data_ptr<int>() + offset);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void copy_to_gpu_no_ce(const at::Tensor& input, at::Tensor& output) {
  check_copy_to_gpu_no_ce_inputs(input, output);

  const int64_t numel = input.numel();
  TORCH_CHECK(numel > 0, "copy_to_gpu_no_ce does not support empty tensors");

  if (numel == 16) {
    copy_to_gpu_no_ce_impl<16>(input, output);
  } else if (numel == 72) {
    copy_to_gpu_no_ce_impl<72>(input, output);
  } else if (numel == 64) {
    copy_to_gpu_no_ce_impl<64>(input, output);
  } else if (numel == 32) {
    copy_to_gpu_no_ce_impl<32>(input, output);
  } else {
    // Preserve compact launch arguments for common expert counts. Less common
    // sizes use as many bounded, no-copy-engine launches as necessary.
    copy_to_gpu_no_ce_fallback_impl<kCopyFallbackChunkN>(input, output);
  }
}
