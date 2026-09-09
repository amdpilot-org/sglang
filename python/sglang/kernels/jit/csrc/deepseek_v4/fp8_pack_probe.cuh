#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

namespace sglang {

__global__ void fp8_pack_probe_kernel(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ metadata,
    int64_t pair_count) {
  const int64_t pair_index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (pair_index < pair_count) {
    const auto packed = static_cast<uint16_t>(
        deepseek_v4::fp8::pack_fp8(input[2 * pair_index], input[2 * pair_index + 1]));
    output[2 * pair_index] = static_cast<uint8_t>(packed & 0xFF);
    output[2 * pair_index + 1] = static_cast<uint8_t>(packed >> 8);
  }
  if (pair_index == 0) {
    metadata[0] = kFP8E4M3Max;
#if HIP_FP8_TYPE_FNUZ
    metadata[1] = 1.0f;
#else
    metadata[1] = 0.0f;
#endif
#ifdef SGL_ROCM_FP8_HW_CVT
    metadata[2] = 1.0f;
#else
    metadata[2] = 0.0f;
#endif
    metadata[3] = 0.0f;
  }
}

struct fp8_pack_probe {
  static void run(
      tvm::ffi::TensorView input,
      tvm::ffi::TensorView output,
      tvm::ffi::TensorView metadata) {
    const int64_t element_count = input.size(0);
    if (element_count == 0) return;
    const int64_t pair_count = element_count / 2;
    constexpr int64_t block_size = 128;
    const int64_t grid_size = (pair_count + block_size - 1) / block_size;
    host::LaunchKernel(
        dim3(static_cast<uint32_t>(grid_size)),
        dim3(static_cast<uint32_t>(block_size)),
        input.device())(
        fp8_pack_probe_kernel,
        static_cast<const float*>(input.data_ptr()),
        static_cast<uint8_t*>(output.data_ptr()),
        static_cast<float*>(metadata.data_ptr()),
        pair_count);
  }
};

}  // namespace sglang
