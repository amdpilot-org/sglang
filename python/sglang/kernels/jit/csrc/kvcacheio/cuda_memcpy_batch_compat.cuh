#pragma once

#include <cstddef>
#include <limits>

namespace sglang {

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080
inline auto call_cuda_memcpy_batch_async(
    void* symbol,
    int runtime_version,
    void** dsts,
    void** srcs,
    size_t* sizes,
    size_t count,
    cudaMemcpyAttributes* attrs,
    size_t* attrs_idxs,
    size_t num_attrs,
    cudaStream_t stream) -> cudaError_t {
  if (runtime_version >= 13000) {
    using FnV13 = cudaError_t (*)(
        void* const*,
        const void* const*,
        const size_t*,
        size_t,
        cudaMemcpyAttributes*,
        size_t*,
        size_t,
        cudaStream_t);
    auto fn = reinterpret_cast<FnV13>(symbol);
    return fn(
        dsts,
        reinterpret_cast<const void* const*>(srcs),
        sizes,
        count,
        attrs,
        attrs_idxs,
        num_attrs,
        stream);
  }

  using FnV12 = cudaError_t (*)(
      void**, void**, size_t*, size_t, cudaMemcpyAttributes*, size_t*, size_t, size_t*, cudaStream_t);
  auto fn = reinterpret_cast<FnV12>(symbol);
  size_t fail_idx = std::numeric_limits<size_t>::max();
  return fn(dsts, srcs, sizes, count, attrs, attrs_idxs, num_attrs, &fail_idx, stream);
}
#endif

}  // namespace sglang
