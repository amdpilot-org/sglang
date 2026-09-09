#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include "/tmp/sglang-j88/candidate/python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh"
namespace sglang {
TVM_FFI_DLL_EXPORT_TYPED_FUNC(forward, (FusedNormRopeKernel<bf16_t, 512, 64, 2, false, 0, false>::forward));
}  // namespace sglang
