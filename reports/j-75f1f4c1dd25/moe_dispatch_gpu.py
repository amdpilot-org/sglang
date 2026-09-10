import contextlib
import json
import os
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import torch
import sglang
from sglang.srt.server_args import ServerArgs
from sglang.srt.runtime_context import publish
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.moe.utils import get_moe_runner_backend
from sglang.benchmark import one_batch


ROOT = os.environ["SGLANG_TEST_ROOT"]
assert os.path.dirname(sglang.__file__) == os.path.join(ROOT, "python", "sglang")


def patch_load_model():
    class FakeModelConfig:
        @staticmethod
        def from_server_args(server_args):
            return object()

    class FakeModelRunner:
        max_total_num_tokens = 1

        def __init__(self, **kwargs):
            self.backend_at_construction = get_moe_runner_backend().value

        def alloc_memory_pool(self):
            pass

        def init_attention_backends(self):
            pass

        def init_cuda_graphs(self):
            pass

    one_batch.suppress_other_loggers = lambda: None
    one_batch.ModelConfig = FakeModelConfig
    one_batch.compute_dp_attention_world_info = lambda *args: (0, 1, 0, 1)
    one_batch.ParallelState = lambda **kwargs: object()
    one_batch.use_mlx = lambda: False
    one_batch.get_tokenizer = lambda *args, **kwargs: "tokenizer"
    one_batch.ModelRunner = FakeModelRunner


def load_model_backend(requested):
    args = ServerArgs(model_path="dummy", moe_runner_backend=requested)
    publish(args, role="scheduler")
    patch_load_model()
    runner, tokenizer = one_batch.load_model(
        args, SimpleNamespace(nccl_port=12345), gpu_id=0, tp_rank=0
    )
    return runner.torch_runner.backend_at_construction


dispatch = {}
for requested in ["auto", "triton", "aiter", "flashinfer_mxfp4"]:
    dispatch[requested] = load_model_backend(requested)

torch.manual_seed(75)
torch.cuda.set_device(0)
E, T, H, N, K = 8, 64, 128, 64, 2
dtype = torch.bfloat16
device = "cuda"
hidden = torch.randn(T, H, device=device, dtype=dtype)
ids = torch.stack(
    [torch.randperm(E, device=device, dtype=torch.int32)[:K] for _ in range(T)]
)
weights = torch.rand(T, K, device=device, dtype=torch.float32)
weights /= weights.sum(-1, keepdim=True)


def reference_g1u1(w1, w2):
    output = torch.zeros_like(hidden)
    for token in range(T):
        for k in range(K):
            expert = int(ids[token, k])
            gate = hidden[token] @ w1[expert, :N].T
            up = hidden[token] @ w1[expert, N:].T
            output[token] += weights[token, k].to(dtype) * (
                torch.nn.functional.silu(gate) * up @ w2[expert].T
            )
    return output


def reference_g1u0(w1, w2):
    output = torch.zeros_like(hidden)
    for token in range(T):
        for k in range(K):
            expert = int(ids[token, k])
            activation = torch.nn.functional.silu(hidden[token] @ w1[expert].T)
            output[token] += weights[token, k].to(dtype) * (activation @ w2[expert].T)
    return output


def timed(function, warmups=3, iterations=20):
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def profile(function):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        function()
        torch.cuda.synchronize()
    return sorted(
        {
            event.key
            for event in profiler.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
            and ("moe" in event.key.lower() or "gemm" in event.key.lower())
        }
    )


args = ServerArgs(model_path="dummy", moe_runner_backend="triton")
publish(args, role="scheduler")
initialize_moe_config()
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as fused_moe_module
from sglang.srt.layers.moe.topk import StandardTopKOutput

fused_moe_module.get_tp_group = lambda: object()
fused_moe_module.use_symmetric_memory = lambda group, disabled: contextlib.nullcontext()
w1 = torch.randn(E, 2 * N, H, device=device, dtype=dtype) * 0.03
w2 = torch.randn(E, H, N, device=device, dtype=dtype) * 0.03
config = MoeRunnerConfig(
    num_experts=E,
    num_local_experts=E,
    hidden_size=H,
    intermediate_size_per_partition=N,
    top_k=K,
    params_dtype=dtype,
    inplace=False,
    gate_up_interleaved=False,
)
topk = StandardTopKOutput(weights.to(dtype), ids, None)
actual = fused_moe(hidden, w1, w2, topk, config)
torch.cuda.synchronize()
reference = reference_g1u1(w1, w2)
difference = (actual.float() - reference.float()).abs()
relative = difference / reference.float().abs().clamp_min(1e-3)
elapsed = timed(lambda: fused_moe(hidden, w1, w2, topk, config))
kernels = profile(lambda: fused_moe(hidden, w1, w2, topk, config))
triton_result = {
    "requested": "triton",
    "actual_global": get_moe_runner_backend().value,
    "max_abs_diff": float(difference.max()),
    "mean_abs_diff": float(difference.mean()),
    "max_rel_diff": float(relative.max()),
    "elapsed_seconds": elapsed,
    "mean_ms_per_call": elapsed * 50,
    "profiler_kernels": kernels,
}

args = ServerArgs(model_path="dummy", moe_runner_backend="aiter")
publish(args, role="scheduler")
initialize_moe_config()
from aiter import ActivationType
from aiter.fused_moe_bf16_asm import asm_moe
from aiter.ops.shuffle import shuffle_weight

w1 = torch.randn(E, N, H, device=device, dtype=dtype) * 0.03
w2 = torch.randn(E, H, N, device=device, dtype=dtype) * 0.03
w1_shuffled = shuffle_weight(w1)
w2_shuffled = shuffle_weight(w2)
actual = asm_moe(
    hidden, w1_shuffled, w2_shuffled, weights, ids, activation=ActivationType.Silu
)
torch.cuda.synchronize()
reference = reference_g1u0(w1, w2)
difference = (actual.float() - reference.float()).abs()
relative = difference / reference.float().abs().clamp_min(1e-3)
elapsed = timed(
    lambda: asm_moe(
        hidden, w1_shuffled, w2_shuffled, weights, ids, activation=ActivationType.Silu
    )
)
kernels = profile(
    lambda: asm_moe(
        hidden, w1_shuffled, w2_shuffled, weights, ids, activation=ActivationType.Silu
    )
)
aiter_result = {
    "requested": "aiter",
    "actual_global": get_moe_runner_backend().value,
    "max_abs_diff": float(difference.max()),
    "mean_abs_diff": float(difference.mean()),
    "max_rel_diff": float(relative.max()),
    "elapsed_seconds": elapsed,
    "mean_ms_per_call": elapsed * 50,
    "profiler_kernels": kernels,
}

args = ServerArgs(model_path="dummy", moe_runner_backend="flashinfer_mxfp4")
publish(args, role="scheduler")
initialize_moe_config()
flashinfer_global = get_moe_runner_backend().value
try:
    from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

    Mxfp4MoEMethod(prefix="dispatch-test")
    flashinfer_error = None
except Exception as exc:
    flashinfer_error = {"type": type(exc).__name__, "message": str(exc)}

result = {
    "label": "real gfx942 dispatch and numerical-reference investigation",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "source_root": ROOT,
    "sglang_import_path": os.path.dirname(sglang.__file__),
    "git_commit": os.environ.get("SGLANG_TEST_COMMIT", "unknown"),
    "gpu": torch.cuda.get_device_name(0),
    "gpu_arch": torch.cuda.get_device_capability(0),
    "gpu_count": torch.cuda.device_count(),
    "torch_version": torch.__version__,
    "hip_version": torch.version.hip,
    "dispatch_probe": (
        "publish(ServerArgs(...)); call one_batch.load_model with ModelRunner "
        "replaced by a recorder; backend read from runtime global inside "
        "__init__; no CLI parsing used."
    ),
    "load_model_construction_backend": dispatch,
    "numerical_cases": {"triton": triton_result, "aiter": aiter_result},
    "flashinfer_mxfp4": {
        "initialized_global": flashinfer_global,
        "method_initialization_error": flashinfer_error,
    },
    "timing_method": (
        "3 warmups, 20 timed calls, one synchronize after the loop; wall time "
        "around calls"
    ),
    "architecture_limitation": (
        "flashinfer_mxfp4 is NVIDIA SM90/SM100/SM120 only; this MI300X gfx942 "
        "cannot provide a numerical FlashInfer MXFP4 reference."
    ),
}
print(json.dumps(result, indent=2))
