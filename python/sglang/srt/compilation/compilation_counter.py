# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/compilation/compilation_counter.py

import copy
import dataclasses
from contextlib import contextmanager


def _register_tilelang_with_dynamo():
    """Register TileLang's CythonKernelWrapper with torch._dynamo.allow_in_graph.

    This enables Piecewise CUDA Graph (PCG) capture for DeepSeek-V4-Flash on MI300X
    when TileLang kernels are used. CythonKernelWrapper.forward is a method_descriptor,
    so we register the class itself rather than the method.
    """
    try:
        from tilelang.jit.adapter.cython.adapter import CythonKernelWrapper
        import torch._dynamo
        torch._dynamo.allow_in_graph(CythonKernelWrapper)
    except ImportError:
        # TileLang not installed, skip registration
        pass
    except Exception as e:
        # Registration failed, log but don't crash
        import logging
        logging.warning(f"Failed to register TileLang with Dynamo: {e}")


# Register on module import
_register_tilelang_with_dynamo()


@dataclasses.dataclass
class CompilationCounter:
    num_models_seen: int = 0
    num_graphs_seen: int = 0
    # including the splitting ops
    num_piecewise_graphs_seen: int = 0
    # not including the splitting ops
    num_piecewise_capturable_graphs_seen: int = 0
    num_backend_compilations: int = 0
    # Number of gpu_model_runner attempts to trigger CUDAGraphs capture
    num_gpu_runner_capture_triggers: int = 0
    # Number of CUDAGraphs captured
    num_cudagraph_captured: int = 0
    # InductorAdapter.compile calls
    num_inductor_compiles: int = 0
    # EagerAdapter.compile calls
    num_eager_compiles: int = 0
    # The number of time vLLM's compiler cache entry was updated
    num_cache_entries_updated: int = 0
    # The number of standalone_compile compiled artifacts saved
    num_compiled_artifacts_saved: int = 0
    # Number of times a model was loaded with CompilationLevel.DYNAMO_AS_IS
    dynamo_as_is_count: int = 0

    def clone(self) -> "CompilationCounter":
        return copy.deepcopy(self)

    @contextmanager
    def expect(self, **kwargs):
        old = self.clone()
        yield
        for k, v in kwargs.items():
            assert getattr(self, k) - getattr(old, k) == v, (
                f"{k} not as expected, before it is {getattr(old, k)}"
                f", after it is {getattr(self, k)}, "
                f"expected diff is {v}"
            )


compilation_counter = CompilationCounter()
