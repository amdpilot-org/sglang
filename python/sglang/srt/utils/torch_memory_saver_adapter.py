import logging
import os
import site
from abc import ABC
from contextlib import contextmanager
from pathlib import Path

import torch

from sglang.srt.utils.common import is_xpu

try:
    import torch_memory_saver

    # Intel XPU requires hook_mode="torch" (in-process pluggable allocator);
    # the LD_PRELOAD-based preload mode is CUDA/HIP-only. Set it before the
    # singleton is initialized on first use.
    if is_xpu():
        torch_memory_saver.torch_memory_saver.hook_mode = "torch"

    _memory_saver = torch_memory_saver.torch_memory_saver
    import_error = None
except ImportError as e:
    import_error = e
    pass

logger = logging.getLogger(__name__)

_warned_xpu_cuda_graph = False


def _get_pip_cuda_runtime_library_dir():
    cuda_version = torch.version.cuda
    if not cuda_version:
        return None

    cuda_major = cuda_version.split(".", 1)[0]
    runtime_soname = f"libcudart.so.{cuda_major}"
    for site_packages in site.getsitepackages():
        for relative_dir in (
            Path("nvidia") / f"cu{cuda_major}" / "lib",
            Path("nvidia") / "cuda_runtime" / "lib",
        ):
            candidate = Path(site_packages) / relative_dir
            if (candidate / runtime_soname).is_file():
                return candidate
    return None


class TorchMemorySaverAdapter(ABC):
    @staticmethod
    def create(enable: bool):
        if enable and import_error is not None:
            if is_xpu():
                # XPU ships no prebuilt wheel; it is built from source against the
                # local oneAPI + torch-XPU runtime. TMS_PLATFORM=xpu forces the XPU
                # backend; --no-build-isolation lets the build see torch and match
                # the libsycl ABI to it.
                logger.warning(
                    "enable_memory_saver is enabled, but torch-memory-saver is "
                    "not installed. On Intel XPU, build it from source with Intel "
                    "oneAPI on PATH: `TMS_PLATFORM=xpu pip3 install "
                    "--no-build-isolation git+https://github.com/fzyzcjy/"
                    "torch_memory_saver.git@a5c99f11b18ebb8e9fda71a68812e476ae49e417`."
                )
            else:
                logger.warning(
                    "enable_memory_saver is enabled, but "
                    "torch-memory-saver is not installed. Please install it "
                    "via `pip3 install torch-memory-saver`. "
                )
            raise import_error
        return (
            _TorchMemorySaverAdapterReal() if enable else _TorchMemorySaverAdapterNoop()
        )

    def check_validity(self, caller_name):
        if not self.enabled:
            logger.warning(
                f"`{caller_name}` will not save memory because torch_memory_saver is not enabled. "
                f"Potential causes: `enable_memory_saver` is false, or torch_memory_saver has installation issues."
            )

    def configure_subprocess(self):
        raise NotImplementedError

    def region(self, tag: str, enable_cpu_backup: bool = False):
        raise NotImplementedError

    def cuda_graph(self, **kwargs):
        raise NotImplementedError

    def disable(self):
        raise NotImplementedError

    def pause(self, tag: str):
        raise NotImplementedError

    def resume(self, tag: str):
        raise NotImplementedError

    @property
    def enabled(self):
        raise NotImplementedError


class _TorchMemorySaverAdapterReal(TorchMemorySaverAdapter):
    """Adapter for TorchMemorySaver with tag-based control.

    Backed by the upstream torch_memory_saver package (CUDA VMM, and Intel XPU via
    Level Zero). XPU requires the in-process pluggable allocator (hook_mode="torch")
    instead of the CUDA LD_PRELOAD path, which is what makes configure_subprocess()
    and cuda_graph() no-ops there; region/pause/resume are fully supported.
    """

    def configure_subprocess(self):
        if is_xpu():
            # Nothing to preload: this LD_PRELOADs the preload-mode .so, which the
            # upstream setup.py does not build for XPU.
            return self._noop_context()
        return self._configure_cuda_subprocess()

    @contextmanager
    def _configure_cuda_subprocess(self):
        runtime_dir = _get_pip_cuda_runtime_library_dir()
        old_library_path = os.environ.get("LD_LIBRARY_PATH")

        if runtime_dir is not None:
            entries = old_library_path.split(os.pathsep) if old_library_path else []
            runtime_dir_str = os.fspath(runtime_dir)
            if runtime_dir_str not in entries:
                os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
                    [runtime_dir_str, *entries]
                )

        try:
            with torch_memory_saver.configure_subprocess():
                yield
        finally:
            if runtime_dir is not None:
                if old_library_path is None:
                    os.environ.pop("LD_LIBRARY_PATH", None)
                else:
                    os.environ["LD_LIBRARY_PATH"] = old_library_path

    def region(self, tag: str, enable_cpu_backup: bool = False):
        return _memory_saver.region(tag=tag, enable_cpu_backup=enable_cpu_backup)

    def cuda_graph(self, **kwargs):
        if is_xpu():
            # Upstream gates pauseable graph capture on hook_mode="preload" while XPU
            # requires hook_mode="torch", so the two are mutually exclusive. Unreachable
            # today (XPU routes to FullXPUGraphBackend, which takes no memory saver);
            # warn rather than raise, so a future XPU graph backend that does route here
            # surfaces that graph memory is not pauseable instead of failing to launch.
            global _warned_xpu_cuda_graph
            if not _warned_xpu_cuda_graph:
                _warned_xpu_cuda_graph = True
                logger.warning(
                    "torch_memory_saver cannot make CUDA-graph memory pauseable on Intel "
                    "XPU; graph allocations will not be released by "
                    "release_memory_occupation(tags=['cuda_graph'])."
                )
            return self._noop_context()
        return _memory_saver.cuda_graph(**kwargs)

    @contextmanager
    def _noop_context(self, **kwargs):
        yield

    def disable(self):
        return _memory_saver.disable()

    def pause(self, tag: str):
        return _memory_saver.pause(tag=tag)

    def resume(self, tag: str):
        return _memory_saver.resume(tag=tag)

    @property
    def enabled(self):
        return _memory_saver is not None and _memory_saver.enabled


class _TorchMemorySaverAdapterNoop(TorchMemorySaverAdapter):
    @contextmanager
    def configure_subprocess(self):
        yield

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool = False):
        yield

    @contextmanager
    def cuda_graph(self, **kwargs):
        yield

    @contextmanager
    def disable(self):
        yield

    def pause(self, tag: str):
        pass

    def resume(self, tag: str):
        pass

    @property
    def enabled(self):
        return False
