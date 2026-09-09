"""Two-rank FSDP Qwen3 encoder weight reload correctness."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file
from torch.distributed import init_device_mesh
from torch.distributed.tensor import DTensor, distribute_tensor

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.test.test_utils import CustomTestCase

_WORLD = 2


def _tiny_config():
    from sglang.multimodal_gen.configs.models.encoders.qwen3 import (
        Qwen3TextArchConfig,
        Qwen3TextConfig,
    )

    config = Qwen3TextConfig(
        arch_config=Qwen3TextArchConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=8,
            text_len=8,
        ),
        prefix="qwen3",
    )
    config.arch_config.rope_parameters = {"rope_theta": 1000000.0}
    return config


def _checkpoint_weights(seed: int) -> dict[str, torch.Tensor]:
    from sglang.multimodal_gen.runtime.models.encoders.qwen3 import Qwen3ForCausalLM

    config = _tiny_config()
    generator = torch.Generator().manual_seed(seed)
    weights = {}
    for name, value in Qwen3ForCausalLM(config).named_parameters():
        if name == "embed_tokens.weight":
            value = value[: config.vocab_size]
        full_name = f"model.{name}"
        if name.endswith(".qkv_proj.weight"):
            for shard_name, shard in zip(
                ("q_proj", "k_proj", "v_proj"), value.chunk(3, dim=0)
            ):
                weights[
                    full_name.replace("qkv_proj", shard_name)
                ] = shard.contiguous()
        elif name.endswith(".gate_up_proj.weight"):
            for shard_name, shard in zip(
                ("gate_proj", "up_proj"), value.chunk(2, dim=0)
            ):
                weights[
                    full_name.replace("gate_up_proj", shard_name)
                ] = shard.contiguous()
        else:
            weights[full_name] = value

    return {
        name: torch.randn(
            value.shape,
            dtype=torch.float32,
            generator=generator,
        ).mul_(0.02)
        for name, value in weights.items()
    }


def _write_checkpoint(root: Path, name: str, seed: int) -> Path:
    module_dir = root / name / "text_encoder"
    module_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        _checkpoint_weights(seed),
        str(module_dir / "model.safetensors"),
    )
    return module_dir


def _prepare_checkpoints(checkpoint_root: str) -> None:
    for name, seed in (
        ("initial", 1729),
        ("first", 3141),
        ("second", 2718),
    ):
        _write_checkpoint(Path(checkpoint_root), name, seed)
    bad = _checkpoint_weights(999)
    bad["model.norm.weight"] = bad["model.norm.weight"][:-1]
    bad_dir = Path(checkpoint_root) / "bad" / "text_encoder"
    bad_dir.mkdir(parents=True)
    save_file(bad, str(bad_dir / "model.safetensors"))


def _distributed_param(param: torch.nn.Parameter) -> DTensor | None:
    if isinstance(param, DTensor):
        return param
    if isinstance(param.data, DTensor):
        return param.data
    return None


def _assert_shards_match(model, reference) -> None:
    expected = dict(reference.named_parameters())
    assert set(expected) == {name for name, _ in model.named_parameters()}
    for name, param in model.named_parameters():
        dtensor = _distributed_param(param)
        assert dtensor is not None, f"{name} is not FSDP-sharded"
        distributed = distribute_tensor(
            expected[name].to(dtensor.dtype),
            dtensor.device_mesh,
            dtensor.placements,
        )
        torch.testing.assert_close(
            dtensor._local_tensor,
            distributed._local_tensor,
            msg=lambda message: f"{name}: {message}",
        )


def _forward(model, device: torch.device) -> torch.Tensor:
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )

    input_ids = torch.tensor([[1, 7, 11, 2, 3]], device=device)
    with torch.no_grad():
        with set_forward_context(current_timestep=0, attn_metadata=None):
            return model(input_ids=input_ids).last_hidden_state


def _forward_latency(model, device: torch.device, iterations: int = 20) -> float:
    for _ in range(5):
        _forward(model, device)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    start = time.perf_counter()
    for _ in range(iterations):
        _forward(model, device)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    return (time.perf_counter() - start) / iterations


def _load_checkpoint(model, module_dir: Path, device: torch.device) -> set[str]:
    with torch.inference_mode():
        return model.load_weights(
            (name, tensor.to(device))
            for name, tensor in load_file(str(module_dir / "model.safetensors")).items()
        )


class _FakePipeline:
    def __init__(self, module: torch.nn.Module, model_path: str):
        self.modules = {"text_encoder": module}
        self.model_path = model_path

    def get_module(self, name: str):
        return self.modules.get(name)


def _worker() -> int:
    from sglang.multimodal_gen.runtime.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.multimodal_gen.runtime.loader.fsdp_load import shard_model
    from sglang.multimodal_gen.runtime.models.encoders.qwen3 import Qwen3ForCausalLM
    from sglang.multimodal_gen.runtime.post_training.weights_updater import (
        WeightsUpdater,
    )
    from sglang.multimodal_gen.runtime.server_args import set_global_server_args

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    set_global_server_args(SimpleNamespace(attention_backend=None))
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        device_id=device,
    )
    initialize_model_parallel(
        data_parallel_size=world_size,
        sequence_parallel_degree=1,
        tensor_parallel_degree=1,
    )
    checkpoint_root = os.environ.get("CHECKPOINT_ROOT")
    owns_checkpoint = checkpoint_root is None
    if checkpoint_root is None:
        payload = (
            [tempfile.mkdtemp(prefix="qwen3_fsdp_reload_")] if rank == 0 else [None]
        )
        torch.distributed.broadcast_object_list(payload, src=0)
        checkpoint_root = payload[0]
    if rank == 0:
        _prepare_checkpoints(checkpoint_root)
    torch.distributed.barrier()

    config = _tiny_config()
    model = Qwen3ForCausalLM(config).to(device).eval()
    initial_dir = Path(checkpoint_root) / "initial" / "text_encoder"
    initial_params = _load_checkpoint(model, initial_dir, device)
    assert len(initial_params) == len(dict(model.named_parameters()))
    shard_model(
        model,
        cpu_offload=False,
        reshard_after_forward=True,
        mesh=init_device_mesh("cuda", (world_size,)),
        fsdp_shard_conditions=model._fsdp_shard_conditions,
    )
    assert all(
        model._get_distributed_param(param) is not None
        for _, param in model.named_parameters()
    )
    torch.cuda.synchronize()
    torch.distributed.barrier()
    sharded_reload_start = time.perf_counter()
    reloaded_params = _load_checkpoint(model, initial_dir, device)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    sharded_reload_seconds = time.perf_counter() - sharded_reload_start
    assert len(reloaded_params) == len(initial_params)

    reference = Qwen3ForCausalLM(config).to(device).eval()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    unsharded_reload_start = time.perf_counter()
    reference_params = _load_checkpoint(reference, initial_dir, device)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    unsharded_reload_seconds = time.perf_counter() - unsharded_reload_start
    assert len(reference_params) == len(initial_params)
    torch.testing.assert_close(_forward(model, device), _forward(reference, device))
    _assert_shards_match(model, reference)
    fsdp_forward_seconds = _forward_latency(model, device)
    unsharded_forward_seconds = _forward_latency(reference, device)

    pipeline = _FakePipeline(model, str(Path(checkpoint_root) / "initial"))
    updater = WeightsUpdater(pipeline)
    updater._module_weight_dirs["text_encoder"] = str(initial_dir)

    update_seconds = {}
    for update_name in ("first", "second"):
        update_root = Path(checkpoint_root) / update_name
        torch.cuda.synchronize()
        torch.distributed.barrier()
        update_start = time.perf_counter()
        success, _ = updater.update_weights_from_disk(
            str(update_root), target_modules=["text_encoder"]
        )
        torch.cuda.synchronize()
        torch.distributed.barrier()
        update_seconds[update_name] = time.perf_counter() - update_start
        assert success
        _load_checkpoint(reference, update_root / "text_encoder", device)
        torch.testing.assert_close(
            _forward(model, device), _forward(reference, device)
        )
        _assert_shards_match(model, reference)

    bad_root = Path(checkpoint_root) / "bad"
    torch.cuda.synchronize()
    torch.distributed.barrier()
    bad_start = time.perf_counter()
    success, _ = updater.update_weights_from_disk(
        str(bad_root), target_modules=["text_encoder"]
    )
    torch.cuda.synchronize()
    torch.distributed.barrier()
    bad_rollback_seconds = time.perf_counter() - bad_start
    assert not success
    second_root = Path(checkpoint_root) / "second"
    _load_checkpoint(reference, second_root / "text_encoder", device)
    torch.testing.assert_close(_forward(model, device), _forward(reference, device))
    _assert_shards_match(model, reference)

    if rank == 0:
        print(
            "QWEN3_FSDP_RELOAD PASS "
            f"sharded_reload_ms={sharded_reload_seconds * 1e3:.3f} "
            f"unsharded_reload_ms={unsharded_reload_seconds * 1e3:.3f} "
            f"first_update_ms={update_seconds['first'] * 1e3:.3f} "
            f"second_update_ms={update_seconds['second'] * 1e3:.3f} "
            f"bad_rollback_ms={bad_rollback_seconds * 1e3:.3f} "
            f"fsdp_forward_ms={fsdp_forward_seconds * 1e3:.3f} "
            f"unsharded_forward_ms={unsharded_forward_seconds * 1e3:.3f}",
            flush=True,
        )
    torch.distributed.barrier()
    if owns_checkpoint and rank == 0:
        shutil.rmtree(checkpoint_root)
    cleanup_dist_env_and_memory()
    return 0


class TestQwen3FSDPReload(CustomTestCase):
    def test_reload_two_ranks(self):
        if not current_platform.is_cuda_alike():
            self.skipTest("CUDA/ROCm-only test")
        if torch.cuda.device_count() < _WORLD:
            self.skipTest(f"needs {_WORLD} GPUs")

        with tempfile.TemporaryDirectory() as checkpoint_root:
            env = os.environ.copy()
            env["CHECKPOINT_ROOT"] = checkpoint_root
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc_per_node={_WORLD}",
                    __file__,
                ],
                env=env,
                text=True,
                capture_output=True,
                timeout=300,
            )
            self.assertEqual(
                result.returncode, 0, f"{result.stdout}\n{result.stderr}"
            )


if __name__ == "__main__":
    if "RANK" in os.environ:
        raise SystemExit(_worker())
    unittest.main()
