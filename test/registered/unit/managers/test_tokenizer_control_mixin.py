# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.lora.lora_registry import LoRARegistry
from sglang.srt.managers.io_struct import (
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterReqInput,
    LoRAUpdateOutput,
    UnloadLoRAAdapterReqInput,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.server_args import LoRARef

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeTokenizerManager(TokenizerControlMixin):
    def __init__(self):
        self.auto_create_handle_loop = MagicMock()
        self.lora_update_lock = asyncio.Lock()
        self.lora_registry = LoRARegistry()
        self.lora_ref_cache = {}
        self.update_lora_adapter_communicator = AsyncMock(
            return_value=[LoRAUpdateOutput(success=True, loaded_adapters={})]
        )


@pytest.fixture
def runtime_context(monkeypatch):
    state = SimpleNamespace(
        enable_lora=True,
        max_loaded_loras=None,
        tokenizer_worker_num=1,
        dp_size=1,
        enable_dp_attention=False,
    )
    module = "sglang.srt.managers.tokenizer_control_mixin"
    monkeypatch.setattr(f"{module}.get_lora", lambda: state)
    monkeypatch.setattr(f"{module}.get_parallel", lambda: state)
    monkeypatch.setattr(f"{module}.get_serving", lambda: state)
    return state


def _load_req():
    return LoadLoRAAdapterReqInput(lora_name="adapter_a", lora_path="/adapter/a")


def _tensor_req():
    return LoadLoRAAdapterFromTensorsReqInput(
        lora_name="adapter_a",
        config_dict={"r": 8},
        serialized_named_tensors=[b"tp0"],
    )


def _unload_req():
    return UnloadLoRAAdapterReqInput(lora_name="adapter_a")


@pytest.mark.parametrize(
    ("method", "request_factory"),
    [
        ("load_lora_adapter", _load_req),
        ("load_lora_adapter_from_tensors", _tensor_req),
        ("unload_lora_adapter", _unload_req),
    ],
)
def test_dynamic_lora_rejected_with_multiple_tokenizer_workers(
    runtime_context, method, request_factory
):
    runtime_context.tokenizer_worker_num = 2
    manager = _FakeTokenizerManager()

    result = asyncio.run(getattr(manager, method)(request_factory()))

    assert not result.success
    assert "--tokenizer-worker-num 1" in result.error_message
    assert "not synchronized" in result.error_message
    manager.update_lora_adapter_communicator.assert_not_awaited()
    assert manager.lora_registry.num_registered_loras == 0
    assert manager.lora_ref_cache == {}


def test_multi_tokenizer_guard_also_applies_with_dp_attention(runtime_context):
    runtime_context.tokenizer_worker_num = 2
    runtime_context.dp_size = 2
    runtime_context.enable_dp_attention = True
    manager = _FakeTokenizerManager()

    result = asyncio.run(manager.load_lora_adapter(_load_req()))

    assert not result.success
    manager.update_lora_adapter_communicator.assert_not_awaited()


def test_single_tokenizer_worker_dynamic_load_and_unload_unchanged(runtime_context):
    manager = _FakeTokenizerManager()

    async def run_updates():
        loaded = await manager.load_lora_adapter(_load_req())
        unloaded = await manager.unload_lora_adapter(_unload_req())
        return loaded, unloaded

    loaded, unloaded = asyncio.run(run_updates())

    assert loaded.success
    assert unloaded.success
    assert manager.update_lora_adapter_communicator.await_count == 2
    assert manager.lora_registry.num_registered_loras == 0
    assert "adapter_a" in manager.lora_ref_cache


def test_startup_preloaded_registries_are_consistent_across_workers():
    startup_ref = LoRARef(lora_name="adapter_a", lora_path="/adapter/a")
    registries = [LoRARegistry([startup_ref]) for _ in range(2)]

    assert [registry.num_registered_loras for registry in registries] == [1, 1]
    assert asyncio.run(registries[0].acquire("adapter_a")) == startup_ref.lora_id
    assert asyncio.run(registries[1].acquire("adapter_a")) == startup_ref.lora_id
