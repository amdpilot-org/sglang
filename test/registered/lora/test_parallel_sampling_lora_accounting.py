import copy
import ast
import asyncio
import logging
import os
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.managers.io_struct import GenerateReqInput


def _load_handle_batch_request():
    """Load the actual method without importing GPU-heavy manager dependencies."""
    source_path = Path(
        os.environ.get(
            "SGLANG_TOKENIZER_MANAGER_SOURCE",
            Path(__file__).parents[3]
            / "python/sglang/srt/managers/tokenizer_manager.py",
        )
    )
    module = ast.parse(source_path.read_text())
    class_node = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "TokenizerManager"
    )
    function_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_handle_batch_request"
    )
    function_node.decorator_list = []
    function_node.returns = None
    for argument in (
        function_node.args.posonlyargs
        + function_node.args.args
        + function_node.args.kwonlyargs
    ):
        argument.annotation = None
    namespace = {
        "asyncio": asyncio,
        "copy": copy,
        "get_bool_env_var": lambda _: False,
        "logger": logging.getLogger(__name__),
        "nullcontext": nullcontext,
    }
    exec(compile(ast.Module([function_node], []), source_path, "exec"), namespace)
    return namespace["_handle_batch_request"]


_handle_batch_request = _load_handle_batch_request()


class _TimeStats:
    def set_finished_time(self):
        pass


class _Manager:
    def __init__(self):
        self.rid_to_state = {}
        self.release_count = 0
        self.sent = []

    async def _tokenize_one_request(self, obj):
        return SimpleNamespace(
            rid=obj.rid,
            lora_id=obj.lora_id,
            sampling_params=SimpleNamespace(max_new_tokens=8),
            stream=obj.stream,
            mm_inputs=None,
            input_ids=[1, 2],
        )

    def _should_use_batch_tokenization(self, batch_size, obj):
        return False

    def _init_req_state(self, obj):
        self.rid_to_state[obj.rid] = SimpleNamespace(
            obj=obj, time_stats=_TimeStats(), prompt_token_ids=None
        )

    async def _send_one_request(self, tokenized_obj):
        self.sent.append(copy.copy(tokenized_obj))

    def _wait_one_response(self, obj, request):
        async def response():
            state = self.rid_to_state.pop(obj.rid)
            if state.obj.lora_path:
                self.release_count += 1
            yield {"rid": obj.rid}

        return response()

    async def _collect_batch_responses(self, generators):
        return [await generator.__anext__() for generator in generators]


class TestParallelSamplingLoRAAccounting(unittest.IsolatedAsyncioTestCase):
    async def _run_request(self, prompts, n):
        obj = GenerateReqInput(
            text=prompts,
            lora_path="adapter",
            sampling_params={"n": n, "max_new_tokens": 8},
        )
        obj.normalize_batch_and_arguments()
        if obj.is_single:
            obj.lora_id = "adapter-id"
            acquired_count = 1
        else:
            obj.lora_id = ["adapter-id"] * len(obj.lora_path)
            acquired_count = len(obj.lora_id)

        manager = _Manager()
        if obj.is_single:
            manager._init_req_state(obj)
        else:
            for index in range(len(obj.rid)):
                manager._init_req_state(obj[index])

        outputs = [
            output
            async for output in _handle_batch_request(manager, obj)
        ]
        return acquired_count, manager, outputs

    async def test_parallel_sampling_balances_one_release_per_acquire(self):
        acquired_count, manager, outputs = await self._run_request("prompt", n=4)

        self.assertEqual(manager.release_count, acquired_count)
        self.assertEqual(len(outputs[0]), 4)
        self.assertEqual(len(manager.sent), 5)
        self.assertEqual(manager.sent[0].lora_id, "adapter-id")

    async def test_parallel_sampling_multiple_prompts_balances_each_prefix(self):
        acquired_count, manager, outputs = await self._run_request(
            ["first", "second"], n=3
        )

        self.assertEqual(manager.release_count, acquired_count)
        self.assertEqual(len(outputs[0]), 6)
        self.assertEqual(len(manager.sent), 8)

    async def test_single_sample_does_not_use_prefix_warmup(self):
        acquired_count, manager, outputs = await self._run_request(["prompt"], n=1)

        self.assertEqual(manager.release_count, acquired_count)
        self.assertEqual(len(outputs[0]), 1)
        self.assertEqual(len(manager.sent), 1)


if __name__ == "__main__":
    unittest.main()
