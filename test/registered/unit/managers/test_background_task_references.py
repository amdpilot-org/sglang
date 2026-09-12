import ast
import asyncio
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.multimodal_gen.runtime.entrypoints.openai import mesh_api
from sglang.srt.managers.tokenizer_manager import TokenizerManager

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


REPO_ROOT = Path(__file__).resolve().parents[4]
REPORTED_FILES = (
    "python/sglang/multimodal_gen/runtime/entrypoints/openai/mesh_api.py",
    "python/sglang/srt/managers/multi_tokenizer_mixin.py",
    "python/sglang/srt/managers/tokenizer_manager.py",
)


def _is_create_task_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_task"
    )


class TestReportedFireAndForgetSites(unittest.TestCase):
    def test_no_create_task_result_is_discarded(self):
        discarded_sites = []
        for relative_path in REPORTED_FILES:
            tree = ast.parse((REPO_ROOT / relative_path).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Expr) and _is_create_task_call(node.value):
                    discarded_sites.append(f"{relative_path}:{node.lineno}")

        self.assertEqual(discarded_sites, [])


class TestTokenizerManagerBackgroundTasks(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = TokenizerManager.__new__(TokenizerManager)
        self.manager.asyncio_tasks = set()

    async def test_pending_task_is_retained_until_success(self):
        release = asyncio.Event()

        async def wait_for_release():
            await release.wait()

        task = self.manager._create_background_task(wait_for_release())
        self.assertIn(task, self.manager.asyncio_tasks)

        release.set()
        await task
        await asyncio.sleep(0)
        self.assertNotIn(task, self.manager.asyncio_tasks)

    async def test_failed_task_is_released(self):
        async def fail():
            raise RuntimeError("expected test failure")

        task = self.manager._create_background_task(fail())
        with self.assertRaisesRegex(RuntimeError, "expected test failure"):
            await task
        await asyncio.sleep(0)
        self.assertNotIn(task, self.manager.asyncio_tasks)


class TestMeshBackgroundTasks(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        tasks = list(mesh_api._MESH_JOB_TASKS)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        mesh_api._MESH_JOB_TASKS.clear()

    async def test_pending_task_is_retained_then_released(self):
        release = asyncio.Event()

        async def wait_for_release():
            await release.wait()

        task = mesh_api._start_mesh_job(wait_for_release())
        self.assertIn(task, mesh_api._MESH_JOB_TASKS)

        release.set()
        await task
        await asyncio.sleep(0)
        self.assertNotIn(task, mesh_api._MESH_JOB_TASKS)
