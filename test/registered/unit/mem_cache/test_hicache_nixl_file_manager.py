"""CPU-only regression tests for scoped NIXL file-cache clearing."""

import os
import shutil
import tempfile
import unittest

from sglang.srt.mem_cache.storage.nixl.nixl_utils import NixlFileManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNixlFileManagerClear(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dirs = [
            os.path.join(self.temp_dir.name, "disk0"),
            os.path.join(self.temp_dir.name, "disk1"),
        ]
        self.manager = NixlFileManager(self.base_dirs, use_direct_io=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write(self, name: str) -> str:
        path = self.manager.get_file_path(name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as output:
            output.write(b"cache entry")
        return path

    def test_clear_only_removes_the_calling_instance_files(self):
        owned = [
            self._write("012345_model-a_0_1"),
            self._write("abcdef_model-a_0_1_mamba_temporal"),
        ]
        preserved = [
            self._write("123456_model-b_0_1"),
            self._write("234567_model-a_0_10"),
            self._write("345678_model-a-extended_0_1"),
            self._write("unrelated.txt"),
        ]

        self.manager.clear(suffix="_model-a_0_1")

        self.assertTrue(all(not os.path.exists(path) for path in owned))
        self.assertTrue(all(os.path.exists(path) for path in preserved))

    def test_clear_rejects_empty_or_degenerate_scope(self):
        path = self._write("012345_model-b_0_1")

        for suffix in (None, "", "_"):
            with (
                self.subTest(suffix=suffix),
                self.assertLogs(
                    "sglang.srt.mem_cache.storage.nixl.nixl_utils", level="ERROR"
                ),
            ):
                self.manager.clear(suffix=suffix)
            self.assertTrue(os.path.exists(path))

    def test_clear_handles_missing_base_directory(self):
        self._write("012345_model-a_0_1")
        shutil.rmtree(self.base_dirs[0])

        self.manager.clear(suffix="_model-a_0_1")


if __name__ == "__main__":
    unittest.main()
