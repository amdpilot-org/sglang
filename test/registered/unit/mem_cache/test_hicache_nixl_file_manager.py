"""CPU-only regression tests for scoped NIXL file-cache clearing."""

import os
import shutil
import tempfile
import unittest

from sglang.srt.mem_cache.storage.nixl.nixl_utils import NixlFileManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNixlFileManagerClear(unittest.TestCase):
    KEY = "a" * 64

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
            self._write(f"{self.KEY}_model-a_0_1"),
            self._write(f"{self.KEY}_model-a_0_1_mamba_temporal"),
        ]
        preserved = [
            self._write(f"{self.KEY}_model-b_0_1"),
            self._write(f"{self.KEY}_model-a_0_10"),
            self._write(f"{self.KEY}_model-a-extended_0_1"),
            self._write("unrelated.txt"),
        ]

        self.manager.clear(suffix="_model-a_0_1")

        self.assertTrue(all(not os.path.exists(path) for path in owned))
        self.assertTrue(all(os.path.exists(path) for path in preserved))

    def test_clear_does_not_match_tail_of_longer_model_name(self):
        cases = [
            ("_model_0_1", "_tenant_model_0_1"),
            ("_model", "_tenant_model"),
        ]

        for suffix, longer_model_suffix in cases:
            with self.subTest(suffix=suffix):
                owned = self._write(f"{self.KEY}{suffix}")
                foreign = self._write(f"{self.KEY}{longer_model_suffix}")

                self.manager.clear(suffix=suffix)

                self.assertFalse(os.path.exists(owned))
                self.assertTrue(os.path.exists(foreign))

    def test_clear_requires_a_sha256_cache_key_prefix(self):
        cache_file = self._write(f"{self.KEY}_model-a_0_1")
        non_cache_file = self._write("notes_model-a_0_1")

        self.manager.clear(suffix="_model-a_0_1")

        self.assertFalse(os.path.exists(cache_file))
        self.assertTrue(os.path.exists(non_cache_file))

    def test_clear_rejects_empty_or_degenerate_scope(self):
        path = self._write(f"{self.KEY}_model-b_0_1")

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
        self._write(f"{self.KEY}_model-a_0_1")
        shutil.rmtree(self.base_dirs[0])

        self.manager.clear(suffix="_model-a_0_1")


if __name__ == "__main__":
    unittest.main()
