import importlib
import sys
import tempfile
from pathlib import Path

import sglang
from sglang.srt.dev_reload import reload_modules


with tempfile.TemporaryDirectory() as directory:
    path = Path(directory)
    sglang.__path__.append(directory)
    (path / "delete_source.py").write_text("def removed(): return 'stale'\n")
    (path / "delete_consumer.py").write_text(
        "from sglang.delete_source import removed\n"
    )
    source = importlib.import_module("sglang.delete_source")
    consumer = importlib.import_module("sglang.delete_consumer")

    (path / "delete_source.py").write_text(
        "replacement = 'new-and-different-length'\n"
    )
    importlib.invalidate_caches()
    result = reload_modules(["sglang.delete_source"])

    print("reload_success", result.modules)
    print("source_removed_absent", not hasattr(source, "removed"))
    print("consumer_stale_alias_callable", consumer.removed())

    for name in ("sglang.delete_source", "sglang.delete_consumer"):
        sys.modules.pop(name, None)
    sglang.__path__.remove(directory)
