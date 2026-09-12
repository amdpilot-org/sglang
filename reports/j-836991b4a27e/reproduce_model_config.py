"""Reproduce the exact VILA 1.5 config-loading failure from issue #2345."""

from sglang.srt.configs.model_config import ModelConfig


MODEL = "Efficient-Large-Model/VILA1.5-3b-AWQ"


try:
    ModelConfig(model_path=MODEL)
except ValueError as exc:
    message = str(exc)
    assert "model type `llava_llama`" in message, message
    assert "does not recognize this architecture" in message, message
    print(message)
else:
    raise AssertionError("VILA1.5-3b-AWQ unexpectedly loaded; update this report")
