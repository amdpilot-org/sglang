import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from torch import nn

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MODELS = Path(__file__).parents[4] / "python/sglang/srt/models"


def _calls_in_class(filename: str, class_name: str):
    tree = ast.parse((MODELS / filename).read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return [node for node in ast.walk(cls) if isinstance(node, ast.Call)]


def _is_add_prefix(node: ast.AST, suffix: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "add_prefix"
        and len(node.args) == 2
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == suffix
        and isinstance(node.args[1], ast.Name)
        and node.args[1].id == "prefix"
    )


@pytest.mark.parametrize(
    ("filename", "class_name", "constructor", "suffix"),
    [
        ("persimmon.py", "PersimmonMLP", "ColumnParallelLinear", "dense_h_to_4h"),
        ("persimmon.py", "PersimmonMLP", "RowParallelLinear", "dense_4h_to_h"),
        ("persimmon.py", "PersimmonAttention", "QKVParallelLinear", "query_key_value"),
        ("persimmon.py", "PersimmonAttention", "RowParallelLinear", "dense"),
        ("persimmon.py", "PersimmonForCausalLM", "ParallelLMHead", "lm_head"),
        ("voxtral.py", "VoxtralWhisperAttention", "QKVParallelLinear", "qkv_proj"),
        ("voxtral.py", "VoxtralWhisperAttention", "RowParallelLinear", "out_proj"),
        (
            "falcon_h1.py",
            "FalconH1HybridAttentionDecoderLayer",
            "QKVParallelLinear",
            "qkv_proj",
        ),
        (
            "falcon_h1.py",
            "FalconH1HybridAttentionDecoderLayer",
            "RowParallelLinear",
            "o_proj",
        ),
        ("hunyuan.py", "HunYuanSparseMoeBlock", "FusedMoE", "experts"),
        ("hunyuan.py", "HunYuanMoEV1ForCausalLM", "ParallelLMHead", "lm_head"),
        ("gpt_j.py", "GPTJAttention", "RadixAttention", "attn"),
        ("gpt_j.py", "GPTJForCausalLM", "ParallelLMHead", "lm_head"),
    ],
)
def test_reviewed_quantization_leaf_has_qualified_prefix(
    filename, class_name, constructor, suffix
):
    calls = _calls_in_class(filename, class_name)
    call = next(
        node
        for node in calls
        if (getattr(node.func, "id", None) or getattr(node.func, "attr", None))
        == constructor
    )
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}

    assert "quant_config" in keywords
    assert _is_add_prefix(keywords["prefix"], suffix)


@pytest.mark.parametrize(
    ("filename", "class_name", "constructor", "expected_prefix"),
    [
        ("persimmon.py", "PersimmonDecoderLayer", "PersimmonMLP", "mlp"),
        ("persimmon.py", "PersimmonModel", "make_layers", "layers"),
        (
            "voxtral.py",
            "VoxtralWhisperEncoderLayer",
            "VoxtralWhisperAttention",
            "self_attn",
        ),
        (
            "voxtral.py",
            "VoxtralForConditionalGeneration",
            "VoxtralWhisperEncoder",
            "audio_tower",
        ),
        (
            "voxtral.py",
            "VoxtralForConditionalGeneration",
            "LlamaForCausalLM",
            "language_model",
        ),
        ("hunyuan.py", "HunYuanDecoderLayer", "HunYuanSparseMoeBlock", "mlp"),
        ("hunyuan.py", "HunYuanMoEV1ForCausalLM", "HunYuanModel", "model"),
    ],
)
def test_reviewed_prefix_is_propagated_through_parent(
    filename, class_name, constructor, expected_prefix
):
    calls = _calls_in_class(filename, class_name)
    call = next(
        node
        for node in calls
        if (getattr(node.func, "id", None) or getattr(node.func, "attr", None))
        == constructor
    )
    prefix = next(keyword.value for keyword in call.keywords if keyword.arg == "prefix")

    assert expected_prefix in ast.unparse(prefix)


@pytest.mark.parametrize(
    ("outer_prefix", "expected"), [("", "lm_head"), ("nested", "nested.lm_head")]
)
def test_persimmon_lm_head_constructor_receives_outer_prefix(
    monkeypatch, outer_prefix, expected
):
    from sglang.srt.models import persimmon

    observed = {}

    class StubModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class RecordingLMHead(StubModule):
        def __init__(self, *args, **kwargs):
            super().__init__()
            observed["prefix"] = kwargs.get("prefix")

    monkeypatch.setattr(persimmon, "PersimmonModel", StubModule)
    monkeypatch.setattr(persimmon, "ParallelLMHead", RecordingLMHead)
    monkeypatch.setattr(persimmon, "LogitsProcessor", StubModule)

    config = SimpleNamespace(vocab_size=32, hidden_size=16)
    persimmon.PersimmonForCausalLM(config, quant_config=object(), prefix=outer_prefix)

    assert observed["prefix"] == expected
