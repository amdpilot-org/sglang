import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha import (
    DeepseekMHAForwardMixin,
    _use_aiter_gfx95,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    make_loc_fn,
)
from sglang.test.kits.attention_unittest.attention_methods.dsa_attention import (
    DSAAttentionCase,
    DSAMockModelRunner,
    TinyDSAModelConfig,
    _make_forward_batch,
)


def source_commit() -> str:
    import sglang

    source_root = Path(sglang.__file__).resolve().parents[2]
    return subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()


def build_raw_reference(cache, physical_indices):
    quantized = cache[physical_indices].view(torch.uint8)
    nope_quantized = quantized[:, 0, :512].view(torch.float8_e4m3fn)
    scales = quantized[:, 0, 512:528].view(torch.float32)
    rope = quantized[:, 0, 528:].view(torch.bfloat16)

    expected_nope = torch.empty(
        (physical_indices.numel(), 512), dtype=torch.bfloat16, device=cache.device
    )
    for block in range(4):
        expected_nope[:, block * 128 : (block + 1) * 128] = (
            nope_quantized[:, block * 128 : (block + 1) * 128].float()
            * scales[:, block, None]
        ).to(torch.bfloat16)
    expected_rope = rope
    return expected_nope, expected_rope


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(5109)
    device = "cuda"
    prefix_lens = (64, 32)
    extend_lens = (2, 2)
    page_size = 1
    max_context_len = 128
    case = DSAAttentionCase(
        name="gfx942_eagle_dsa_fp8_mha_prefix",
        backend="dsa",
        forward_mode=ForwardMode.EXTEND,
        num_heads=4,
        num_kv_heads=1,
        page_size=page_size,
        prefix_lens=prefix_lens,
        extend_lens=extend_lens,
    )
    model_config = TinyDSAModelConfig(
        num_heads=case.num_heads,
        num_kv_heads=case.num_kv_heads,
        head_dim=576,
        hidden_size=512,
        context_len=max_context_len,
        qk_nope_head_dim=512,
        qk_rope_head_dim=64,
        kv_lora_rank=512,
        index_topk=8,
    )
    runner = DSAMockModelRunner(
        case=case,
        model_config=model_config,
        dtype=torch.bfloat16,
        device=device,
        max_context_len=max_context_len,
        head_dim=576,
        fp8_kv_cache=True,
    )

    spec_override = get_context().override_server_args(
        attention_backend="dsa",
        dsa_prefill_backend="flashmla_auto",
        dsa_decode_backend="flashmla_kv",
        speculative_algorithm="EAGLE",
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=4,
        speculative_num_steps=3,
        tp_size=1,
    )
    spec_override.install()
    try:
        factory = DraftBackendFactory(
            runner,
            topk=1,
            speculative_num_steps=3,
        )
        draft_backend = factory.create_decode_backend()
        draft_extend_backend = factory.create_draft_extend_backend()

        loc_fn = make_loc_fn(
            "shuffled_pages",
            batch_size=case.batch_size,
            seq_lens=case.seq_lens,
            prefix_lens=prefix_lens,
            page_size=page_size,
            max_context_len=max_context_len,
            seed=5109,
        )
        forward_batch = _make_forward_batch(
            case,
            runner,
            max_context_len=max_context_len,
            device=device,
            loc_fn=loc_fn,
        )
        expected_indices = torch.tensor(
            [
                loc_fn(request_index, position)
                for request_index, sequence_length in enumerate(case.seq_lens)
                for position in range(sequence_length)
            ],
            dtype=torch.int64,
            device=device,
        )

        cache = runner.token_to_kv_pool.get_key_buffer(0).view(torch.uint8)
        quantized_values = torch.linspace(-2.0, 2.0, 512, device=device).to(
            torch.float8_e4m3fn
        )
        rope_values = torch.linspace(-1.0, 1.0, 64, device=device).to(
            torch.bfloat16
        )
        for physical_index in expected_indices.tolist():
            row = cache[physical_index].reshape(-1)
            row[:512] = quantized_values.view(torch.uint8)
            row[512:528] = torch.tensor(
                [0.25, 0.5, 1.0, 2.0], device=device, dtype=torch.float32
            ).view(torch.uint8)
            row[528:] = (
                rope_values + (physical_index % 16) * 0.01
            ).to(torch.bfloat16).view(torch.uint8)

        assert isinstance(draft_extend_backend, DeepseekSparseAttnBackend)
        draft_extend_backend._get_device_sm = lambda: 90
        with forward_context(ForwardContext(attn_backend=draft_extend_backend)):
            draft_extend_backend.init_forward_metadata(forward_batch)
            actual_indices = draft_extend_backend.forward_metadata.page_table_1_flattened
            attention = SimpleNamespace(
                attn_mha=SimpleNamespace(layer_id=0),
                kv_lora_rank=512,
                qk_rope_head_dim=64,
            )
            actual_nope, actual_rope = (
                DeepseekMHAForwardMixin._get_mla_kv_buffer_from_fp8_for_dsa(
                    attention, forward_batch
                )
            )

        expected_nope, expected_rope = build_raw_reference(cache, expected_indices)
        actual_rope = actual_rope.squeeze(1)
        translated_indices = runner.kv_index_translator.translate_dcp_read_ids(
            expected_indices
        )
        indices_equal = torch.equal(
            actual_indices.to(torch.int64), expected_indices
        )
        nope_equal = torch.equal(actual_nope, expected_nope)
        rope_equal = torch.equal(actual_rope, expected_rope)
        nope_max_abs = (
            (actual_nope.float() - expected_nope.float()).abs().max().item()
        )
        rope_max_abs = (
            (actual_rope.float() - expected_rope.float()).abs().max().item()
        )

        result = {
            "label": args.label,
            "commit": source_commit(),
            "gpu": {
                "name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "gcn_arch_name": torch.cuda.get_device_properties(0).gcnArchName,
            },
            "case": {
                "prefix_lens": list(prefix_lens),
                "extend_lens": list(extend_lens),
                "page_size": page_size,
                "kv_cache_dtype": str(torch.float8_e4m3fn),
                "physical_index_count": int(expected_indices.numel()),
            },
            "late_backend_binding": {
                "owning_runner_translator": repr(runner.kv_index_translator),
                "draft_backend_type": type(draft_backend).__name__,
                "draft_child_translators": [
                    repr(child.kv_index_translator)
                    for child in draft_backend.attn_backends
                ],
                "draft_extend_backend_type": type(draft_extend_backend).__name__,
                "draft_extend_translator": repr(
                    draft_extend_backend.kv_index_translator
                ),
            },
            "prefix_read": {
                "gfx95_read_door": _use_aiter_gfx95,
                "translator_passthrough_equal": torch.equal(
                    translated_indices, expected_indices
                ),
                "translator_needs_read_translate": (
                    runner.kv_index_translator.needs_read_translate
                ),
                "metadata_indices_equal_expected": indices_equal,
                "nope_equal_reference": nope_equal,
                "rope_equal_reference": rope_equal,
                "nope_max_abs": nope_max_abs,
                "rope_max_abs": rope_max_abs,
                "use_mha": draft_extend_backend.use_mha,
                "using_mha_one_shot_fp8_dequant": (
                    forward_batch.using_mha_one_shot_fp8_dequant
                ),
            },
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        if not (indices_equal and nope_equal and rope_equal):
            raise SystemExit(1)
    finally:
        spec_override.restore()


if __name__ == "__main__":
    main()
