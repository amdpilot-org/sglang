import os

os.environ["SGLANG_ROPE_CACHE_FP32"] = "1"
os.environ["USE_ROCM_AITER_ROPE_BACKEND"] = "0"

import json
import tempfile
import unittest
from array import array

import torch

from sglang.benchmark.one_batch import TreeCacheNamespace
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
)
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    set_forward_context,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import publish
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


INPUT_CASES = {
    "zeros": torch.zeros(2, 16, dtype=torch.int64),
    "tiny_finite": torch.arange(32, dtype=torch.int64).remainder(4).reshape(2, 16),
    "mixed_magnitudes": torch.arange(32, dtype=torch.int64).remainder(128).reshape(2, 16),
    "alternating_extremes": torch.tensor(
        [0, 127] * 16, dtype=torch.int64
    ).reshape(2, 16),
    "skewed_low": torch.tensor(
        [1] * 14 + [2, 3], dtype=torch.int64
    ).repeat(2, 1),
    "skewed_high": torch.tensor(
        [126] * 14 + [125, 124], dtype=torch.int64
    ).repeat(2, 1),
}


class TestLlamaStructuredInputRobustness(unittest.TestCase):
    """Compare real ModelRunner prefill outputs with a raw Torch control."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA/ROCm is not available")
        if torch.version.hip is None:
            raise unittest.SkipTest("ROCm is required")
        cls.config_dir = tempfile.TemporaryDirectory()
        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "torch_dtype": "bfloat16",
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "attention_bias": False,
            "tie_word_embeddings": True,
        }
        with open(f"{cls.config_dir.name}/config.json", "w") as config_file:
            json.dump(config, config_file)

        cls.server_args = ServerArgs(
            model_path=cls.config_dir.name,
            tokenizer_path=cls.config_dir.name,
            host="127.0.0.1",
            port=30000,
            tp_size=1,
            mem_fraction_static=0.2,
            load_format="dummy",
            attention_backend="triton",
            disable_cuda_graph=True,
            disable_radix_cache=True,
            disable_hybrid_swa_memory=True,
            dtype="bfloat16",
            context_length=32,
            max_total_tokens=256,
        )
        port_args = PortArgs.init_new(cls.server_args)
        publish(cls.server_args, role="scheduler")
        model_config = ModelConfig.from_server_args(cls.server_args)
        cls.model_runner = ModelRunner(
            model_config=model_config,
            mem_fraction_static=cls.server_args.mem_fraction_static,
            gpu_id=0,
            ps=ParallelState.trivial(tp_size=1),
            nccl_port=port_args.nccl_port,
            server_args=cls.server_args,
        )
        cls.model_runner.alloc_memory_pool()
        cls.model_runner.init_attention_backends()
        cls.model_runner.init_cuda_graphs()
        set_forward_context(ForwardContext(attn_backend=cls.model_runner.attn_backend))

    @classmethod
    def tearDownClass(cls):
        cls.config_dir.cleanup()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def _run_model_runner(self, input_ids):
        sampling_params = SamplingParams(temperature=0.0, max_new_tokens=1)
        requests = []
        for sequence in input_ids.tolist():
            request = Req(
                rid=len(requests),
                origin_input_text="",
                origin_input_ids=array("q", sequence),
                sampling_params=sampling_params,
            )
            request.full_untruncated_fill_ids = request.origin_input_ids
            request.logprob_start_len = -1
            request.set_extend_range(
                len(request.prefix_indices), len(request.full_untruncated_fill_ids)
            )
            requests.append(request)

        tree_cache = TreeCacheNamespace(
            page_size=self.model_runner.page_size,
            device=self.model_runner.device,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
        )
        batch = ScheduleBatch.init_new(
            reqs=requests,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        if batch.input_ids is None and batch.prefill_input_ids_cpu is not None:
            batch.input_ids = batch.prefill_input_ids_cpu.to(
                batch.device, non_blocking=True
            )
            batch.prefill_input_ids_cpu = None
        forward_batch = ForwardBatch.init_new(
            batch,
            self.model_runner,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            return_hidden_states_before_norm=False,
        )
        output = self.model_runner.forward(forward_batch).logits_output.hidden_states
        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()
        return output

    def _reference_forward(self, input_ids):
        parameters = {
            name: value.detach().float()
            for name, value in self.model_runner.model.state_dict().items()
        }
        num_heads = 4
        num_kv_heads = 2
        head_dim = 16
        hidden_states = torch.empty(
            input_ids.shape[0], input_ids.shape[1], 64,
            dtype=torch.float32,
            device=input_ids.device,
        )

        for sequence_index, sequence_ids in enumerate(input_ids):
            hidden = parameters["model.embed_tokens.weight"][sequence_ids]
            residual = None
            positions = torch.arange(
                sequence_ids.shape[0], device=sequence_ids.device, dtype=torch.float32
            )
            inverse_frequency = 1.0 / (
                10000.0
                ** (
                    torch.arange(0, head_dim, 2, device=sequence_ids.device)
                    / head_dim
                )
            )
            frequencies = positions[:, None] * inverse_frequency
            cosine = torch.cos(frequencies).unsqueeze(1)
            sine = torch.sin(frequencies).unsqueeze(1)

            for layer_index in range(2):
                prefix = f"model.layers.{layer_index}"
                if residual is None:
                    residual = hidden
                else:
                    residual = residual + hidden
                variance = residual.pow(2).mean(dim=-1, keepdim=True)
                normalized = residual * torch.rsqrt(variance + 1e-5)
                normalized = normalized * parameters[f"{prefix}.input_layernorm.weight"]

                qkv = normalized @ parameters[f"{prefix}.self_attn.qkv_proj.weight"].T
                query, key, value = qkv.split([64, 32, 32], dim=-1)
                query = query.reshape(-1, num_heads, head_dim)
                key = key.reshape(-1, num_kv_heads, head_dim)
                value = value.reshape(-1, num_kv_heads, head_dim)

                def apply_rope(tensor):
                    first, second = tensor.chunk(2, dim=-1)
                    return torch.cat(
                        (
                            first * cosine - second * sine,
                            second * cosine + first * sine,
                        ),
                        dim=-1,
                    )

                query = apply_rope(query)
                key = apply_rope(key)
                repeated_key = key.repeat_interleave(
                    num_heads // num_kv_heads, dim=1
                )
                repeated_value = value.repeat_interleave(
                    num_heads // num_kv_heads, dim=1
                )
                scores = torch.einsum("qhd,khd->hqk", query, repeated_key)
                scores = scores / (head_dim**0.5)
                causal_mask = torch.ones(
                    scores.shape[-2:], dtype=torch.bool, device=scores.device
                ).triu(diagonal=1)
                scores = scores.masked_fill(causal_mask, float("-inf"))
                probabilities = torch.softmax(scores, dim=-1)
                attention = torch.einsum(
                    "hqk,khd->qhd", probabilities, repeated_value
                ).reshape(-1, 64)
                attention = attention @ parameters[f"{prefix}.self_attn.o_proj.weight"].T

                residual = residual + attention
                variance = residual.pow(2).mean(dim=-1, keepdim=True)
                normalized = residual * torch.rsqrt(variance + 1e-5)
                normalized = normalized * parameters[
                    f"{prefix}.post_attention_layernorm.weight"
                ]
                gate_up = normalized @ parameters[f"{prefix}.mlp.gate_up_proj.weight"].T
                gate, up = gate_up.split([128, 128], dim=-1)
                hidden = (
                    torch.nn.functional.silu(gate) * up
                ) @ parameters[f"{prefix}.mlp.down_proj.weight"].T

            residual = residual + hidden
            variance = residual.pow(2).mean(dim=-1, keepdim=True)
            hidden = residual * torch.rsqrt(variance + 1e-5)
            hidden = hidden * parameters["model.norm.weight"]
            hidden_states[sequence_index] = hidden

        return hidden_states.reshape(-1, 64)

    def test_structured_inputs_match_independent_torch_reference(self):
        for name, input_ids in INPUT_CASES.items():
            with self.subTest(case=name):
                input_ids = input_ids.cuda()
                actual = self._run_model_runner(input_ids)
                expected = self._reference_forward(input_ids)
                self.assertEqual(actual.shape, (32, 64))
                self.assertEqual(actual.dtype, torch.bfloat16)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue(torch.isfinite(expected).all())
                difference = (actual.float() - expected).abs()
                max_absolute_error = difference.max().item()
                relative_denominator = expected.abs().clamp_min(1e-6)
                max_relative_error = (
                    (difference / relative_denominator).max().item()
                )
                print(
                    f"{name}: max_abs={max_absolute_error:.8g}, "
                    f"max_rel={max_relative_error:.8g}"
                )
                torch.testing.assert_close(
                    actual.float(),
                    expected,
                    rtol=1e-2,
                    atol=2e-6,
                    msg=f"Structured input case {name} mismatch",
                )


if __name__ == "__main__":
    unittest.main()
