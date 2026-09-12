import concurrent.futures
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import torch

import sglang.srt.model_loader.loader as loader_mod
import sglang.srt.model_loader.utils as model_loader_utils
import sglang.srt.model_loader.weight_utils as weight_utils
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.models.deepseek_common import deepseek_weight_loader
from sglang.srt.models.deepseek_v4 import (
    _dequant_fp8_wo_a,
    _dequant_fp8_wo_a_streaming,
)
from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _FakeModel:
    def eval(self):
        return self


class TestRunaiModelStreamerLoader(CustomTestCase):
    def _get_streamed_files(self, weight_map, draft_model_idx=1):
        with tempfile.TemporaryDirectory() as model_dir:
            files = [
                os.path.join(model_dir, "target-1.safetensors"),
                os.path.join(model_dir, "target-2.safetensors"),
                os.path.join(model_dir, "draft-0.safetensors"),
                os.path.join(model_dir, "draft-1.safetensors"),
                os.path.join(model_dir, "draft-shared.safetensors"),
            ]
            if weight_map is not None:
                with open(
                    os.path.join(model_dir, "model.safetensors.index.json"), "w"
                ) as index_file:
                    if isinstance(weight_map, str):
                        index_file.write(weight_map)
                    else:
                        json.dump({"weight_map": weight_map}, index_file)

            load_config = LoadConfig(
                load_format=LoadFormat.RUNAI_STREAMER,
                model_loader_extra_config={},
                draft_model_idx=draft_model_idx,
            )
            runai_loader = loader_mod.RunaiModelStreamerLoader(load_config)
            runai_loader.target_device_str = "cpu"
            source = runai_loader.Source(model_dir, revision=None)

            with (
                patch.object(
                    runai_loader,
                    "_prepare_weights",
                    return_value=(model_dir, files),
                ),
                patch.object(
                    loader_mod,
                    "maybe_add_mtp_safetensors",
                    side_effect=lambda files, *_args: files,
                ),
                patch.object(
                    weight_utils,
                    "runai_safetensors_weights_iterator",
                    return_value=iter(()),
                ) as mock_iterator,
            ):
                list(runai_loader._get_weights_iterator(source))

            return mock_iterator.call_args.args[0], files

    def test_selects_hf_mtp_layer_and_shared_shards_before_streaming(self):
        selected, files = self._get_streamed_files(
            {
                "model.layers.0.weight": "target-1.safetensors",
                "model.layers.1.weight": "target-2.safetensors",
                "model.mtp.layers.0.weight": "draft-0.safetensors",
                "model.mtp.layers.1.weight": "draft-1.safetensors",
                "model.mtp.shared_head.weight": "draft-shared.safetensors",
            }
        )

        self.assertEqual(selected, [files[3], files[4]])

    def test_selects_native_mtp_layer_and_shared_shards_before_streaming(self):
        selected, files = self._get_streamed_files(
            {
                "model.layers.0.weight": "target-1.safetensors",
                "model.layers.1.weight": "target-2.safetensors",
                "mtp.0.decoder.weight": "draft-0.safetensors",
                "mtp.1.decoder.weight": "draft-1.safetensors",
                "mtp.shared_head.norm.weight": "draft-shared.safetensors",
            }
        )

        self.assertEqual(selected, [files[3], files[4]])

    def test_mixed_recognized_and_unknown_draft_layout_falls_back(self):
        selected, files = self._get_streamed_files(
            {
                "model.layers.0.weight": "target-1.safetensors",
                "model.layers.1.weight": "target-2.safetensors",
                "mtp.0.decoder.weight": "draft-0.safetensors",
                "mtp.1.decoder.weight": "draft-1.safetensors",
                "model.nextn.layers.1.shared.weight": "draft-shared.safetensors",
            }
        )

        self.assertEqual(selected, files)

    def test_selects_remote_shards_using_cached_object_storage_index(self):
        model_uri = "s3://bucket/model"
        files = [
            f"{model_uri}/target.safetensors",
            f"{model_uri}/draft-1.safetensors",
            f"{model_uri}/draft-shared.safetensors",
        ]
        with tempfile.TemporaryDirectory() as metadata_dir:
            with open(
                os.path.join(metadata_dir, "model.safetensors.index.json"), "w"
            ) as index_file:
                json.dump(
                    {
                        "weight_map": {
                            "model.layers.0.weight": "target.safetensors",
                            "mtp.1.decoder.weight": "draft-1.safetensors",
                            "mtp.shared_head.weight": "draft-shared.safetensors",
                        }
                    },
                    index_file,
                )

            with patch(
                "sglang.srt.utils.runai_utils.ObjectStorageModel.get_path",
                return_value=metadata_dir,
            ):
                selected = loader_mod.RunaiModelStreamerLoader._select_mtp_safetensors(
                    files, model_uri, 1
                )

        self.assertEqual(selected, files[1:])

    def test_incomplete_index_keeps_auto_added_mtp_safetensors(self):
        with tempfile.TemporaryDirectory() as model_dir:
            target_file = os.path.join(model_dir, "target.safetensors")
            draft_file = os.path.join(model_dir, "draft-1.safetensors")
            supplemental_file = os.path.join(model_dir, "mtp.safetensors")
            files = [target_file, draft_file]
            for path in (*files, supplemental_file):
                Path(path).touch()

            with open(
                os.path.join(model_dir, "model.safetensors.index.json"), "w"
            ) as index_file:
                json.dump(
                    {
                        "weight_map": {
                            "model.layers.0.weight": "target.safetensors",
                            "mtp.1.decoder.weight": "draft-1.safetensors",
                        }
                    },
                    index_file,
                )

            load_config = LoadConfig(
                load_format=LoadFormat.RUNAI_STREAMER,
                model_loader_extra_config={},
                draft_model_idx=1,
            )
            runai_loader = loader_mod.RunaiModelStreamerLoader(load_config)
            runai_loader.target_device_str = "cpu"
            source = runai_loader.Source(
                model_dir,
                revision=None,
                model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(
                        architectures=["Glm4MoeForCausalLM"],
                        num_nextn_predict_layers=1,
                    )
                ),
            )

            with (
                patch.object(
                    runai_loader,
                    "_prepare_weights",
                    return_value=(model_dir, files),
                ),
                patch.object(
                    weight_utils,
                    "runai_safetensors_weights_iterator",
                    return_value=iter(()),
                ) as mock_iterator,
            ):
                list(runai_loader._get_weights_iterator(source))

            self.assertEqual(
                mock_iterator.call_args.args[0],
                [target_file, draft_file, supplemental_file],
            )

    def test_mtp_shard_selection_falls_back_safely(self):
        cases = {
            "missing index": None,
            "malformed index": "{not-json",
            "invalid weight map": {"model.mtp.layers.1.weight": 3},
            "unknown layout": {"model.nextn.layers.1.weight": "draft-1.safetensors"},
            "indexed shard absent": {
                "model.mtp.layers.1.weight": "missing.safetensors"
            },
            "requested layer absent": {
                "model.mtp.layers.0.weight": "draft-0.safetensors",
                "model.mtp.shared.weight": "draft-shared.safetensors",
            },
        }

        for label, weight_map in cases.items():
            with self.subTest(label=label):
                selected, files = self._get_streamed_files(weight_map)
                self.assertEqual(selected, files)

    def test_passes_quant_config_to_model_init(self):
        quant_config = object()
        fake_model = _FakeModel()

        with (
            patch.object(
                loader_mod,
                "_get_quantization_config",
                return_value=quant_config,
            ),
            patch.object(loader_mod, "_initialize_model") as mock_initialize_model,
            patch.object(
                loader_mod.DefaultModelLoader,
                "load_weights_and_postprocess",
            ) as mock_load_weights,
        ):
            mock_initialize_model.return_value = fake_model
            runai_loader = loader_mod.RunaiModelStreamerLoader(
                LoadConfig(
                    load_format=LoadFormat.RUNAI_STREAMER,
                    model_loader_extra_config={},
                )
            )
            model_config = cast(
                ModelConfig,
                SimpleNamespace(dtype=torch.float16, modelopt_quant=False),
            )

            model = runai_loader.load_model(
                model_config=model_config,
                device_config=DeviceConfig("cpu"),
            )

        self.assertIs(model, fake_model)
        self.assertIs(mock_load_weights.call_args.args[0], fake_model)
        self.assertIs(mock_initialize_model.call_args.args[2], quant_config)

    def test_marks_streamer_tensors(self):
        source_tensor = torch.tensor([1], dtype=torch.int32)

        class FakeStreamer:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def stream_files(self, *_args, **_kwargs):
                self.files_to_tensors_metadata = {0: [object()]}

            def get_tensors(self):
                yield "weight", source_tensor

        with patch.dict(
            sys.modules,
            {"runai_model_streamer": SimpleNamespace(SafetensorsStreamer=FakeStreamer)},
        ):
            weights = list(
                weight_utils.runai_safetensors_weights_iterator(["model.safetensors"])
            )

        self.assertEqual(weights[0][0], "weight")
        self.assertTrue(getattr(weights[0][1], weight_utils.RUNAI_STREAMER_TENSOR_ATTR))

    def test_deepseek_clone_only_clones_marked_tensors(self):
        unmarked = torch.tensor([1], dtype=torch.int32)

        self.assertIs(
            deepseek_weight_loader._clone_if_runai_streamed_tensor(unmarked),
            unmarked,
        )

        marked = torch.tensor([1], dtype=torch.int32)
        setattr(marked, weight_utils.RUNAI_STREAMER_TENSOR_ATTR, True)

        cloned = deepseek_weight_loader._clone_if_runai_streamed_tensor(marked)

        self.assertIsNot(cloned, marked)
        marked.fill_(2)
        self.assertEqual(cloned.item(), 1)

    def test_runai_streamed_tensor_is_consumed_before_buffer_reuse(self):
        def consume_view(mark_as_runai: bool):
            shared_buffer = torch.tensor([1], dtype=torch.int32)
            view = shared_buffer[:]
            if mark_as_runai:
                setattr(view, weight_utils.RUNAI_STREAMER_TENSOR_ATTR, True)

            release_worker = threading.Event()
            observed = []
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # Keep the sole worker busy so an async consumer cannot read the
                # zero-copy view until after the simulated streamer buffer reuse.
                blocker = executor.submit(release_worker.wait)
                model_loader_utils.maybe_executor_submit(
                    executor=executor,
                    futures=futures,
                    use_async=model_loader_utils.should_async_load(view),
                    func=lambda tensor: observed.append(tensor.item()),
                    func_args=(view,),
                )
                shared_buffer.fill_(2)
                release_worker.set()
                blocker.result()
                for future in futures:
                    future.result()

            return observed, len(futures)

        # The control demonstrates the race: an async consumer observes the
        # overwritten buffer rather than the value present when it was queued.
        self.assertEqual(consume_view(mark_as_runai=False), ([2], 1))
        # A RunAI-tagged view is consumed inline before the buffer is reused.
        self.assertEqual(consume_view(mark_as_runai=True), ([1], 0))

    def test_deepseek_v4_streaming_dequant_fp8_wo_a_pairs_weight_and_scale(self):
        weight = torch.eye(128, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale = torch.ones((1, 1), dtype=torch.float32)

        for weights in (
            [
                ("layers.0.attn.wo_a.scale", scale),
                ("layers.0.attn.wo_a.weight", weight),
                ("layers.0.attn.wq.weight", torch.tensor([3])),
            ],
            [
                ("layers.0.attn.wo_a.weight", weight),
                ("layers.0.attn.wq.weight", torch.tensor([3])),
                ("layers.0.attn.wo_a.scale", scale),
            ],
        ):
            converted = list(_dequant_fp8_wo_a_streaming(weights))

            converted_names = [name for name, _ in converted]
            self.assertIn("layers.0.attn.wo_a.weight", converted_names)
            self.assertNotIn("layers.0.attn.wo_a.scale", converted_names)
            converted_weight = dict(converted)["layers.0.attn.wo_a.weight"]
            self.assertEqual(converted_weight.dtype, torch.bfloat16)

    def test_deepseek_v4_streaming_dequant_matches_legacy_by_name(self):
        weight = torch.eye(128, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale = torch.ones((1, 1), dtype=torch.float32)
        ordinary = torch.tensor([3])
        weights = [
            ("layers.0.attn.wo_a.weight", weight),
            ("layers.0.attn.wq.weight", ordinary),
            ("layers.0.attn.wo_a.scale", scale),
        ]

        legacy = list(_dequant_fp8_wo_a(weights))
        streaming = list(_dequant_fp8_wo_a_streaming(weights))

        self.assertNotEqual(
            [name for name, _ in legacy], [name for name, _ in streaming]
        )
        self.assertEqual(set(dict(legacy)), set(dict(streaming)))
        for name, legacy_tensor in dict(legacy).items():
            torch.testing.assert_close(legacy_tensor, dict(streaming)[name])

    def test_deepseek_v4_streaming_dequant_clones_pending_runai_tensors(self):
        weight = torch.eye(128, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale = torch.ones((1, 1), dtype=torch.float32)
        setattr(scale, weight_utils.RUNAI_STREAMER_TENSOR_ATTR, True)

        def weights():
            yield "layers.0.attn.wo_a.scale", scale
            scale.fill_(0)
            yield "layers.0.attn.wo_a.weight", weight

        converted = dict(_dequant_fp8_wo_a_streaming(weights()))

        converted_weight = converted["layers.0.attn.wo_a.weight"]
        self.assertGreater(converted_weight.abs().sum().item(), 0)

    def test_deepseek_v4_dspark_load_weights_streams_wo_a_dequant(self):
        weight = torch.eye(128, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale = torch.ones((1, 1), dtype=torch.float32)
        setattr(scale, weight_utils.RUNAI_STREAMER_TENSOR_ATTR, True)
        loaded_weights = []

        def weight_loader(_param, loaded_weight):
            loaded_weights.append(loaded_weight)

        param = SimpleNamespace(weight_loader=weight_loader)
        remapper = SimpleNamespace(confidence_head=None)
        model = SimpleNamespace(
            config=SimpleNamespace(n_routed_experts=1),
            num_fused_shared_experts=0,
            named_parameters=lambda: [
                ("stages.0.self_attn.wo_a.weight", param),
            ],
            _remap_dspark_weight_name=lambda name: (
                DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(remapper, name)
            ),
            _assert_confidence_head_loaded=lambda **_kwargs: None,
        )

        def weights():
            yield "mtp.0.attn.wo_a.scale", scale
            scale.fill_(0)
            yield "mtp.0.attn.wo_a.weight", weight

        DeepseekV4ForCausalLMDSpark.load_weights(model, weights())

        self.assertEqual(len(loaded_weights), 1)
        self.assertEqual(loaded_weights[0].dtype, torch.bfloat16)
        self.assertGreater(loaded_weights[0].abs().sum().item(), 0)

    def test_deepseek_v4_streaming_dequant_preserves_missing_scale_behavior(self):
        weight = torch.eye(128, dtype=torch.float32).to(torch.float8_e4m3fn)
        ordinary = torch.tensor([3])

        converted = dict(
            _dequant_fp8_wo_a_streaming(
                [
                    ("layers.0.attn.wo_a.weight", weight),
                    ("layers.0.attn.wq.weight", ordinary),
                ]
            )
        )

        self.assertIs(converted["layers.0.attn.wo_a.weight"], weight)
        self.assertIs(converted["layers.0.attn.wq.weight"], ordinary)

        with self.assertRaises(AssertionError):
            list(
                _dequant_fp8_wo_a_streaming(
                    [
                        ("layers.0.attn.wo_a.weight", weight),
                        ("layers.1.attn.wo_a.scale", torch.ones((1, 1))),
                    ]
                )
            )

    def test_get_model_loader_uses_runai_for_prequantized_modelopt(self):
        load_config = LoadConfig(
            load_format=LoadFormat.RUNAI_STREAMER,
            model_loader_extra_config={},
        )
        model_config = cast(
            ModelConfig,
            SimpleNamespace(
                quantization="modelopt_fp4",
                modelopt_quant=False,
                _is_already_quantized=lambda: True,
            ),
        )

        model_loader = loader_mod.get_model_loader(load_config, model_config)

        self.assertIsInstance(model_loader, loader_mod.RunaiModelStreamerLoader)

    def test_get_model_loader_uses_remote_instance_for_prequantized_modelopt(self):
        load_config = LoadConfig(
            load_format=LoadFormat.REMOTE_INSTANCE,
            model_loader_extra_config={},
        )
        model_config = cast(
            ModelConfig,
            SimpleNamespace(
                quantization="modelopt_fp4",
                modelopt_quant=False,
                _is_already_quantized=lambda: True,
            ),
        )

        model_loader = loader_mod.get_model_loader(load_config, model_config)

        self.assertIsInstance(model_loader, loader_mod.RemoteInstanceModelLoader)


if __name__ == "__main__":
    unittest.main()
