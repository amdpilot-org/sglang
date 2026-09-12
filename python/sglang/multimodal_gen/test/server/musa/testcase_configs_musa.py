from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from sglang.multimodal_gen.test.server.testcase_configs import (
    T2V_PROMPT,
    DiffusionSamplingParams,
    DiffusionServerArgs,
    DiffusionTestCase,
    MULTI_FRAME_I2I_sampling_params,
    MULTI_IMAGE_TI2I_sampling_params,
    T2I_sampling_params,
    T2V_sampling_params,
    TI2I_sampling_params,
    TI2V_sampling_params,
)
from sglang.multimodal_gen.test.test_utils import (
    DEFAULT_FAST_HUNYUAN_MODEL_NAME_FOR_TEST,
    DEFAULT_QWEN_IMAGE_2512_MODEL_NAME_FOR_TEST,
    DEFAULT_QWEN_IMAGE_EDIT_2509_MODEL_NAME_FOR_TEST,
    DEFAULT_QWEN_IMAGE_EDIT_MODEL_NAME_FOR_TEST,
    DEFAULT_QWEN_IMAGE_LAYERED_MODEL_NAME_FOR_TEST,
    DEFAULT_QWEN_IMAGE_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_WAN_2_1_I2V_14B_480P_MODEL_NAME_FOR_TEST,
    DEFAULT_WAN_2_1_T2V_1_3B_MODEL_NAME_FOR_TEST,
)


@lru_cache(maxsize=None)
def hf_cached_model(repo_id: str) -> str:
    """Resolve an HF repo id to the local cache snapshot prepared on MUSA runners."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id, local_files_only=True)


MUSA_TI2I_sampling_params = replace(
    TI2I_sampling_params,
    image_path="/hf-cache/hub/musa-test-assets/TI2I_Qwen_Image_Edit_Input.jpg",
)

ONE_GPU_MUSA_CASES: list[DiffusionTestCase] = [
    DiffusionTestCase(
        "qwen_image_t2i_musa",
        DiffusionServerArgs(
            model_path=hf_cached_model(DEFAULT_QWEN_IMAGE_MODEL_NAME_FOR_TEST),
            modality="image",
        ),
        T2I_sampling_params,
        run_consistency_check=False,
    ),
    DiffusionTestCase(
        "wan2_1_t2v_1.3b_musa",
        DiffusionServerArgs(
            model_path=hf_cached_model(DEFAULT_WAN_2_1_T2V_1_3B_MODEL_NAME_FOR_TEST),
            modality="video",
            custom_validator="video",
            # Server warmup caps videos at 17 frames, while this test's
            # one-second request resolves to 24. Reuse the first real request
            # shape for an unmeasured warmup so first-shape initialization
            # stays outside the denoising performance metrics.
            extras=["--warmup-mode", "request"],
        ),
        DiffusionSamplingParams(
            prompt=T2V_PROMPT,
        ),
        run_consistency_check=False,
    ),
]


NIGHTLY_1_GPU_MUSA_CASES: list[DiffusionTestCase] = [
    DiffusionTestCase(
        "zimage_image_t2i_musa",
        DiffusionServerArgs(
            model_path=hf_cached_model(DEFAULT_SMALL_MODEL_NAME_FOR_TEST),
            modality="image",
        ),
        T2I_sampling_params,
        run_consistency_check=False,
    ),
    DiffusionTestCase(
        "qwen_image_layered_i2i_musa",
        DiffusionServerArgs(
            model_path=hf_cached_model(DEFAULT_QWEN_IMAGE_LAYERED_MODEL_NAME_FOR_TEST),
            modality="image",
        ),
        MULTI_FRAME_I2I_sampling_params,
        run_consistency_check=False,
    ),
    DiffusionTestCase(
        "fast_hunyuan_video_musa",
        DiffusionServerArgs(
            model_path=hf_cached_model(DEFAULT_FAST_HUNYUAN_MODEL_NAME_FOR_TEST),
            modality="video",
            custom_validator="video",
        ),
        T2V_sampling_params,
        run_consistency_check=False,
    ),
    DiffusionTestCase(
        "qwen_image_2512_t2i_musa",
        DiffusionServerArgs(
            model_path=hf_cached_model(DEFAULT_QWEN_IMAGE_2512_MODEL_NAME_FOR_TEST),
            modality="image",
        ),
        T2I_sampling_params,
        run_consistency_check=False,
    ),
    DiffusionTestCase(
        "qwen_image_edit_t2i_musa",
        DiffusionServerArgs(
            model_path=hf_cached_model(DEFAULT_QWEN_IMAGE_EDIT_MODEL_NAME_FOR_TEST),
            modality="image",
        ),
        MUSA_TI2I_sampling_params,
        run_consistency_check=False,
    ),
    DiffusionTestCase(
        "qwen_image_edit_2509_ti2i_musa",
        DiffusionServerArgs(
            model_path=hf_cached_model(
                DEFAULT_QWEN_IMAGE_EDIT_2509_MODEL_NAME_FOR_TEST
            ),
            modality="image",
        ),
        MULTI_IMAGE_TI2I_sampling_params,
        run_consistency_check=False,
    ),
]


ONE_GPU_NIGHTLY_MUSA_CASES: list[DiffusionTestCase] = (
    ONE_GPU_MUSA_CASES + NIGHTLY_1_GPU_MUSA_CASES
)


TWO_GPU_MUSA_CASES: list[DiffusionTestCase] = [
    DiffusionTestCase(
        "wan2_1_i2v_14b_480P_2gpu_musa",
        DiffusionServerArgs(
            model_path=hf_cached_model(
                DEFAULT_WAN_2_1_I2V_14B_480P_MODEL_NAME_FOR_TEST
            ),
            modality="video",
            custom_validator="video",
            num_gpus=2,
        ),
        TI2V_sampling_params,
        run_consistency_check=False,
    ),
]
