from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    MiniMaxH3DenoiseBranch,
    minimax_h3_denoise_loop,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    MiniMaxH3DenoisingStage,
)
from sglang.multimodal_gen.runtime.platforms import current_platform

SIGMAS_VIDEO = [1.0, 0.65, 0.3, 0.0]
SIGMAS_AUDIO = [1.0, 0.5, 0.2, 0.0]


def _packed_layout() -> dict[str, torch.Tensor]:
    return minimax_h3_packed_sequence(
        text_len=3,
        latent_t=2,
        latent_h=4,
        latent_w=4,
        audio_t=3,
        include_keyframe_cond=False,
    )


def _initial_rows() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(0x5D3A11)
    packed = _packed_layout()
    video_rows = torch.randn(
        int(packed["img_pos"].numel()), 96, generator=generator, dtype=torch.float32
    )
    audio_rows = torch.randn(
        int(packed["audio_pos"].numel()), 32, generator=generator, dtype=torch.float32
    )
    return video_rows, audio_rows


def _run_actual(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    packed = _packed_layout()
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(3, 5120, dtype=torch.float32),
        token_tags=packed["token_tags"],
        device=device,
    )
    initial_video_rows, initial_audio_rows = _initial_rows()

    def model_forward(_model, kwargs, _step):
        video_state = kwargs["x"][0].index_select(0, branch.img_target_seq_idx)
        audio_state = kwargs["audio_x"][0].index_select(
            0, branch.audio_target_seq_idx
        )
        return 0.125 * video_state + 0.03125, -0.25 * audio_state + 0.0625

    return minimax_h3_denoise_loop(
        model=SimpleNamespace(prepare_adaln_plans=lambda _plans: None),
        model_forward=model_forward,
        positive=branch,
        initial_video_rows=initial_video_rows,
        initial_audio_rows=initial_audio_rows,
        keyframe_cond_rows=None,
        sigmas_video=SIGMAS_VIDEO,
        sigmas_audio=SIGMAS_AUDIO,
        device=device,
    )


def _run_control() -> tuple[torch.Tensor, torch.Tensor]:
    packed = _packed_layout()
    video_rows, audio_rows = _initial_rows()
    video_target = packed["update_mask"].view(-1).to(torch.bool)
    audio_target = torch.ones(
        packed["audio_pos"].numel(), dtype=torch.bool, device=video_rows.device
    )

    def update(state, velocity, sigma_curr, sigma_next):
        timestep = torch.tensor(1.0 - sigma_curr, dtype=torch.float32)
        sigma_ratio = torch.tensor(
            0.0 if sigma_curr == 0.0 else sigma_next / sigma_curr, dtype=torch.float32
        )
        denoised = state + (1.0 - timestep) * velocity
        if sigma_curr == 0.0:
            return denoised
        return sigma_ratio * state + (1.0 - sigma_ratio) * denoised

    for step in range(len(SIGMAS_VIDEO) - 1):
        video_velocity = 0.125 * video_rows[video_target] + 0.03125
        audio_velocity = -0.25 * audio_rows[audio_target] + 0.0625
        video_rows[video_target] = update(
            video_rows[video_target],
            video_velocity,
            SIGMAS_VIDEO[step],
            SIGMAS_VIDEO[step + 1],
        )
        audio_rows[audio_target] = update(
            audio_rows[audio_target],
            audio_velocity,
            SIGMAS_AUDIO[step],
            SIGMAS_AUDIO[step + 1],
        )
    return video_rows, audio_rows


def _probe_guard() -> dict[str, str]:
    batch = SimpleNamespace(
        extra={
            "minimax_h3_text_embeddings": {},
            "minimax_h3_denoise_state": {
                "latent_t": 2,
                "latent_h": 4,
                "latent_w": 4,
                "audio_t": 3,
            },
            "minimax_h3_sigmas": {"video": [1.0, 0.0], "audio": [1.0, 0.0]},
        }
    )
    stage = SimpleNamespace(
        _maybe_enable_cache_dit_and_torch_compile=lambda *_args, **_kwargs: None
    )
    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising._assemble_condition_rows",
        side_effect=RuntimeError("reached post-guard work"),
    ):
        try:
            MiniMaxH3DenoisingStage._run_full_loop(
                stage, batch, SimpleNamespace()
            )
        except RuntimeError as exc:
            if str(exc) != "reached post-guard work":
                raise
            rocm_result = "admitted"
        else:
            raise AssertionError("post-guard sentinel was not reached")

    flags = {
        name: lambda: False
        for name in (
            "is_cuda",
            "is_cpu",
            "is_hip",
            "is_mps",
            "is_npu",
            "is_xpu",
        )
    }
    with patch.multiple(current_platform, **flags):
        try:
            MiniMaxH3DenoisingStage._run_full_loop(
                stage, batch, SimpleNamespace()
            )
        except RuntimeError as exc:
            unsupported_error = str(exc)
        else:
            raise AssertionError("unsupported platform was admitted")
    return {"rocm": rocm_result, "unsupported_error": unsupported_error}


def _difference(actual, control):
    difference = (actual.detach().cpu() - control.detach().cpu()).abs()
    return {
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "actual_sum": float(actual.detach().cpu().sum().item()),
        "control_sum": float(control.detach().cpu().sum().item()),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this validation requires the assigned CUDA-compatible ROCm GPU")
    device = torch.device("cuda:0")
    if torch.cuda.get_device_capability(device) != (9, 4):
        raise RuntimeError(
            f"expected gfx942/capability (9, 4), got {torch.cuda.get_device_capability(device)}"
        )

    gpu_video, gpu_audio = _run_actual(device)
    control_video, control_audio = _run_control()
    torch.testing.assert_close(gpu_video.cpu(), control_video, rtol=0, atol=2e-6)
    torch.testing.assert_close(gpu_audio.cpu(), control_audio, rtol=0, atol=2e-6)
    results = {
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "platform": type(current_platform).__name__,
        "steps": len(SIGMAS_VIDEO) - 1,
        "guard": _probe_guard(),
        "video": _difference(gpu_video, control_video),
        "audio": _difference(gpu_audio, control_audio),
        "comparison_gate": {"rtol": 0.0, "atol": 2e-6, "result": "passed"},
    }
    output = Path(__file__).with_name("reduced_denoiser_results.json")
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
