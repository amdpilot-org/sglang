import argparse
import json
import os
import time
from pathlib import Path

import torch

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)


DTYPE_PAIRS = (
    (torch.float32, torch.float32),
    (torch.bfloat16, torch.float32),
    (torch.float32, torch.bfloat16),
    (torch.bfloat16, torch.float16),
    (torch.float16, torch.bfloat16),
    (torch.float16, torch.float16),
)


def reference_update(
    scheduler,
    step_index,
    sample_before,
    model_output_before,
    stochastic_sampling,
    noise,
):
    current_sigma = scheduler.sigmas[step_index].double().cpu()
    next_sigma = scheduler.sigmas[step_index + 1].double().cpu()
    sample_reference = sample_before.double().cpu()
    model_reference = model_output_before.double().cpu()
    if stochastic_sampling:
        predicted_original = sample_reference - current_sigma * model_reference
        result = (1.0 - next_sigma) * predicted_original
        return result + next_sigma * noise.double().cpu()
    delta = next_sigma - current_sigma
    return sample_reference + delta * model_reference


def run_case(
    scheduler,
    step_index,
    sample_dtype,
    model_dtype,
    stochastic_sampling,
    device,
):
    torch.manual_seed(3746)
    sample = torch.randn((2, 4, 8, 8), device=device, dtype=sample_dtype)
    model_output = torch.randn(
        (2, 4, 8, 8), device=device, dtype=model_dtype
    )
    sample_before = sample.detach().clone()
    model_output_before = model_output.detach().clone()
    timestep = scheduler.timesteps[step_index].detach().clone()
    generator = torch.Generator(device=device).manual_seed(456)
    reference_generator = torch.Generator(device=device)
    reference_generator.set_state(generator.get_state())
    scheduler._step_index = step_index

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    start.record()
    output = scheduler.step(
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        generator=generator,
        return_dict=False,
    )[0]
    end.record()
    torch.cuda.synchronize(device)
    first_event_ms = start.elapsed_time(end)

    if stochastic_sampling:
        noise = torch.randn(
            sample.shape,
            device=device,
            dtype=torch.float32,
            generator=reference_generator,
        )
    else:
        noise = None
    reference = reference_update(
        scheduler,
        step_index,
        sample_before,
        model_output_before,
        stochastic_sampling,
        noise,
    ).to(model_dtype)
    max_abs_diff = (
        output.detach().double().cpu() - reference.double()
    ).abs().max().item()

    for _ in range(3):
        scheduler._step_index = step_index
        scheduler.step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            generator=generator,
            return_dict=False,
        )
    torch.cuda.synchronize(device)
    timing_samples = []
    for _ in range(10):
        scheduler._step_index = step_index
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        scheduler.step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            generator=generator,
            return_dict=False,
        )
        end.record()
        torch.cuda.synchronize(device)
        timing_samples.append(start.elapsed_time(end))

    return {
        "stochastic_sampling": stochastic_sampling,
        "step_index": step_index,
        "timestep": float(timestep.item()),
        "sample_dtype": str(sample_dtype).removeprefix("torch."),
        "model_output_dtype": str(model_dtype).removeprefix("torch."),
        "output_dtype": str(output.dtype).removeprefix("torch."),
        "max_abs_diff_vs_float64_reference": max_abs_diff,
        "tolerance": 4.0 * torch.finfo(model_dtype).eps,
        "inputs_preserved": bool(
            torch.equal(sample, sample_before)
            and torch.equal(model_output, model_output_before)
        ),
        "first_call_cuda_event_ms": first_event_ms,
        "cuda_event_ms_samples": timing_samples,
        "cuda_event_ms_median": sorted(timing_samples)[len(timing_samples) // 2],
    }


def profile_case(scheduler, step_index, stochastic_sampling, device):
    torch.manual_seed(3746)
    sample = torch.randn(
        (2, 4, 8, 8), device=device, dtype=torch.bfloat16
    )
    model_output = torch.randn(
        (2, 4, 8, 8), device=device, dtype=torch.float32
    )
    scheduler._step_index = step_index
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as profile:
        scheduler.step(
            model_output=model_output,
            timestep=scheduler.timesteps[step_index],
            sample=sample,
            generator=torch.Generator(device=device).manual_seed(456),
            return_dict=False,
        )
        torch.cuda.synchronize(device)
    kernels = []
    for event in profile.key_averages():
        if (
            event.device_type == torch.profiler.DeviceType.CUDA
            and event.self_device_time_total > 0
        ):
            kernels.append(
                {
                    "name": event.key,
                    "count": event.count,
                    "self_cuda_time_us": event.self_device_time_total,
                }
            )
    return kernels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    device = torch.device("cuda", torch.cuda.current_device())
    cases = []
    profiler_results = {}

    for stochastic_sampling in (False, True):
        scheduler = FlowMatchEulerDiscreteScheduler(
            stochastic_sampling=stochastic_sampling
        )
        scheduler.set_timesteps(sigmas=[0.8, 0.4, 0.0], device=device)
        for step_index in (0, 1):
            for sample_dtype, model_dtype in DTYPE_PAIRS:
                cases.append(
                    run_case(
                        scheduler,
                        step_index,
                        sample_dtype,
                        model_dtype,
                        stochastic_sampling,
                        device,
                    )
                )
        profiler_results[
            "stochastic" if stochastic_sampling else "deterministic"
        ] = profile_case(
            scheduler, 0, stochastic_sampling, device
        )

    result = {
        "label": "mirror-checkout gfx942 flow-Euler step fidelity evidence",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "first_gpu_execution_elapsed_s": time.monotonic() - started,
        "gpu": {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "device_index": torch.cuda.current_device(),
        },
        "image_identity": {
            "local_image_id": "sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1",
            "source": "operator-provided qualified local image ID",
        },
        "stack": {
            "python": os.sys.executable,
            "torch": torch.__version__,
            "torch_hip": torch.version.hip,
            "torch_module": torch.__file__,
            "torch_native_lib_dir": str(Path(torch.__file__).parent / "lib"),
            "sglang_module": __import__("sglang").__file__,
            "scheduler_module": __import__(
                "sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete",
                fromlist=["FlowMatchEulerDiscreteScheduler"],
            ).__file__,
            "sgl_kernel_module": __import__("sgl_kernel").__file__,
        },
        "commands": [
            "PYTHONPATH=/job/sglang/python /opt/venv/bin/python reports/j-3746a3fe1f4b/run_step_fidelity.py reports/j-3746a3fe1f4b/results.json",
        ],
        "method": {
            "schedule": "set_timesteps(sigmas=[0.8, 0.4, 0.0], device=cuda)",
            "endpoints": "step indices 0 and 1",
            "reference": "CPU float64 independent formula, cast to model_output dtype",
            "input_preservation": "torch.equal against detached clones after step",
            "timing": "CUDA events; 3 warmups and 10 measured calls per case",
            "profiler": "One CUDA profiler pass per deterministic and stochastic mode",
        },
        "cases": cases,
        "profiler_kernels": profiler_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
