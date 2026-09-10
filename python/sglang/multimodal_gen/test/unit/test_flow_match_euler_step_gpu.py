import unittest

import torch

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)


class TestFlowMatchEulerStepGPU(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA-compatible GPU")
    def test_step_matches_float64_reference_and_preserves_inputs(self):
        device = torch.device("cuda", torch.cuda.current_device())
        dtype_pairs = (
            (torch.float32, torch.float32),
            (torch.bfloat16, torch.float32),
            (torch.float32, torch.bfloat16),
            (torch.bfloat16, torch.float16),
            (torch.float16, torch.bfloat16),
            (torch.float16, torch.float16),
        )
        step_indices = (0, 1)

        for stochastic_sampling in (False, True):
            scheduler = FlowMatchEulerDiscreteScheduler(
                stochastic_sampling=stochastic_sampling
            )
            scheduler.set_timesteps(sigmas=[0.8, 0.4, 0.0], device=device)

            for step_index in step_indices:
                for sample_dtype, model_dtype in dtype_pairs:
                    with self.subTest(
                        stochastic_sampling=stochastic_sampling,
                        step_index=step_index,
                        sample_dtype=sample_dtype,
                        model_dtype=model_dtype,
                    ):
                        torch.manual_seed(3746)
                        sample = torch.randn(
                            (2, 4, 8, 8), device=device, dtype=sample_dtype
                        )
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

                        output = scheduler.step(
                            model_output=model_output,
                            timestep=timestep,
                            sample=sample,
                            generator=generator,
                            return_dict=False,
                        )[0]

                        current_sigma = scheduler.sigmas[step_index].double().cpu()
                        next_sigma = scheduler.sigmas[step_index + 1].double().cpu()
                        sample_reference = sample_before.double().cpu()
                        model_reference = model_output_before.double().cpu()
                        if stochastic_sampling:
                            noise = torch.randn(
                                sample.shape,
                                device=device,
                                dtype=torch.float32,
                                generator=reference_generator,
                            )
                            predicted_original = (
                                sample_reference - current_sigma * model_reference
                            )
                            reference = (1.0 - next_sigma) * predicted_original
                            reference = reference + next_sigma * noise.double().cpu()
                        else:
                            delta = next_sigma - current_sigma
                            reference = sample_reference + delta * model_reference
                        reference = reference.to(model_dtype)

                        tolerance = 4.0 * torch.finfo(model_dtype).eps
                        self.assertEqual(output.dtype, model_dtype)
                        self.assertTrue(
                            torch.allclose(
                                output.detach().cpu(),
                                reference,
                                rtol=0.0,
                                atol=tolerance,
                            )
                        )
                        self.assertTrue(torch.equal(sample, sample_before))
                        self.assertTrue(torch.equal(model_output, model_output_before))


if __name__ == "__main__":
    unittest.main()
