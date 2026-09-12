import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
    DenoisingStepState,
)
from sglang.multimodal_gen.runtime.pipelines_core.step_reuse import (
    StepReuseController,
    StepReuseDecision,
    StepReusePolicy,
    StepSideEffectContract,
)


class _Adapter:
    def __init__(
        self,
        *,
        history_size=2,
        max_skip_steps=2,
        terminal=True,
        budget=2,
        trace_only=False,
    ):
        self.policy = StepReusePolicy(
            policy_name="synthetic_similarity",
            observation_point="post_cfg_velocity",
            history_size=history_size,
            max_skip_steps=max_skip_steps,
            force_real_steps=frozenset({"terminal"}),
            state_scope=("request", "modality", "cfg_role"),
            trace_only=trace_only,
        )
        self.side_effect_contract = StepSideEffectContract(
            terminal_write_required=terminal,
            write_tags=frozenset({"session_kv"}),
        )
        self.budget = budget
        self.verified = True

    def scope_values(self, batch):
        return {
            "request": batch.request_id,
            "modality": batch.modality,
            "cfg_role": batch.cfg_role,
        }

    def observe_real_prediction(self, prediction, *, step_index, state):
        return prediction.float()

    def decide_reuse(self, *, step_index, state):
        delta = (
            float((state.real_history[-1] - state.real_history[-2]).abs().max())
            if len(state.real_history) > 1
            else 0.0
        )
        return self.budget, {"max_abs_delta": delta}

    def verify_side_effects(self, *, state):
        return self.verified


def _batch(request="r1", modality="video", cfg_role="guided"):
    return SimpleNamespace(request_id=request, modality=modality, cfg_role=cfg_role)


class TestStepReuseController(unittest.TestCase):
    def test_real_history_budget_and_terminal_side_effect(self):
        controller = StepReuseController(_Adapter())
        controller.begin_scope(_batch())
        trace = []
        for step in range(6):
            decision = controller.before_step(step, 6)
            trace.append("reuse" if decision.reuse else "real")
            if decision.reuse:
                controller.reused_prediction()
            else:
                controller.after_real_forward(torch.tensor([float(step)]), step)
            controller.after_scheduler_step()

        metrics = controller.finalize_scope()
        self.assertEqual(trace, ["real", "real", "reuse", "reuse", "real", "real"])
        self.assertEqual(metrics["real_forwards"], 4)
        self.assertEqual(metrics["reused_steps"], 2)
        self.assertEqual(metrics["scheduler_steps"], 6)
        self.assertEqual(metrics["reuse_streaks"], [2])
        self.assertIn("terminal_side_effect", metrics["forced_real_reasons"])
        self.assertTrue(metrics["terminal_write_verified"])

    def test_reused_predictions_never_extend_real_history(self):
        controller = StepReuseController(_Adapter(history_size=1, budget=2))
        state = controller.begin_scope(_batch())
        controller.before_step(0, 4)
        controller.after_real_forward(torch.tensor([1.0]), 0)
        original = state.real_history[0]
        for step in (1, 2):
            self.assertTrue(controller.before_step(step, 4).reuse)
            self.assertIs(controller.reused_prediction(), original)
        self.assertEqual(len(state.real_history), 1)
        self.assertEqual(state.real_forwards, 1)

    def test_scope_isolation_and_retry_reset(self):
        controller = StepReuseController(_Adapter(history_size=1))
        first = controller.begin_scope(_batch("request-a", "video", "cond"))
        controller.before_step(0, 3)
        controller.after_real_forward(torch.tensor([4.0]), 0)
        second = controller.begin_scope(_batch("request-a", "action", "cond"))
        self.assertIsNot(first, second)
        self.assertIsNone(second.last_real_prediction)
        retried = controller.begin_scope(_batch("request-a", "action", "cond"))
        self.assertIsNot(second, retried)
        self.assertEqual(retried.real_forwards, 0)
        controller.abort_scope()
        with self.assertRaises(RuntimeError):
            _ = controller.state

    def test_invalid_configuration_and_adapter_budget_fail_closed(self):
        with self.assertRaises(ValueError):
            StepReusePolicy("bad", "post_cfg", 0, 1)
        with self.assertRaises(ValueError):
            StepReusePolicy("bad", "post_cfg", 1, -1)
        adapter = _Adapter(history_size=1, max_skip_steps=1, budget=2)
        controller = StepReuseController(adapter)
        controller.begin_scope(_batch())
        controller.before_step(0, 3)
        with self.assertRaises(ValueError):
            controller.after_real_forward(torch.tensor([1.0]), 0)

    def test_all_ranks_consume_control_rank_token(self):
        tokens = []

        def control_sync(decision):
            tokens.append(decision)
            return decision

        control = StepReuseController(
            _Adapter(history_size=1), synchronize_decision=control_sync
        )
        follower = StepReuseController(
            _Adapter(history_size=1),
            synchronize_decision=lambda decision: tokens[-1],
            is_control_rank=False,
        )
        for controller in (control, follower):
            controller.begin_scope(_batch())
            controller.state.last_real_prediction = torch.tensor([1.0])
            controller.state.real_history.append(torch.tensor([1.0]))
            controller.state.skip_remaining = 1
        leader_decision = control.before_step(1, 4)
        follower_decision = follower.before_step(1, 4)
        self.assertEqual(leader_decision, StepReuseDecision(True, "reuse_budget", 0))
        self.assertEqual(follower_decision, leader_decision)

    def test_terminal_verification_failure_is_fatal(self):
        adapter = _Adapter(history_size=1)
        adapter.verified = False
        controller = StepReuseController(adapter)
        controller.begin_scope(_batch())
        controller.before_step(0, 1)
        controller.after_real_forward(torch.tensor([1.0]), 0)
        controller.after_scheduler_step()
        with self.assertRaisesRegex(RuntimeError, "session_kv"):
            controller.finalize_scope()

    def test_trace_only_records_decision_but_does_not_reuse(self):
        controller = StepReuseController(
            _Adapter(history_size=1, terminal=False, trace_only=True)
        )
        controller.begin_scope(_batch())
        for step in range(3):
            decision = controller.before_step(step, 3)
            if decision.reuse:
                controller.record_trace_only_reuse()
            controller.after_real_forward(torch.tensor([float(step)]), step)
            controller.after_scheduler_step()
        metrics = controller.finalize_scope()
        self.assertEqual(metrics["real_forwards"], 3)
        self.assertEqual(metrics["reused_steps"], 0)
        self.assertEqual(metrics["trace_only_reuse_steps"], 1)

    def test_cfg_roles_have_independent_lifecycle(self):
        controller = StepReuseController(_Adapter(history_size=1))
        cond = controller.begin_scope(_batch(cfg_role="cond"))
        controller.before_step(0, 2)
        controller.after_real_forward(torch.tensor([1.0]), 0)
        uncond = controller.begin_scope(_batch(cfg_role="uncond"))
        self.assertIsNot(cond, uncond)
        self.assertEqual(uncond.real_forwards, 0)


class _Scheduler:
    def __init__(self):
        self.outputs = []

    def scale_model_input(self, value, timestep):
        return value

    def step(self, model_output, timestep, sample, **kwargs):
        self.outputs.append(model_output.clone())
        return (sample - model_output,)


class TestDenoisingStepReuseIntegration(unittest.TestCase):
    def test_scheduler_advances_when_transformer_is_skipped(self):
        stage = DenoisingStage.__new__(DenoisingStage)
        stage._current_use_nvtx = False
        stage.expand_timestep_before_forward = lambda *args: args[2]
        stage.post_forward_for_ti2v_task = lambda *args: args[-2]
        calls = []

        def predict(**kwargs):
            calls.append(kwargs["timestep_index"])
            return torch.tensor([2.0])

        stage._predict_noise_with_cfg = predict
        controller = StepReuseController(_Adapter(history_size=1, terminal=False))
        controller.begin_scope(_batch())
        scheduler = _Scheduler()
        ctx = SimpleNamespace(
            latents=torch.tensor([10.0]),
            target_dtype=torch.float32,
            seq_len=None,
            reserved_frames_mask=None,
            timesteps=torch.tensor([3, 2, 1]),
            scheduler=scheduler,
            extra_step_kwargs={},
            cfg_policy=None,
            guidance=None,
            z=None,
            extra={"step_reuse_controller": controller},
        )
        batch = SimpleNamespace(image_latent=None)
        server_args = SimpleNamespace(
            comfyui_mode=False,
            pipeline_config=SimpleNamespace(task_type=None),
        )
        for index in range(3):
            step = DenoisingStepState(
                index,
                ctx.timesteps[index],
                ctx.timesteps[index],
                int(ctx.timesteps[index]),
                None,
                None,
                None,
            )
            stage._run_denoising_step(ctx, step, batch, server_args)

        self.assertEqual(calls, [0, 2])
        self.assertEqual(len(scheduler.outputs), 3)
        self.assertTrue(torch.equal(ctx.latents, torch.tensor([4.0])))


if __name__ == "__main__":
    unittest.main()
