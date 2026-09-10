import sys

import pytest
import torch
from sgl_kernel import tree_speculative_sampling_target_only


def reference_tree_speculative_sampling(
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    threshold_single,
    threshold_acc,
    num_spec_steps,
):
    batch_size, num_draft_tokens = candidates.shape
    vocab_size = target_probs.shape[-1]

    candidates_cpu = candidates.cpu()
    retrive_index_cpu = retrive_index.cpu()
    retrive_next_token_cpu = retrive_next_token.cpu()
    retrive_next_sibling_cpu = retrive_next_sibling.cpu()
    uniform_cpu = uniform_samples.cpu()
    final_uniform_cpu = uniform_samples_for_final_sampling.cpu()
    target_cpu = target_probs.cpu().clone()
    draft_cpu = draft_probs.cpu().clone()

    predicts = torch.full((batch_size * num_draft_tokens,), -1, dtype=torch.int32)
    accept_index = torch.full((batch_size, num_spec_steps), -1, dtype=torch.int32)
    accept_token_num = torch.zeros(batch_size, dtype=torch.int32)
    effective_threshold_acc = max(float(threshold_acc), 1e-9)

    for batch_index in range(batch_size):
        current_row = 0
        coin = float(uniform_cpu[batch_index, 0])
        probability_accumulator = 0.0
        last_accepted_index = int(retrive_index_cpu[batch_index, 0])
        accept_index[batch_index, 0] = last_accepted_index
        accepted_count = 0
        current_index = 0

        for _ in range(1, num_spec_steps):
            current_index = int(retrive_next_token_cpu[batch_index, current_index])
            while current_index != -1:
                draft_token = int(candidates_cpu[batch_index, current_index])
                target_probability = float(
                    target_cpu[batch_index, current_row, draft_token]
                )
                probability_accumulator += target_probability

                if (
                    coin <= probability_accumulator / effective_threshold_acc
                    or target_probability >= threshold_single
                ):
                    current_row = current_index
                    coin = float(uniform_cpu[batch_index, current_index])
                    predicts[last_accepted_index] = draft_token
                    accepted_count += 1
                    accept_index[batch_index, accepted_count] = int(
                        retrive_index_cpu[batch_index, current_index]
                    )
                    last_accepted_index = int(
                        retrive_index_cpu[batch_index, current_index]
                    )
                    break

                draft_cpu[batch_index, current_row, draft_token] = target_probability
                current_index = int(
                    retrive_next_sibling_cpu[batch_index, current_index]
                )

            if current_index == -1:
                break

        accept_token_num[batch_index] = accepted_count
        final_coin = float(final_uniform_cpu[batch_index])

        if accepted_count == num_spec_steps - 1:
            residual = target_cpu[batch_index, current_row].clone()
        else:
            residual = (
                target_cpu[batch_index, current_row]
                - draft_cpu[batch_index, current_row]
            ).clamp_min(0)

        residual_mass = float(residual.sum())
        if residual_mass > 0:
            inclusive_cdf = torch.cumsum(residual, dim=0)
            crossings = torch.nonzero(inclusive_cdf > final_coin * residual_mass)
            sampled_token = int(crossings[0, 0]) if crossings.numel() else vocab_size - 1
        else:
            sampled_token = vocab_size - 1

        predicts[last_accepted_index] = sampled_token

    return predicts, accept_index, accept_token_num, draft_cpu


def run_tree_speculative_sampling(
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    threshold_single=1.0,
    threshold_acc=1.0,
    num_spec_steps=2,
    deterministic=True,
):
    device = candidates.device
    batch_size, num_draft_tokens = candidates.shape
    predicts = torch.full(
        (batch_size * num_draft_tokens,), -1, dtype=torch.int32, device=device
    )
    accept_index = torch.full(
        (batch_size, num_spec_steps), -1, dtype=torch.int32, device=device
    )
    accept_token_num = torch.zeros(batch_size, dtype=torch.int32, device=device)

    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=deterministic,
    )
    return predicts, accept_index, accept_token_num, draft_probs


def assert_matches_reference(outputs, reference_outputs):
    predicts, accept_index, accept_token_num, draft_probs = outputs
    expected = reference_outputs
    torch.testing.assert_close(predicts.cpu(), expected[0])
    torch.testing.assert_close(accept_index.cpu(), expected[1])
    torch.testing.assert_close(accept_token_num.cpu(), expected[2])
    torch.testing.assert_close(draft_probs.cpu(), expected[3])


def test_rejection_residual_sampling_and_padded_rows():
    device = "cuda"
    candidates = torch.tensor([[0, 0, 0, 0]], dtype=torch.int64, device=device)
    retrive_index = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64, device=device)
    retrive_next_token = torch.tensor([[1, -1, -1, -1]], dtype=torch.int64, device=device)
    retrive_next_sibling = torch.tensor(
        [[-1, -1, -1, -1]], dtype=torch.int64, device=device
    )
    uniform_samples = torch.tensor([[0.5, 0.0, 0.0, 0.0]], device=device)
    final_uniform_samples = torch.tensor([[0.25]], device=device)
    target_probs = torch.tensor(
        [[[0.25, 0.5, 0.25], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        device=device,
    )
    draft_probs = torch.zeros_like(target_probs)
    initial_draft_probs = draft_probs.clone()

    outputs = run_tree_speculative_sampling(
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        final_uniform_samples,
        target_probs,
        draft_probs,
    )
    reference_outputs = reference_tree_speculative_sampling(
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        final_uniform_samples,
        target_probs,
        initial_draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        num_spec_steps=2,
    )

    assert outputs[0].tolist() == [1, -1, -1, -1]
    assert outputs[1].tolist() == [[0, -1]]
    assert outputs[2].tolist() == [0]
    assert_matches_reference(outputs, reference_outputs)


def test_zero_and_unit_acceptance_probabilities():
    device = "cuda"
    candidates = torch.tensor(
        [[0, 1, 0, 1], [0, 1, 0, 1]], dtype=torch.int64, device=device
    )
    retrive_index = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int64, device=device
    )
    retrive_next_token = torch.tensor(
        [[1, -1, 3, -1]], dtype=torch.int64, device=device
    ).repeat(2, 1)
    retrive_next_sibling = torch.tensor(
        [[-1, 2, -1, -1]], dtype=torch.int64, device=device
    ).repeat(2, 1)
    uniform_samples = torch.tensor(
        [[0.5, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]], device=device
    )
    final_uniform_samples = torch.tensor([[0.5], [0.25]], device=device)
    target_probs = torch.tensor(
        [
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        device=device,
    )
    draft_probs = torch.zeros_like(target_probs)
    initial_draft_probs = draft_probs.clone()

    outputs = run_tree_speculative_sampling(
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        final_uniform_samples,
        target_probs,
        draft_probs,
    )
    reference_outputs = reference_tree_speculative_sampling(
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        final_uniform_samples,
        target_probs,
        initial_draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        num_spec_steps=2,
    )

    assert outputs[0].tolist() == [2, -1, -1, -1, 0, -1, 2, -1]
    assert outputs[1].tolist() == [[0, -1], [4, 6]]
    assert outputs[2].tolist() == [0, 1]
    assert_matches_reference(outputs, reference_outputs)


def test_zero_residual_mass():
    device = "cuda"
    candidates = torch.tensor([[0, 0]], dtype=torch.int64, device=device)
    retrive_index = torch.tensor([[0, 1]], dtype=torch.int64, device=device)
    retrive_next_token = torch.tensor([[1, -1]], dtype=torch.int64, device=device)
    retrive_next_sibling = torch.tensor([[-1, -1]], dtype=torch.int64, device=device)
    uniform_samples = torch.tensor([[0.5, 0.0]], device=device)
    final_uniform_samples = torch.tensor([[0.5]], device=device)
    target_probs = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], device=device
    )
    draft_probs = torch.zeros_like(target_probs)
    initial_draft_probs = draft_probs.clone()

    outputs = run_tree_speculative_sampling(
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        final_uniform_samples,
        target_probs,
        draft_probs,
    )
    reference_outputs = reference_tree_speculative_sampling(
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        final_uniform_samples,
        target_probs,
        initial_draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        num_spec_steps=2,
    )

    assert outputs[0].tolist() == [0, 2]
    assert outputs[1].tolist() == [[0, 1]]
    assert outputs[2].tolist() == [1]
    assert_matches_reference(outputs, reference_outputs)


def test_reproducible_uniform_distribution():
    device = "cuda"
    seed = 30344
    sample_count = 4096
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    uniform_samples = torch.rand(
        (sample_count, 2), generator=generator, dtype=torch.float32
    ).to(device)
    final_uniform_samples = uniform_samples[:, 1:2].contiguous()
    uniform_samples = uniform_samples.contiguous()

    candidates = torch.zeros((sample_count, 2), dtype=torch.int64, device=device)
    retrive_index = torch.arange(
        sample_count * 2, device=device, dtype=torch.int64
    ).reshape(sample_count, 2)
    retrive_next_token = torch.tensor([[1, -1]], device=device).repeat(sample_count, 1)
    retrive_next_sibling = torch.tensor([[-1, -1]], device=device).repeat(sample_count, 1)
    root_probs = torch.tensor([0.25, 0.5, 0.25], device=device)
    bonus_probs = torch.tensor([0.25, 0.25, 0.5], device=device)
    target_probs = torch.stack(
        [root_probs.repeat(sample_count, 1), bonus_probs.repeat(sample_count, 1)], dim=1
    )
    draft_probs = torch.zeros_like(target_probs)
    initial_draft_probs = draft_probs.clone()

    outputs = run_tree_speculative_sampling(
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        final_uniform_samples,
        target_probs,
        draft_probs,
        deterministic=False,
    )
    reference_outputs = reference_tree_speculative_sampling(
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        final_uniform_samples,
        target_probs,
        initial_draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        num_spec_steps=2,
    )

    assert_matches_reference(outputs, reference_outputs)
    sampled_tokens = outputs[0][::2].cpu()
    empirical_distribution = torch.bincount(sampled_tokens, minlength=3).float() / sample_count
    expected_distribution = torch.tensor([0.25, 0.5, 0.25])
    torch.testing.assert_close(
        empirical_distribution,
        expected_distribution,
        atol=0.03,
        rtol=0.0,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
