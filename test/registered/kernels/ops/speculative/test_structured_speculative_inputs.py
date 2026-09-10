from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch
from sgl_kernel import verify_tree_greedy
from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.test.test_utils import CustomTestCase


BATCH_SIZE = 8
NUM_DRAFT_TOKENS = 5
VOCAB_SIZE = 64
SENTINEL = -123
CASE_NAMES = [
    "zeros",
    "tiny",
    "mixed_magnitudes",
    "cancellation",
    "skewed",
    "state",
]


def _make_tree_metadata(device):
    retrieve_index = torch.arange(
        BATCH_SIZE * NUM_DRAFT_TOKENS, dtype=torch.long, device=device
    ).view(BATCH_SIZE, NUM_DRAFT_TOKENS)
    candidates = torch.arange(
        NUM_DRAFT_TOKENS, dtype=torch.long, device=device
    ).repeat(BATCH_SIZE, 1)

    verifier_next_token = torch.tensor(
        [[1, 2, 3, -1, -1]],
        dtype=torch.long,
        device=device,
    ).repeat(BATCH_SIZE, 1)
    verifier_next_sibling = torch.tensor(
        [[-1, 4, -1, -1, -1]],
        dtype=torch.long,
        device=device,
    ).repeat(BATCH_SIZE, 1)

    chain_next_token = torch.tensor(
        [[1, 2, 3, 4, -1]],
        dtype=torch.long,
        device=device,
    ).repeat(BATCH_SIZE, 1)
    chain_next_sibling = torch.full(
        (BATCH_SIZE, NUM_DRAFT_TOKENS), -1, dtype=torch.long, device=device
    )

    return {
        "retrieve_index": retrieve_index,
        "candidates": candidates,
        "verifier_next_token": verifier_next_token,
        "verifier_next_sibling": verifier_next_sibling,
        "chain_next_token": chain_next_token,
        "chain_next_sibling": chain_next_sibling,
    }


def _make_structured_probs(case_name):
    target_probs = torch.zeros(
        (BATCH_SIZE, NUM_DRAFT_TOKENS, VOCAB_SIZE), dtype=torch.float32
    )
    draft_probs = torch.zeros(
        (BATCH_SIZE, NUM_DRAFT_TOKENS - 1, VOCAB_SIZE), dtype=torch.float32
    )

    if case_name == "zeros":
        pass
    elif case_name == "tiny":
        for row in range(NUM_DRAFT_TOKENS):
            target_probs[:, row, row + 1] = 1e-40
    elif case_name == "mixed_magnitudes":
        values = [1e-30, 1e-10, 1.0, 1e-5, 1e-20]
        for row, value in enumerate(values):
            target_probs[:, row, row + 1] = value
    elif case_name == "cancellation":
        target_values = [0.7, 0.6, 0.5, 0.4, 1.0]
        draft_values = [0.2, 0.3, 0.4, 0.5]
        for row, value in enumerate(target_values):
            target_probs[:, row, row + 1] = value
        for row, value in enumerate(draft_values):
            draft_probs[:, row, row + 1] = value
    elif case_name == "skewed":
        target_values = [0.9, 0.8, 0.7, 0.6, 1.0]
        for row, value in enumerate(target_values):
            target_probs[:, row, row + 1] = value
        for row in range(NUM_DRAFT_TOKENS - 1):
            draft_probs[:, row, 10 + row] = 0.5
    elif case_name == "state":
        accept_counts = [0, 1, 2, 3, 4, 0, 1, 2]
        for batch, accept_count in enumerate(accept_counts):
            for row in range(NUM_DRAFT_TOKENS - 1):
                if row < accept_count:
                    target_probs[batch, row, row + 1] = 1.0
                else:
                    target_probs[batch, row, 60] = 1.0
            target_probs[batch, NUM_DRAFT_TOKENS - 1, 5] = 1.0
    else:
        raise ValueError(f"unknown case: {case_name}")

    return target_probs, draft_probs


def _reference_verify_tree_greedy(
    candidates,
    retrieve_index,
    retrieve_next_token,
    retrieve_next_sibling,
    target_predict,
):
    predicts = torch.full(
        (BATCH_SIZE * NUM_DRAFT_TOKENS,), SENTINEL, dtype=torch.int32
    )
    accept_index = torch.full(
        (BATCH_SIZE, NUM_DRAFT_TOKENS), SENTINEL, dtype=torch.int32
    )
    accept_token_num = torch.full((BATCH_SIZE,), SENTINEL, dtype=torch.int32)

    for batch in range(BATCH_SIZE):
        last_accept_retrieve_idx = int(retrieve_index[batch, 0])
        accept_index[batch, 0] = last_accept_retrieve_idx
        num_accept = 0
        current_index = 0
        should_continue = True

        for _ in range(1, NUM_DRAFT_TOKENS):
            if not should_continue:
                break
            current_index = int(retrieve_next_token[batch, current_index])
            target_row = last_accept_retrieve_idx // NUM_DRAFT_TOKENS
            target_col = last_accept_retrieve_idx % NUM_DRAFT_TOKENS
            target_token = int(target_predict[target_row, target_col])
            found_match = False

            while current_index != -1:
                draft_index = int(retrieve_index[batch, current_index])
                draft_token = int(candidates[batch, current_index])
                if draft_token == target_token:
                    predicts[last_accept_retrieve_idx] = target_token
                    num_accept += 1
                    accept_index[batch, num_accept] = draft_index
                    last_accept_retrieve_idx = draft_index
                    found_match = True
                    break
                current_index = int(retrieve_next_sibling[batch, current_index])

            if not found_match:
                should_continue = False

        accept_token_num[batch] = num_accept
        target_row = last_accept_retrieve_idx // NUM_DRAFT_TOKENS
        target_col = last_accept_retrieve_idx % NUM_DRAFT_TOKENS
        predicts[last_accept_retrieve_idx] = target_predict[target_row, target_col]

    return predicts, accept_index, accept_token_num


def _reference_chain_speculative_sampling(
    candidates,
    retrieve_index,
    target_probs,
    draft_probs,
    uniform_samples,
    final_samples,
):
    predicts = torch.full(
        (BATCH_SIZE * NUM_DRAFT_TOKENS,), SENTINEL, dtype=torch.int32
    )
    accept_index = torch.full(
        (BATCH_SIZE, NUM_DRAFT_TOKENS), SENTINEL, dtype=torch.int32
    )
    accept_token_num = torch.full((BATCH_SIZE,), SENTINEL, dtype=torch.int32)

    for batch in range(BATCH_SIZE):
        root_global_idx = int(retrieve_index[batch, 0])
        accept_index[batch, 0] = root_global_idx
        last_accepted_global_idx = root_global_idx
        num_accept = 0
        current_prob_row = 0
        continue_verifying = True

        for step in range(1, NUM_DRAFT_TOKENS):
            draft_token = int(candidates[batch, step])
            target_prob = float(target_probs[batch, current_prob_row, draft_token])
            draft_prob = float(draft_probs[batch, current_prob_row, draft_token])
            coin = float(uniform_samples[batch, step - 1])
            if coin * draft_prob < target_prob:
                num_accept += 1
                current_prob_row = step
                predicts[last_accepted_global_idx] = draft_token
                current_global_idx = int(retrieve_index[batch, step])
                accept_index[batch, num_accept] = current_global_idx
                last_accepted_global_idx = current_global_idx
            else:
                continue_verifying = False
                break

        accept_token_num[batch] = num_accept
        all_drafts_accepted = continue_verifying
        coin_final = float(final_samples[batch])

        if all_drafts_accepted:
            residual = target_probs[batch, current_prob_row]
        else:
            residual = torch.clamp(
                target_probs[batch, current_prob_row]
                - draft_probs[batch, current_prob_row],
                min=0.0,
            )

        norm_sum = float(residual.sum())
        target_u = coin_final * norm_sum
        cumulative_sum = 0.0
        final_token = VOCAB_SIZE - 1
        for token, value in enumerate(residual.tolist()):
            cumulative_sum += value
            if cumulative_sum > target_u:
                final_token = token
                break

        predicts[last_accepted_global_idx] = final_token

    return predicts, accept_index, accept_token_num


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestStructuredSpeculativeInputs(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.device = torch.device("cuda")

    def test_structured_inputs_match_independent_references(self):
        metadata = _make_tree_metadata(self.device)
        uniform_samples = torch.full(
            (BATCH_SIZE, NUM_DRAFT_TOKENS - 1), 0.5, device=self.device
        )
        final_samples = torch.zeros(BATCH_SIZE, device=self.device)

        for case_name in CASE_NAMES:
            with self.subTest(case=case_name):
                target_probs_cpu, draft_probs_cpu = _make_structured_probs(case_name)
                self.assertTrue(torch.isfinite(target_probs_cpu).all())
                self.assertTrue(torch.isfinite(draft_probs_cpu).all())
                self.assertEqual(target_probs_cpu.dtype, torch.float32)
                self.assertEqual(draft_probs_cpu.dtype, torch.float32)

                target_probs = target_probs_cpu.to(self.device)
                draft_probs = draft_probs_cpu.to(self.device)
                target_predict_cpu = torch.argmax(target_probs_cpu, dim=-1)
                target_predict = target_predict_cpu.to(self.device)

                native_predicts = torch.full(
                    (BATCH_SIZE * NUM_DRAFT_TOKENS,),
                    SENTINEL,
                    dtype=torch.int32,
                    device=self.device,
                )
                native_accept_index = torch.full(
                    (BATCH_SIZE, NUM_DRAFT_TOKENS),
                    SENTINEL,
                    dtype=torch.int32,
                    device=self.device,
                )
                native_accept_token_num = torch.full(
                    (BATCH_SIZE,), SENTINEL, dtype=torch.int32, device=self.device
                )

                verify_tree_greedy(
                    predicts=native_predicts,
                    accept_index=native_accept_index,
                    accept_token_num=native_accept_token_num,
                    candidates=metadata["candidates"],
                    retrive_index=metadata["retrieve_index"],
                    retrive_next_token=metadata["verifier_next_token"],
                    retrive_next_sibling=metadata["verifier_next_sibling"],
                    target_predict=target_predict,
                )
                torch.cuda.synchronize()

                expected_native_predicts, expected_native_accept_index, expected_native_accept_token_num = _reference_verify_tree_greedy(
                    metadata["candidates"].cpu(),
                    metadata["retrieve_index"].cpu(),
                    metadata["verifier_next_token"].cpu(),
                    metadata["verifier_next_sibling"].cpu(),
                    target_predict_cpu,
                )
                self.assertTrue(
                    torch.equal(native_predicts.cpu(), expected_native_predicts)
                )
                self.assertTrue(
                    torch.equal(native_accept_index.cpu(), expected_native_accept_index)
                )
                self.assertTrue(
                    torch.equal(
                        native_accept_token_num.cpu(), expected_native_accept_token_num
                    )
                )

                chain_predicts = torch.full(
                    (BATCH_SIZE * NUM_DRAFT_TOKENS,),
                    SENTINEL,
                    dtype=torch.int32,
                    device=self.device,
                )
                chain_accept_index = torch.full(
                    (BATCH_SIZE, NUM_DRAFT_TOKENS),
                    SENTINEL,
                    dtype=torch.int32,
                    device=self.device,
                )
                chain_accept_token_num = torch.full(
                    (BATCH_SIZE,), SENTINEL, dtype=torch.int32, device=self.device
                )

                chain_speculative_sampling_triton(
                    predicts=chain_predicts,
                    accept_index=chain_accept_index,
                    accept_token_num=chain_accept_token_num,
                    candidates=metadata["candidates"],
                    retrive_index=metadata["retrieve_index"],
                    retrive_next_token=metadata["chain_next_token"],
                    retrive_next_sibling=metadata["chain_next_sibling"],
                    uniform_samples=uniform_samples,
                    uniform_samples_for_final_sampling=final_samples,
                    target_probs=target_probs,
                    draft_probs=draft_probs,
                    threshold_single=1.0,
                    threshold_acc=1.0,
                    deterministic=True,
                )
                torch.cuda.synchronize()

                expected_chain_predicts, expected_chain_accept_index, expected_chain_accept_token_num = _reference_chain_speculative_sampling(
                    metadata["candidates"].cpu(),
                    metadata["retrieve_index"].cpu(),
                    target_probs_cpu,
                    draft_probs_cpu,
                    uniform_samples.cpu(),
                    final_samples.cpu(),
                )
                self.assertTrue(
                    torch.equal(chain_predicts.cpu(), expected_chain_predicts)
                )
                self.assertTrue(
                    torch.equal(chain_accept_index.cpu(), expected_chain_accept_index)
                )
                self.assertTrue(
                    torch.equal(
                        chain_accept_token_num.cpu(), expected_chain_accept_token_num
                    )
                )


if __name__ == "__main__":
    unittest.main()
