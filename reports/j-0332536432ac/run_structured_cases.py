import importlib.util
import json
import os
import sys

import torch


TEST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "test",
    "registered",
    "kernels",
    "ops",
    "speculative",
    "test_structured_speculative_inputs.py",
)


def _load_test_module():
    spec = importlib.util.spec_from_file_location("structured_test", TEST_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    module = _load_test_module()
    device = torch.device("cuda")
    metadata = module._make_tree_metadata(device)
    uniform_samples = torch.full(
        (module.BATCH_SIZE, module.NUM_DRAFT_TOKENS - 1), 0.5, device=device
    )
    final_samples = torch.zeros(module.BATCH_SIZE, device=device)

    results = {}
    for case_name in module.CASE_NAMES:
        target_probs_cpu, draft_probs_cpu = module._make_structured_probs(case_name)
        target_probs = target_probs_cpu.to(device)
        draft_probs = draft_probs_cpu.to(device)
        target_predict_cpu = torch.argmax(target_probs_cpu, dim=-1)
        target_predict = target_predict_cpu.to(device)

        native_predicts = torch.full(
            (module.BATCH_SIZE * module.NUM_DRAFT_TOKENS,),
            module.SENTINEL,
            dtype=torch.int32,
            device=device,
        )
        native_accept_index = torch.full(
            (module.BATCH_SIZE, module.NUM_DRAFT_TOKENS),
            module.SENTINEL,
            dtype=torch.int32,
            device=device,
        )
        native_accept_token_num = torch.full(
            (module.BATCH_SIZE,), module.SENTINEL, dtype=torch.int32, device=device
        )
        module.verify_tree_greedy(
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

        chain_predicts = torch.full(
            (module.BATCH_SIZE * module.NUM_DRAFT_TOKENS,),
            module.SENTINEL,
            dtype=torch.int32,
            device=device,
        )
        chain_accept_index = torch.full(
            (module.BATCH_SIZE, module.NUM_DRAFT_TOKENS),
            module.SENTINEL,
            dtype=torch.int32,
            device=device,
        )
        chain_accept_token_num = torch.full(
            (module.BATCH_SIZE,), module.SENTINEL, dtype=torch.int32, device=device
        )
        module.chain_speculative_sampling_triton(
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

        results[case_name] = {
            "native_predicts": native_predicts.cpu().tolist(),
            "native_accept_index": native_accept_index.cpu().tolist(),
            "native_accept_token_num": native_accept_token_num.cpu().tolist(),
            "chain_predicts": chain_predicts.cpu().tolist(),
            "chain_accept_index": chain_accept_index.cpu().tolist(),
            "chain_accept_token_num": chain_accept_token_num.cpu().tolist(),
        }

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
