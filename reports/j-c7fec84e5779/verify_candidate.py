import importlib.util
import json
import time
from pathlib import Path

import torch

from build_candidate import load_candidate


REPORT_DIR = Path(__file__).resolve().parent
UPSTREAM_TEST = REPORT_DIR.parents[1] / "test/registered/spec/utils/test_build_eagle_tree_malformed.py"


def load_upstream_test():
    if not UPSTREAM_TEST.exists():
        raise FileNotFoundError(
            "Apply sgl-project/sglang pull 36201 before running this harness"
        )
    spec = importlib.util.spec_from_file_location("build_eagle_tree_malformed", UPSTREAM_TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_upstream_matrix(op):
    module = load_upstream_test()
    case = module.TestBuildEagleTreeMalformed("test_matrix")
    for name, parents in module.CASES.items():
        for mode in module.MODES:
            with case.subTest(case=name, mode=module.MODE_NAMES[mode]):
                case._check(name, parents, module.SELECTED, mode, op=op)
    cross = module.TestBuildEagleTreeMalformed("test_cross_row_lookup")
    for mode in module.MODES:
        with cross.subTest(mode=module.MODE_NAMES[mode]):
            cross._check(
                "cross_row",
                module.CROSS_ROW_PARENT,
                module.CROSS_ROW_SELECTED,
                mode,
                op=op,
            )
    return len(module.CASES) * len(module.MODES) + len(module.MODES)


def bytes_per_item(draft_token_num):
    if draft_token_num > 16:
        return 4
    if draft_token_num > 8:
        return 2
    return 1


def make_boundary_tree(draft_token_num):
    topk = 2
    depth = draft_token_num - 1
    parent_of = [0, 1, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    parent_of = parent_of[: draft_token_num - 1]
    selected = [0] + [2 * token for token in range(2, draft_token_num)]
    parent_list = [0] * (topk * (depth - 1) + 1)
    for token in range(2, draft_token_num):
        parent_list[token] = selected[parent_of[token - 1] - 1]
    return (
        torch.tensor([parent_list, parent_list], dtype=torch.int64),
        torch.tensor([selected, selected], dtype=torch.int64),
    )


def reference(parent_list, selected_index, seq_lens, topk, depth, draft_token_num):
    batch_size = parent_list.shape[0]
    positions = torch.zeros(batch_size * draft_token_num, dtype=torch.int64)
    retrieve_index = torch.full((batch_size, draft_token_num), -1, dtype=torch.int64)
    retrieve_next_token = torch.full((batch_size, draft_token_num), -1, dtype=torch.int64)
    retrieve_next_sibling = torch.full((batch_size, draft_token_num), -1, dtype=torch.int64)
    matrix = torch.zeros(
        batch_size, draft_token_num, draft_token_num, dtype=torch.bool
    )

    def find(selected, token_index):
        for position in range(draft_token_num - 1):
            if selected[position] == token_index:
                return position
        return -1

    for batch in range(batch_size):
        selected = selected_index[batch].tolist()
        parents = parent_list[batch].tolist()
        seq_len = int(seq_lens[batch])
        for token in range(draft_token_num - 1, 0, -1):
            retrieve_index[batch, token] = batch * draft_token_num + token
            parent_tb_index = selected[token - 1] // topk
            if parent_tb_index == 0:
                parent_position = 0
            else:
                parent_token = parents[parent_tb_index]
                found = find(selected, parent_token)
                if found < 0:
                    continue
                parent_position = found + 1
            if retrieve_next_token[batch, parent_position] == -1:
                retrieve_next_token[batch, parent_position] = token
            else:
                origin = int(retrieve_next_token[batch, parent_position])
                retrieve_next_token[batch, parent_position] = token
                retrieve_next_sibling[batch, token] = origin
        retrieve_index[batch, 0] = batch * draft_token_num

        for token in range(draft_token_num):
            matrix[batch, token, 0] = True
            if token == 0:
                positions[batch * draft_token_num] = seq_len
                continue
            position = 0
            current = token - 1
            while position < depth:
                position += 1
                matrix[batch, token, current + 1] = True
                if selected[current] // topk == 0:
                    break
                found = find(selected, parents[selected[current] // topk])
                if found < 0:
                    break
                current = found
            positions[batch * draft_token_num + token] = position + seq_len
    return positions, matrix, retrieve_index, retrieve_next_token, retrieve_next_sibling


def decode_mask(raw, batch_size, draft_token_num, seq_lens, mode):
    if mode == 1:
        return raw.reshape(batch_size, draft_token_num, draft_token_num).clone()
    if mode == 2:
        item_bytes = bytes_per_item(draft_token_num)
        decoded = torch.zeros(
            batch_size, draft_token_num, draft_token_num, dtype=torch.bool
        )
        for batch in range(batch_size):
            for token in range(draft_token_num):
                start = (batch * draft_token_num + token) * item_bytes
                item = raw[start : start + item_bytes]
                for column in range(draft_token_num):
                    decoded[batch, token, column] = bool(
                        int(item[column // 8]) >> (column % 8) & 1
                    )
        return decoded
    decoded = torch.zeros(
        batch_size, draft_token_num, draft_token_num, dtype=torch.bool
    )
    offset = 0
    for batch in range(batch_size):
        seq_len = int(seq_lens[batch])
        row_length = seq_len + draft_token_num
        for token in range(draft_token_num):
            base = offset + row_length * token + seq_len
            decoded[batch, token] = raw[base : base + draft_token_num]
        offset += row_length * draft_token_num
    return decoded


def run_mode(op, parent_list, selected_index, seq_lens, topk, depth, draft_token_num, mode):
    batch_size = parent_list.shape[0]
    if mode == 0:
        total = sum(int(length) + draft_token_num for length in seq_lens) * draft_token_num
        raw = torch.zeros(total, dtype=torch.bool, device="cuda")
    elif mode == 1:
        raw = torch.zeros(
            batch_size * draft_token_num * draft_token_num,
            dtype=torch.bool,
            device="cuda",
        )
    else:
        raw = torch.zeros(
            batch_size * draft_token_num * bytes_per_item(draft_token_num),
            dtype=torch.uint8,
            device="cuda",
        )
    positions = torch.zeros(
        batch_size * draft_token_num, dtype=torch.int64, device="cuda"
    )
    buffers = torch.full(
        (3, batch_size, draft_token_num), -1, dtype=torch.int64, device="cuda"
    )
    retrieve_index, retrieve_next_token, retrieve_next_sibling = buffers
    op(
        parent_list.cuda(),
        selected_index.cuda(),
        seq_lens.cuda(),
        raw,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        topk,
        depth,
        draft_token_num,
        mode,
    )
    torch.cuda.synchronize()
    return (
        positions.cpu(),
        raw.cpu(),
        retrieve_index.cpu(),
        retrieve_next_token.cpu(),
        retrieve_next_sibling.cpu(),
    )


def run_boundary_case(op, draft_token_num):
    topk = 2
    depth = draft_token_num - 1
    parent_list, selected_index = make_boundary_tree(draft_token_num)
    seq_lens = torch.tensor([7, 11], dtype=torch.int64)
    expected = reference(
        parent_list, selected_index, seq_lens, topk, depth, draft_token_num
    )
    mode_results = []
    for mode in range(3):
        got = run_mode(
            op,
            parent_list,
            selected_index,
            seq_lens,
            topk,
            depth,
            draft_token_num,
            mode,
        )
        got_mask = decode_mask(
            got[1], parent_list.shape[0], draft_token_num, seq_lens, mode
        )
        for label, actual, wanted in zip(
            ("positions", "tree_mask", "retrieve_index", "retrieve_next_token", "retrieve_next_sibling"),
            (got[0], got_mask, got[2], got[3], got[4]),
            expected,
        ):
            if not torch.equal(actual, wanted):
                raise AssertionError(
                    f"draft_token_num={draft_token_num} mode={mode} {label} mismatch"
                )
        if mode == 2:
            item_bytes = bytes_per_item(draft_token_num)
            padding_bits = 0
            for item in got[1].split(item_bytes):
                value = int.from_bytes(item.tolist(), "little")
                padding_bits += bin(value >> draft_token_num).count("1")
            if padding_bits:
                raise AssertionError(
                    f"draft_token_num={draft_token_num} padding bits are nonzero"
                )
        mode_results.append(mode)
    matrix = expected[1]
    for batch in range(matrix.shape[0]):
        for token in range(matrix.shape[1]):
            if bool(matrix[batch, token, token + 1 :].any()):
                raise AssertionError("mask violates causal column bounds")
    if not bool((expected[4] >= 0).any()):
        raise AssertionError("synthetic tree has no branch sibling")
    if draft_token_num >= 10 and not bool(matrix[:, :, 7:10].any()):
        raise AssertionError("mask does not span the first byte boundary")
    if draft_token_num >= 20 and not bool(matrix[:, :, 15:20].any()):
        raise AssertionError("mask does not span the second byte boundary")
    if bool((expected[0] > seq_lens.repeat_interleave(draft_token_num) + depth).any()):
        raise AssertionError("position exceeds depth bound")
    return mode_results


def main():
    started = time.perf_counter()
    module = load_candidate()
    op = module.build_tree_kernel_efficient
    upstream_cases = run_upstream_matrix(op)
    boundary_results = {
        str(tokens): run_boundary_case(op, tokens) for tokens in (10, 20)
    }
    elapsed = time.perf_counter() - started
    result = {
        "candidate_extension": module.__file__,
        "upstream_pr": 36201,
        "upstream_head_commit": "f5e6e0bcf5129fe439a4d9c5361a09f26721eb1f",
        "upstream_cases_passed": upstream_cases,
        "boundary_draft_token_nums": boundary_results,
        "checks": [
            "positions",
            "tree_mask",
            "retrieve_index",
            "retrieve_next_token",
            "retrieve_next_sibling",
            "bit_padding_zero",
            "causal_column_bounds",
            "branch_sibling_present",
            "position_depth_bound",
        ],
        "elapsed_seconds": elapsed,
        "status": "passed",
    }
    output = REPORT_DIR / "results.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
