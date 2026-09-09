import torch

from sglang.kernels.ops.kvcache.cache_move import store_cache_4d


NUM_PAGES = 17
PAGE_SIZE = 8
HEAD_NUM = 16
HEAD_DIM = 128
V_HEAD_DIM = 96
NUM_WRITES = 64


def _write_mask(loc):
    page_id = torch.div(loc, PAGE_SIZE, rounding_mode="floor")
    token_in_page = loc % PAGE_SIZE
    mask = torch.zeros(
        (NUM_PAGES, PAGE_SIZE, HEAD_NUM, HEAD_DIM), device="cuda", dtype=torch.bool
    )
    mask[page_id, token_in_page] = True
    return mask


def _reference_write(k_view, v_view, loc, cache_k, cache_v):
    page_id = torch.div(loc, PAGE_SIZE, rounding_mode="floor")
    token_in_page = loc % PAGE_SIZE
    k_view[page_id, token_in_page] = cache_k
    v_view[page_id, token_in_page] = cache_v



def _check(layout, loc, cache_k, cache_v):
    torch.manual_seed(29864 + (0 if layout == "contiguous" else 1))
    if layout == "contiguous":
        k_storage = torch.randn(
            (NUM_PAGES, PAGE_SIZE, HEAD_NUM, HEAD_DIM),
            device="cuda",
            dtype=torch.float32,
        ).to(torch.bfloat16)
        v_storage = torch.randn(
            (NUM_PAGES, PAGE_SIZE, HEAD_NUM, V_HEAD_DIM),
            device="cuda",
            dtype=torch.float32,
        ).to(torch.bfloat16)
        k_view = k_storage
        v_view = v_storage
    else:
        k_storage = torch.randn(
            (NUM_PAGES * 2, PAGE_SIZE * 2, HEAD_NUM, HEAD_DIM),
            device="cuda",
            dtype=torch.float32,
        ).to(torch.bfloat16)
        v_storage = torch.randn(
            (NUM_PAGES * 2, PAGE_SIZE * 2, HEAD_NUM, V_HEAD_DIM),
            device="cuda",
            dtype=torch.float32,
        ).to(torch.bfloat16)
        k_view = k_storage[0::2, 0::2, :, :]
        v_view = v_storage[0::2, 0::2, :, :]

    k_reference = k_storage.clone()
    v_reference = v_storage.clone()
    if layout == "contiguous":
        k_reference_view = k_reference
        v_reference_view = v_reference
    else:
        k_reference_view = k_reference[0::2, 0::2, :, :]
        v_reference_view = v_reference[0::2, 0::2, :, :]
    cache_k_before = cache_k.clone()
    cache_v_before = cache_v.clone()
    assert k_view.stride(-1) == 1 and k_view.stride(-2) == HEAD_DIM
    assert v_view.stride(-1) == 1 and v_view.stride(-2) == V_HEAD_DIM

    store_cache_4d(k_view, v_view, cache_k, cache_v, loc, PAGE_SIZE)
    if layout == "contiguous":
        _reference_write(k_reference, v_reference, loc, cache_k, cache_v)
    else:
        _reference_write(
            k_reference[0::2, 0::2, :, :],
            v_reference[0::2, 0::2, :, :],
            loc,
            cache_k,
            cache_v,
        )

    k_mask = _write_mask(loc)
    v_mask = torch.zeros(
        (NUM_PAGES, PAGE_SIZE, HEAD_NUM, V_HEAD_DIM), device="cuda", dtype=torch.bool
    )
    page_id = torch.div(loc, PAGE_SIZE, rounding_mode="floor")
    token_in_page = loc % PAGE_SIZE
    v_mask[page_id, token_in_page] = True

    gates = {
        "k_written_equal": torch.equal(k_view[k_mask], k_reference_view[k_mask]),
        "v_written_equal": torch.equal(v_view[v_mask], v_reference_view[v_mask]),
        "k_unchanged_equal": torch.equal(k_view[~k_mask], k_reference_view[~k_mask]),
        "v_unchanged_equal": torch.equal(v_view[~v_mask], v_reference_view[~v_mask]),
        "k_whole_equal": torch.equal(k_storage, k_reference),
        "v_whole_equal": torch.equal(v_storage, v_reference),
        "source_unchanged": torch.equal(cache_k, cache_k_before)
        and torch.equal(cache_v, cache_v_before),
    }
    k_max_diff = (k_storage.float() - k_reference.float()).abs().max().item()
    v_max_diff = (v_storage.float() - v_reference.float()).abs().max().item()

    print(f"layout={layout}")
    print(f"k_shape={tuple(k_view.shape)} k_stride={k_view.stride()}")
    print(f"v_shape={tuple(v_view.shape)} v_stride={v_view.stride()}")
    print(f"n={NUM_WRITES} page_size={PAGE_SIZE} dtype={k_view.dtype}")
    print(f"gates={gates}")
    print(f"max_abs_diff=k:{k_max_diff},v:{v_max_diff}")
    assert all(gates.values())
    assert k_max_diff == 0.0 and v_max_diff == 0.0


def main():
    torch.manual_seed(29864)
    loc = torch.randperm(NUM_PAGES * PAGE_SIZE, device="cuda", dtype=torch.int64)[
        :NUM_WRITES
    ]
    cache_k = torch.randn(
        (NUM_WRITES, HEAD_NUM, HEAD_DIM), device="cuda", dtype=torch.float32
    ).to(torch.bfloat16)
    cache_v = torch.randn(
        (NUM_WRITES, HEAD_NUM, V_HEAD_DIM), device="cuda", dtype=torch.float32
    ).to(torch.bfloat16)
    _check("contiguous", loc, cache_k, cache_v)
    _check("outer_strided", loc, cache_k, cache_v)
    print("RESULT: PASS (2/2 layouts; all byte-identity gates true; max abs diff 0)")


if __name__ == "__main__":
    main()
