"""Compile and numerically check the historical store_cache_4d Triton kernel.

This is an issue-investigation artifact, not production code. The kernel is
copied from the parent of sglang commit 4bea51d885538466caef09223e8beb1c307b4489,
which removed it when dense unified-memory KV views replaced the 4-D layout.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def store_cache_4d_kernel(
    k_view_ptr,
    v_view_ptr,
    cache_k_ptr,
    cache_v_ptr,
    loc_ptr,
    stride_k_page,
    stride_k_tok,
    stride_v_page,
    stride_v_tok,
    stride_src_k_row,
    stride_src_v_row,
    K_ROW_DIM: tl.constexpr,
    V_ROW_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_kv = tl.program_id(2)
    loc = tl.load(loc_ptr + pid_n).to(tl.int64)
    if PAGE_SIZE == 1:
        page_id = loc
        tok_in_p = tl.zeros([], dtype=tl.int64)
    else:
        page_id = loc // PAGE_SIZE
        tok_in_p = loc % PAGE_SIZE
    base_off = pid_b * BLOCK + tl.arange(0, BLOCK)
    if pid_kv == 0:
        mask = base_off < K_ROW_DIM
        src_ptr = cache_k_ptr + pid_n * stride_src_k_row + base_off
        dst_ptr = (
            k_view_ptr
            + page_id * stride_k_page
            + tok_in_p * stride_k_tok
            + base_off
        )
    else:
        mask = base_off < V_ROW_DIM
        src_ptr = cache_v_ptr + pid_n * stride_src_v_row + base_off
        dst_ptr = (
            v_view_ptr
            + page_id * stride_v_page
            + tok_in_p * stride_v_tok
            + base_off
        )
    src = tl.load(src_ptr, mask=mask)
    tl.store(dst_ptr, src, mask=mask)


def run_case(page_size, loc_dtype, head_dim, v_head_dim):
    torch.manual_seed(1234 + page_size + head_dim + v_head_dim)
    num_pages, head_num, n = 7, 3, 5
    k = torch.zeros(
        (num_pages, page_size, head_num, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.zeros(
        (num_pages, page_size, head_num, v_head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    ref_k, ref_v = torch.zeros_like(k), torch.zeros_like(v)
    second_loc = 1 if page_size == 1 else page_size - 1
    loc = torch.tensor(
        [0, second_loc, 2 * page_size, 4 * page_size - 1, num_pages * page_size - 1],
        device="cuda",
        dtype=loc_dtype,
    )
    src_k = torch.randn((n, head_num, head_dim), device="cuda").to(torch.bfloat16)
    src_v = torch.randn((n, head_num, v_head_dim), device="cuda").to(
        torch.bfloat16
    )
    grid = (n, triton.cdiv(max(head_num * head_dim, head_num * v_head_dim), 128), 2)
    store_cache_4d_kernel[grid](
        k,
        v,
        src_k,
        src_v,
        loc,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        src_k.stride(0),
        src_v.stride(0),
        K_ROW_DIM=head_num * head_dim,
        V_ROW_DIM=head_num * v_head_dim,
        PAGE_SIZE=page_size,
        BLOCK=128,
        num_warps=4,
    )
    ref_k[loc // page_size, loc % page_size] = src_k
    ref_v[loc // page_size, loc % page_size] = src_v
    torch.cuda.synchronize()
    assert torch.equal(k, ref_k)
    assert torch.equal(v, ref_v)
    print(
        f"PASS page_size={page_size} loc={loc_dtype} "
        f"head_dim={head_dim} v_head_dim={v_head_dim} exact=True"
    )


if __name__ == "__main__":
    props = torch.cuda.get_device_properties(0)
    print(
        "gpu",
        torch.cuda.get_device_name(0),
        "arch",
        props.gcnArchName,
        "torch",
        torch.__version__,
        "hip",
        torch.version.hip,
        "triton",
        triton.__version__,
    )
    run_case(1, torch.int64, 64, 64)
    run_case(4, torch.int32, 65, 33)
    print("ALL_PASS no PassManager::run or TritonAMDGPUCanonicalizePointers failure")
