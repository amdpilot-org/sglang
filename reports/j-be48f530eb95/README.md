# DSA trtllm query-row limit investigation

## Scope

This records the `gfx942` investigation for sgl-project/sglang issue 34947. The
production failure is specific to the CUDA/SM100 `trtllm-gen` sparse-MLA path:
sglang flattens extend tokens into query rows, and the kernel maps that row
count onto `gridDim.z`, whose CUDA launch limit is 65,535.

The tested upstream candidate is sgl-project/sglang PR 34948, commit
`7b7177ab9ab04fcde80c8b72f0aa394e1b7d1084`. It is preserved in this branch and
supplemented with an architecture-independent boundary test. No CUDA-specific
kernel was executed on ROCm.

## Environment

- Image requested: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Requested local image ID: `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- Runtime image metadata was not exposed; hostname was not used as image identity.
- Python: `/opt/venv/bin/python` (3.10.12)
- Torch: `2.9.1+rocm7.2.0.lw.git7e1940d4`
- Triton: `3.7.0+amd.rocm7.2.0.git89002410`
- GPU: one AMD Instinct MI300X, `gfx942`, unique ID `0x586343f382a69f1d`
- Native Aiter module: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`
- Native `sgl_kernel`: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`

## Controls

1. Installed-source baseline:
   `cd /sgl-workspace/sglang && /opt/venv/bin/python -m pytest test/registered/attention/test_triton_attention_kernels.py -q -k 'compact_grid'`
   Result: 1 passed; legacy and compact extend outputs satisfied
   `torch.allclose(rtol=1e-2, atol=1e-3)`. Outer wall time: 19.114 s.

2. Unsupported sparse controls:
   - `sgl_kernel.flash_mla` failed with `ImportError: cannot import name
     'flashmla_ops'` and reported a CUDA-driver requirement.
   - `sgl_kernel.sparse_flash_attn` had no registered `fwd_sparse`,
     `varlen_fwd_sparse`, or `convert_vertical_slash_indexes` ops on ROCm.
   - The stock DSA fixture used `page_size=64`; this ROCm Aiter build requires
     the legacy HIP `page_size=1` path.

3. Supported `gfx942` sparse control:
   A DSA fixture with `page_size=1`, prefix length 2048, one extend token, four
   heads, and top-k 128 ran through Aiter and matched the suite's independent
   PyTorch reference. Outer wall time: 31.146 s.

4. Boundary control:
   `aiter.mla.mla_decode_fwd` ran with 4, 65,535, and 65,536 query rows. Each
   full launch and a 32,768-row split launch matched a vectorized FP32
   reference within the DSA sparse gate (`atol=1.6e-1`,
   `rtol=1.6e-1`); observed maximum absolute differences were at most
   `2.44140625e-4`. Full and split outputs were bitwise identical. This
   demonstrates a supported AMD split path, but it does not execute or validate
   the CUDA-only trtllm-gen kernel.

5. Guard test:
   `PYTHONPATH=python /opt/venv/bin/python -m pytest test/registered/attention/unittests/dsa/test_dsa.py -q -k test_trtllm_query_row_limit_guard`
   Result: 1 passed. The test accepts 65,535 rows and requires 65,536 rows to
   raise `ValueError` before kernel launch.

## Limitations

- The SM100 trtllm-gen kernel cannot run on `gfx942`; no CUDA-specific kernel
  result is claimed from this host.
- The Aiter control validates a supported ROCm sparse-MLA path and a split
  strategy, not the upstream SM100 failure mode.
- The structural `cum_seq_lens_q` adoption from FlashInfer PR 3238 remains
  untested here because the installed ROCm stack has no FlashInfer module.
