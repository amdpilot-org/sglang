# Commands run

All work was under `/job/sglang` unless noted. Failed or unsupported commands are
included for completeness.

```bash
git clone --depth=200 https://github.com/amdpilot-org/sglang.git /job/sglang
git rev-parse HEAD
git branch --show-current

gh issue view 36207 --repo sgl-project/sglang \
  --json title,body,state,author,createdAt,comments,url
gh pr view 36003 --repo sgl-project/sglang \
  --json title,state,mergedAt,headRefName,headRefOid,baseRefName,body,url,commits

rg -n "reserved_skip_index|set_mla_kv_buffer|fused_qk_rope_cat_and_cache_mla|save_kv_cache|store_cache\\(" --glob '!*.md'
rg -n "fused_qk_rope_cat_and_cache_mla" python test
rg -n "^def |reserved_skip_index|DCP_WORLD_SIZE|DCP_RANK" \
  python/sglang/kernels/ops/kvcache/mla_buffer.py
rg -n -C 12 "set_mla_kv_buffer_triton|set_mla_kv_buffer\\(" python/sglang/srt/mem_cache

export PYTHONPATH=/job/sglang/python
export PYTORCH_TRITON_CACHE=/tmp/j-c53e5e83ebd6-triton-cache-1
export TRITON_CACHE_DIR=/tmp/j-c53e5e83ebd6-triton-cache-1
/opt/venv/bin/python -m pytest \
  test/registered/kernels/ops/kvcache/test_set_mla_kv_buffer.py \
  -q -k 'reserved_skip_index or skip_disabled'

/opt/venv/bin/python -m py_compile \
  reports/j-c53e5e83ebd6/validate_mla_slot_zero.py

export PYTORCH_TRITON_CACHE=/tmp/j-c53e5e83ebd6-triton-cache-7
export TRITON_CACHE_DIR=/tmp/j-c53e5e83ebd6-triton-cache-7
/opt/venv/bin/python reports/j-c53e5e83ebd6/validate_mla_slot_zero.py \
  2>&1 | tee reports/j-c53e5e83ebd6/validation.log

rocm-smi --showproductname --showserial --showuniqueid --showbus
cat /etc/os-release
git status --short
git diff --check
git diff --stat
git rev-parse HEAD
```

Earlier validation runs used the same command with cache directories
`/tmp/j-c53e5e83ebd6-triton-cache-2` through
`/tmp/j-c53e5e83ebd6-triton-cache-6`. They exposed and fixed only harness issues:
an indentation error, an uninitialized DCP parallel context, uint8 source
sentinels, and a dtype call. The final run above is the recorded raw result.

The unsupported TMA probe was performed by the final validation script through
`can_use_set_mla_kv_buffer(256, 128)`. It attempted a job-private native build
under `/job/.cache/sglang/jit/gfx942` and failed because `cuda/ptx` was missing.
