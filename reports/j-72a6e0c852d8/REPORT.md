# ImageEncodingStage mm_token_type_ids investigation

Status: **positive, candidate-validated**
Completion marker: **OPEN_TASK_REPORTED**

## Result

Current mirror commit `ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4` omits
`mm_token_type_ids` from both positive and classifier-free-guidance negative
`ImageEncodingStage` text-encoder calls. A tiny processor/encoder fixture using
a real `transformers.BatchFeature` and synthetic `image_grid_thw=[[1, 2, 2]]`
captured `mm_token_type_ids=None` at the actual `self.text_encoder(...)` call.

Upstream candidate pull request 34494, commit
`d504058bb895bbeb818c2e4748af184a0021cdd2` (parent
`cbd0271574e2543c4b3f25a493c4063249fbe56a`), fixes the omission by forwarding
the processor-provided tensor when present. The exact candidate commit passed
its three tests, and the same source change passed this investigation's four
enhanced tests. The delivery patch preserves the candidate's minimal plumbing
and adds checks for image-grid alignment and unchanged text-only behavior.

No full JoyAI or Qwen3-VL weights were downloaded. No upstream issue, pull
request, or comment was posted or modified.

## Environment identity

- Working clone: `/job/sglang`
- Mirror base: `amdpilot-org/sglang` `main` at
  `ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4`
- Candidate worktree: `/job/sglang-candidate`
- Candidate commit: `d504058bb895bbeb818c2e4748af184a0021cdd2`
- Python: `/opt/venv/bin/python`, Python `3.10.12`
- Working source path:
  `/job/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/image_encoding.py`
- Installed source context: `/sgl-workspace/sglang/python/sglang/__init__.py`
- Torch source: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Torch native module:
  `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- Torch HIP: `7.2.26015-fc0010cf6a`
- Transformers: `5.12.1` at
  `/opt/venv/lib/python3.10/site-packages/transformers/__init__.py`
- Operator-qualified image:
  `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Operator-provided local image ID:
  `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- Container hostname: `banff-cyxtera-cx57-4` (not used as image identity)
- Docker was unavailable inside the container, so the image ID above is the
  operator-provided identity rather than an independently inspected digest.

### GPU identity

`rocm-smi --showproductname --showdriverversion` raw result:

```text
Driver version: 6.19.14.31400000
GPU[0] Card Series: AMD Instinct MI300X
GPU[0] Card Model: 0x74a1
GPU[0] Card Vendor: Advanced Micro Devices, Inc. [AMD/ATI]
GPU[0] Card SKU: M3000108
GPU[0] Subsystem ID: 0x74a1
GPU[0] Device Rev: 0x00
GPU[0] Node ID: 8
GPU[0] GUID: 25408
GPU[0] GFX Version: gfx942
```

`rocminfo` reported ROCk module `6.19.14.31400000`, ROCm runtime `1.18`,
one CPU agent, and the assigned MI300X GPU. The GPU control observed
`torch.cuda.device_count() == 1` and device name `AMD Instinct MI300X`.

## Reproduction

The fixture uses:

- a real `transformers.BatchFeature`;
- `input_ids` of shape `(1, 7)`;
- synthetic image metadata `image_grid_thw=[[1, 2, 2]]`;
- `pixel_values` of shape `(1, 3, 2, 2)`;
- `mm_token_type_ids=[[0, 1, 1, 1, 1, 0, 0]]` for the positive path;
- `mm_token_type_ids=[[0, 3, 3, 3, 3, 0, 0]]` for the negative path;
- a strict legacy encoder signature without `mm_token_type_ids`;
- a text-only request with `condition_image=None`.

The fixture calls the actual `ImageEncodingStage.forward` and the actual
`self.text_encoder(...)` operation. It does not replace the stage with an
imagined expansion.

### Current source

Command:

```bash
cd /job/sglang
TORCHDYNAMO_DISABLE=1 PYTHONPATH=/job/sglang/python \
  /opt/venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_image_encoding_stage.py -vv
```

Raw result before the fix:

```text
test_forwards_mm_token_type_ids_to_image_edit_text_encoder[False] FAILED
test_forwards_mm_token_type_ids_to_image_edit_text_encoder[True] FAILED
test_omits_mm_token_type_ids_when_processor_does_not_return_them PASSED
test_text_only_request_returns_without_encoding PASSED
=================== 2 failed, 2 passed, 2 warnings in 0.79s ===================
```

Both forwarding failures captured:

```text
call = {'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1]]),
        'image_grid_thw': tensor([[1, 2, 2]]),
        'input_ids': tensor([[1, 2, 3, 4, 5, 6, 7]]),
        'mm_token_type_ids': None, ...}
assert None is not None
```

This directly reproduces the omission: the processor supplied the tensor, but
the current text-encoder call did not forward it.

### Candidate commit

Command against the exact fetched candidate commit:

```bash
cd /job/sglang-candidate
TORCHDYNAMO_DISABLE=1 PYTHONPATH=/job/sglang-candidate/python \
  /opt/venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_image_encoding_stage.py -vv
```

Raw result:

```text
test_forwards_mm_token_type_ids_to_image_edit_text_encoder[False] PASSED
test_forwards_mm_token_type_ids_to_image_edit_text_encoder[True] PASSED
test_omits_mm_token_type_ids_when_processor_does_not_return_them PASSED
=========================== 3 passed, 2 warnings in 0.61s ===================
```

The enhanced four-test fixture was also run with the candidate source first on
`PYTHONPATH`:

```text
test_forwards_mm_token_type_ids_to_image_edit_text_encoder[False] PASSED
test_forwards_mm_token_type_ids_to_image_edit_text_encoder[True] PASSED
test_omits_mm_token_type_ids_when_processor_does_not_return_them PASSED
test_text_only_request_returns_without_encoding PASSED
=========================== 4 passed, 2 warnings in 0.66s ===================
```

The enhanced checks assert:

- positive and negative `mm_token_type_ids` are forwarded exactly;
- token-type shape matches `input_ids`;
- four marked image positions match `prod(image_grid_thw)`;
- dtype is `torch.long` and device is preserved;
- processors that omit the optional field retain the old call shape;
- text-only requests return the same batch without calling processor or encoder.

## gfx942 embedding control

Command:

```bash
/opt/venv/bin/python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"hip={torch.version.hip}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"device_count={torch.cuda.device_count()}")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected one assigned GPU, got {torch.cuda.device_count()}")
device = torch.device("cuda:0")
print(f"device={device}")
print(f"device_name={torch.cuda.get_device_name(device)}")

class TinyTokenTypedEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_embeddings = torch.nn.Embedding(8, 4)
        self.mm_token_type_embeddings = torch.nn.Embedding(2, 4)

    def forward(self, input_ids, mm_token_type_ids=None):
        hidden_states = self.input_embeddings(input_ids)
        if mm_token_type_ids is None:
            return hidden_states
        return hidden_states + self.mm_token_type_embeddings(mm_token_type_ids)

torch.manual_seed(0)
encoder = TinyTokenTypedEncoder().to(device=device, dtype=torch.float32).eval()
input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], device=device)
mm_token_type_ids = torch.tensor([[0, 1, 1, 1, 1, 0, 0]], device=device)
with torch.inference_mode():
    without_token_types = encoder(input_ids)
    with_token_types = encoder(input_ids, mm_token_type_ids=mm_token_type_ids)
torch.cuda.synchronize()
print(f"input_ids={input_ids.cpu()}")
print(f"mm_token_type_ids={mm_token_type_ids.cpu()}")
print(f"without_mm_token_type_ids={without_token_types.cpu()}")
print(f"with_mm_token_type_ids={with_token_types.cpu()}")
print(f"delta={with_token_types.cpu() - without_token_types.cpu()}")
print(f"without_is_zero={bool(torch.equal(without_token_types, torch.zeros_like(without_token_types)))}")
print(f"with_is_zero={bool(torch.equal(with_token_types, torch.zeros_like(with_token_types)))}")
PY
```

Raw result:

```text
torch=2.9.1+rocm7.2.0.git7e1940d4
hip=7.2.26015-fc0010cf6a
cuda_available=True
device_count=1
device=cuda:0
device_name=AMD Instinct MI300X
input_ids=tensor([[1, 2, 3, 4, 5, 6, 7]])
mm_token_type_ids=tensor([[0, 1, 1, 1, 1, 0, 0]])
without_mm_token_type_ids=tensor([[[ 0.8487,  0.6920, -0.3160, -2.1152],
         [ 0.3223, -1.2633,  0.3500,  0.3081],
         [ 0.1198,  1.2377,  1.1168, -0.2473],
         [-1.3527, -1.6959,  0.5667,  0.7935],
         [ 0.5988, -1.5551, -0.3414,  1.8530],
         [ 0.7502, -0.5855, -0.1734,  0.1835],
         [ 1.3894,  1.5863,  0.9463, -0.8437]]])
with_mm_token_type_ids=tensor([[[ 0.2824,  1.0651, -1.2080, -3.6243],
         [ 0.6927,  0.1932,  1.2898,  1.0830],
         [ 0.4902,  2.6942,  2.0566,  0.5276],
         [-0.9823, -0.2394,  1.5065,  1.5684],
         [ 0.9692, -0.0986,  0.5984,  2.6279],
         [ 0.1839, -0.2124, -1.0654, -1.3256],
         [ 0.8230,  1.9594,  0.0543, -2.3528]]])
delta=tensor([[[-0.5663,  0.3731, -0.8920, -1.5091],
         [ 0.3704,  1.4565,  0.9398,  0.7748],
         [ 0.3704,  1.4565,  0.9398,  0.7748],
         [ 0.3704,  1.4565,  0.9398,  0.7748],
         [ 0.3704,  1.4565,  0.9398,  0.7748],
         [-0.5663,  0.3731, -0.8920, -1.5091],
         [-0.5663,  0.3731, -0.8920, -1.5091]]])
without_is_zero=False
with_is_zero=False
```

This control shows that explicit `mm_token_type_ids` are not numerically
inert on gfx942; omitting them changes the reduced encoder output.

## Regression checks

After applying the candidate-equivalent source change:

```text
python/sglang/multimodal_gen/test/unit/test_image_encoding_stage.py
=========================== 4 passed, 2 warnings in 0.64s ===================
```

Existing `ImageEncodingStage` component-residency checks:

```text
test_image_encoder_use_has_exact_precision PASSED
test_image_encoder_use_preserves_loaded_dtype_without_override PASSED
=========================== 2 passed, 2 warnings in 0.66s ===================
```

`git diff --check` produced no output. `ruff` and `black` were not installed in
`/opt/venv`, so no formatter was run.

## Commands and scope

Commands included bounded `git clone --depth 1`, one bounded candidate fetch,
read-only `gh issue view`, read-only `gh pr view`, `gh pr diff`, `rocminfo`,
`rocm-smi`, targeted pytest runs, and the small gfx942 control above. No model
weights were fetched. No node-wide state was modified. No upstream content was
created or changed.

## Uncertainty and left undone

- The full JoyAI image-edit end-to-end case was not run because it requires
  full model weights, which this task explicitly forbids.
- The candidate's upstream CI states were not treated as local validation;
  only the recorded local tests support the fix claim.
- `ImageEncodingStage` passes `image_grid_thw`, not `video_grid_thw`. The
  fixture therefore checks image-grid alignment. No video-path behavior was
  changed or claimed.
- Docker was unavailable inside the container, so image identity is recorded
  from the operator-provided qualified image and local image ID.

OPEN_TASK_REPORTED
