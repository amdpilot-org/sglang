# Raw result summary

## Current source

```json
{
  "label": "current_fresh4",
  "commit": "ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4",
  "tuner_filename": "E=8,N=64,device_name=AMD_Instinct_MI300X,dtype=int4_w4a16.json",
  "runtime_filename": "E=8,N=128,device_name=AMD_Instinct_MI300X,dtype=int4_w4a16.json",
  "filename_match": false,
  "pre_copy_runtime_configs": null,
  "pre_copy_selected_config": {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1
  },
  "selected_config": {
    "BLOCK_SIZE_M": 32,
    "BLOCK_SIZE_N": 16,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 1,
    "num_warps": 1,
    "num_stages": 2,
    "waves_per_eu": 0
  },
  "gpu_output_norm": 0.07435648888349533,
  "reference_norm": 0.07439721375703812,
  "max_absolute_error": 6.103515625e-05,
  "mean_absolute_error": 1.0615835890348535e-05,
  "relative_l2_error": 0.00443923150804798,
  "gate_passed": true
}
```

## Candidate source

```json
{
  "label": "candidate_fresh",
  "commit": "429350e4d9cdd975d4de246cdd253e2d99603c2d",
  "tuner_filename": "E=8,N=128,device_name=AMD_Instinct_MI300X,dtype=int4_w4a16.json",
  "runtime_filename": "E=8,N=128,device_name=AMD_Instinct_MI300X,dtype=int4_w4a16.json",
  "filename_match": true,
  "pre_copy_runtime_configs": null,
  "pre_copy_selected_config": {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1
  },
  "selected_config": {
    "BLOCK_SIZE_M": 32,
    "BLOCK_SIZE_N": 16,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 1,
    "num_warps": 1,
    "num_stages": 2,
    "waves_per_eu": 0
  },
  "gpu_output_norm": 0.07435648888349533,
  "reference_norm": 0.07439721375703812,
  "max_absolute_error": 6.103515625e-05,
  "mean_absolute_error": 1.0615835890348535e-05,
  "relative_l2_error": 0.00443923150804798,
  "gate_passed": true
}
```

## Candidate focused test

```text
3 passed, 1 warning in 8.24s
```

## Tuner CLI blocker

```text
ModuleNotFoundError: No module named 'ray'
```
