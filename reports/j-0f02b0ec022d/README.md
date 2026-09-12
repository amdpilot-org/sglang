# XPU `CustomOp` dispatch investigation

The prepared base still returned `self.forward_xpu` from `dispatch_forward()` without defining that method on `CustomOp`. An issue-specific native-only subclass reproduced the report at construction time:

```text
construction_error=AttributeError: 'NativeOnly' object has no attribute 'forward_xpu'
```

Searches of open and closed upstream pull requests for `forward_xpu multimodal_gen CustomOp` returned no related fix. The narrow correction adds the same native fallback contract already used for TPU, OOT, and NPU.

The new regression test was run with the production hunk removed and produced 2 failures at construction and 1 pass (the subclass with its own XPU override). With the hunk restored, all 3 tests pass. Raw evidence is retained in `raw/`, including both issue snapshots, related-PR search output, standalone before/after output, test before/after output, and hardware inventory.

No GPU computation was run. The available accelerator is an AMD Instinct MI350X (`gfx950`), while the reported backend is Intel XPU. This fix is verified at the platform-dispatch contract level; actual Intel device execution and full multimodal model loading remain unverified.
