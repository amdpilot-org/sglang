# Independent review of PR 2823

Upstream issue: https://github.com/sgl-project/sglang/issues/31021

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2791

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2857

Candidate: https://github.com/amdpilot-org/sglang/pull/2823 at exact commit
`206d0a92883d859f4ebc6d64c7b480c7006904dc`.

Recommendation: **request changes**. The candidate is a meaningful partial
implementation: it adds an opt-in authenticated endpoint, scheduler dispatch,
selected pure-Python module reload, and graph-recapture calls. Its focused tests
pass. It does not safely satisfy its central promise that existing class
instances can continue through reloaded definitions.

## Blocking counterexample

The implementation reloads a module, copies attributes from each newly-created
class onto the old class object, and restores the old class identity. Python
methods that use zero-argument `super()` retain a hidden `__class__` closure
pointing to the new class. Calling such a copied method on an existing instance
of the old class fails:

```text
before base-v1-child-v1
reload ReloadResult(modules=('sglang.reload_super_fixture',), rebound_references=0)
after_error TypeError super(type, obj): obj must be an instance or subtype of type
```

This is a normal Python inheritance pattern and directly contradicts the API's
claim to rebind existing class instances. The raw reproducer and traceback are
in `evidence/adversarial_reload.py` and `evidence/adversarial-super.txt`.

## Scope and limitations

- The recorded base and image-prepared checkout were identical at
  `358c163250ad3b1f62939b01ce1314a0a31a0365`. On that base the flag and reload
  module are absent, reproducing the lack of the requested feature.
- The candidate changes only Python, tests, documentation, and report files; no
  C++, CUDA, HIP, or FlyDSL source changed. A native rebuild was therefore not
  applicable. Imports were confirmed from `/job/repo/python/sglang`, while Torch
  resolved from the pinned prepared environment with ROCm 7.2.
- Hardware inventory exposed one AMD Instinct MI355X (`gfx950`). No GPU execution
  was performed in this review because the deterministic CPU semantic
  counterexample already rejects the implementation, and the candidate's
  retained startup smoke cannot establish safe Python object rebinding.
- Multi-rank reload, rollback after partial module execution, large-model
  weight preservation/timing, torch.compile/JIT invalidation, non-Llama model
  paths, and recovery from serving-time exceptions remain unverified or
  explicitly unsupported. The alternative fault-tolerant event-loop direction
  from the issue is not implemented.

This report reviews the candidate only; it does not modify or duplicate the
candidate patch.
