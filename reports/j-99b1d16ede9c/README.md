# Developer reload correction report

This correction preserves the implementation from candidate
https://github.com/amdpilot-org/sglang/pull/2823 at exact commit
`206d0a92883d859f4ebc6d64c7b480c7006904dc` and addresses the concrete
counterexample independently reported by
https://github.com/amdpilot-org/sglang/pull/2923.

## Reproduction and cause

The review's exact adversarial script was run against the candidate after it was
applied to prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365`. It exited
1 with `TypeError: super(type, obj): obj must be an instance or subtype of type`.
The candidate preserves the old class object for existing instances, but copies
functions from the newly created class. A zero-argument `super()` function retains
a `__class__` closure cell containing that discarded new class.

The correction rewrites that cell to the preserved old class before installing the
updated attribute. It handles functions directly and the standard classmethod,
staticmethod, and property descriptors. The class identity remains unchanged and
the old instance then dispatches through both the updated child and base methods.

## Evidence

- `evidence/failing-before.txt`: exact candidate failure, exit 1.
- `evidence/passing-after.txt`: exact review script after correction, exit 0.
- `evidence/focused-tests.txt`: all six focused tests pass, including the committed
  regression.
- `evidence/adversarial_reload.py`: the independent review's reproduction script.

No GPU execution or native rebuild was needed for this pure-Python object-model
correction. The candidate's retained report contains its earlier TP=1 ROCm serving
evidence; this correction did not repeat that unrelated model run.

## Remaining limitations

The candidate's documented limits remain: multi-rank coordination was not
hardware-tested; native extensions, newly imported modules, layout/topology changes,
torch.compile artifacts, and JIT kernel changes require restart; reload is not
transactional when module top-level code raises; and the optional fault-tolerant
scheduler exception loop is not implemented. Arbitrary third-party descriptors or
decorators that hide a `__class__`-capturing function are not introspected.
