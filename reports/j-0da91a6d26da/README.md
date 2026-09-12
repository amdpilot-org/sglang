# Developer reload correction generation 2

This correction preserves candidate
https://github.com/amdpilot-org/sglang/pull/3006 at exact commit
`9080a9f9b7d091a3f6ea8c7c1717c753dc592349` and addresses both concrete
counterexamples independently reported by
https://github.com/amdpilot-org/sglang/pull/3059.

## Reproduction and correction

The independent review script was run unchanged against the candidate after its two
commits were applied to prepared base
`358c163250ad3b1f62939b01ce1314a0a31a0365`. Reload reported success while a removed
function and class remained exposed, and while a retained `Child` class edited from
`BaseA` to `BaseB` kept `BaseA` as its base.

The correction removes non-import metadata from the selected module namespace before
`importlib.reload`, preventing Python's documented retention of names absent from the
new source. It also constructs all new-to-preserved class mappings before updating
classes, then applies the new base tuple through those preserved identities. This
keeps existing class/instance identity and zero-argument `super()` behavior while
making both existing and new instances follow the edited hierarchy. If Python rejects
an incompatible `__bases__` assignment, reload raises instead of reporting success.

## Evidence

- `evidence/failing-before.txt`: unchanged review script output against the candidate.
- `evidence/passing-after.txt`: both counterexamples pass after correction.
- `evidence/focused-tests.txt`: all eight focused tests pass.
- `evidence/adversarial_cases.py`: retained correction reproduction.

No GPU execution or native rebuild was applicable to this pure-Python object-model
correction. The preserved candidate contains its earlier serving evidence; it was not
repeated or claimed as evidence for these defects.

## Remaining limitations

Multi-rank coordination remains unverified on hardware. Native extensions, newly
imported modules, incompatible object-layout changes, process-topology changes,
torch.compile artifacts, and JIT kernel changes still require restart. Reload remains
non-transactional when module execution or class migration raises. The optional
fault-tolerant scheduler exception loop is not implemented. The prepared interpreter
does not contain Ruff.
