# Investigation evidence

Source checkout: `/job/repo`

Prepared interpreter: `/tmp/amdpilot-repo-j-95de80bf7287/venv/bin/python`

Prepared native artifact: none (`repository-environment.json` records `"native": null`)

Compiler evidence: installed `xgrammar==0.2.1`; probes constructed `xgrammar.GrammarCompiler` with a 256-entry raw-byte vocabulary, one compiler thread, and cache disabled, then called `compile_json_schema` on each generated schema in a separate process.

## Bounded pre-fix reproduction

```text
depth=16   bytes=1678  seconds=0.002048  result=CompiledGrammar
depth=32   bytes=3358  seconds=0.008130  result=CompiledGrammar
depth=64   bytes=6718  seconds=0.048804  result=CompiledGrammar
depth=128  bytes=13494 seconds=0.358549  result=CompiledGrammar
depth=192  bytes=20342 seconds=1.202919  result=CompiledGrammar
depth=256  bytes=27190 seconds=2.902661  result=CompiledGrammar
depth=320  hard deadline=6s exit=137 (killed by timeout)
branch=128 bytes=5747  seconds=0.004091  result=CompiledGrammar
oneOf=64   bytes=7661  seconds=0.003631  result=CompiledGrammar
```

These small, bounded cases isolate nesting as the demonstrated nonlinear resource vector; they do not support a claim about compiler state counts.

## Post-fix probes

```text
depth=320 elapsed=0.000677 rejected=True compiler_called=False
error='JSON grammar nesting depth exceeds the supported limit of 256'

depth=128 elapsed=0.355862 result_type=BaseGrammarObject invalid=False
```

## GPU numerical evidence

```text
device='AMD Instinct MI355X'
torch='2.11.0+rocm7.2'
hip='7.2.26015'
result=[[19.0, 22.0], [43.0, 50.0]]
independent_reference=[[19.0, 22.0], [43.0, 50.0]]
max_abs_error=0.0
```

## Focused regression output

```text
84 passed, 17 warnings, 17 subtests passed in 8.98s
```
