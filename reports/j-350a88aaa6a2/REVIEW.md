# Correction-generation result

Candidate PR: https://github.com/amdpilot-org/sglang/pull/557

Independent review PR: https://github.com/amdpilot-org/sglang/pull/701

At exact candidate commit `8c342b1aea310267115f223bfbb9573791147543`, xgrammar 0.2.2 accepts `{"data":2,"extra":{"data":"ok"}}` but rejects the semantically equivalent `{"extra":{"data":"ok"},"data":2}`. It still correctly rejects the original invalid named object, so the candidate contains a valid partial fix.

The additional-first rejection also reproduces directly through `xgrammar.Grammar.from_json_schema` with the latest released xgrammar 0.2.6. Because the failure occurs without SGLang's structural-tag builder, changing SGLang source would be speculative and would risk replacing the compiler's JSON Schema implementation with an incomplete local approximation.

This branch therefore preserves the candidate's dependency upgrade and original regression, and adds a strict expected-failure regression for the unresolved ordering defect. A complete fix requires a corrected xgrammar JSON-schema compiler release.
