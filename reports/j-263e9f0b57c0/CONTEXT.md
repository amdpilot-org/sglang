# Reference context

The read-only upstream reference is sgl-project/sglang issue 15194, “[Roadmap] Quantization Modifications.” The issue is open and proposes a scheme-based quantization layout, separation of weight loading from inference, and additional format support. Its comments discuss organizing hardware-backend kernels consistently and where ModelOpt formats should fit.

Related changes inspected on 2026-09-10:

- sgl-project/sglang PR 21126, AWQ schemes and kernel/weight-init split: merged.
- sgl-project/sglang PR 26402, GPTQ schemes and kernel split: merged.
- sgl-project/sglang PR 26786, CPU GPTQ quantization schemes: merged.
- sgl-project/sglang PR 26846, AutoRound package-layout split: open.
- sgl-project/sglang PR 26942, GGUF hardware-backend/scheme split: open.

This deliverable does not duplicate those implementation refactors. It adds a bounded, report-only GPU memory-scaling benchmark for the distinct follow-up described by amdpilot-org/sglang issue 391. No upstream issue, PR, or comment was posted or modified.
