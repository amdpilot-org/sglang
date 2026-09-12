# Seeded min-p sampling investigation

The prepared base still contained the reported assertion, and the original
reproduction failed on the assigned gfx950 GPU before sampling. Removing only
that assertion allows the already-existing deterministic Gumbel-max path to
consume the filtered log-weights. Filtered zero weights become negative
infinity and cannot win the argmax.

An upstream candidate with the same source correction was found at
https://github.com/sgl-project/sglang/pull/33699. It remains open and is not
present in the prepared base. This change adds independent coverage beyond that
candidate's single regression: explicit comparison with separately normalized
filtered weights, plus an inclusive min-p threshold combined with top-k.

The GPU checks demonstrate repeatability, exclusion of filtered vocabulary
IDs, and correct mapping from sorted positions back to original token IDs. No
native code changed, so no native rebuild was applicable.

An unrelated full-model sampling-mask check was attempted but was blocked by
missing access to gated Llama weights. No serving-path or full-model claim is
made. The deterministic tiny Llama fixture was not needed because this issue is
inside the directly exercised sampling operation; that fixture would validate
transport/engine execution rather than add evidence about this numerical path.
