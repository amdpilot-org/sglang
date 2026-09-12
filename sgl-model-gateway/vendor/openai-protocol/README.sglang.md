# Vendored protocol patch

This is the published `openai-protocol` 1.0.0 crate used by
`sgl-model-gateway`, vendored so all direct and transitive users resolve to the
same patched crate. The only source change from the published crate is the
addition of the Responses API tool-type variants accepted by SGLang's Python
protocol.

The path override can be removed once the gateway migrates to a compatible
published `openai-protocol` release containing those variants.
