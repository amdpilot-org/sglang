# Diffusion sleep/wake investigation

The feature request is implemented in the prepared base through upstream PR
[#22659](https://github.com/sgl-project/sglang/pull/22659), which superseded the
closed proposals [#19152](https://github.com/sgl-project/sglang/pull/19152) and
[#19153](https://github.com/sgl-project/sglang/pull/19153). Inspection found a
remaining public API gap: the HTTP routes existed, but `DiffGenerator` could not
invoke sleep, wake, or disk refit in local/offline workflows.

This contribution adds those three lifecycle methods, regression and adversarial
response tests, a real GPU numerical round-trip test, and user documentation.
Raw command output and exit-code files are retained in `raw/`.

Full diffusion-model generation and the requested sleep+wake+refit versus
kill-and-relaunch benchmark could not be run because the prepared environment
does not contain a diffusion checkpoint. The tiny Llama fixture is not suitable
evidence for a diffusion architecture and was intentionally not used.
