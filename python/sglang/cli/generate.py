import argparse
import os
import time

_MODULE_IMPORT_BEGIN_S = time.perf_counter()

from sglang.cli.utils import get_is_diffusion_model, get_model_path


def generate(args, extra_argv):
    if os.getenv("SGLANG_DIFFUSION_STARTUP_PROFILE", "0").lower() in (
        "1",
        "true",
    ):
        # The console entry point supplies an earlier timestamp from cli.main.
        # Keep this fallback for callers that import and invoke generate directly.
        os.environ.setdefault(
            "SGLANG_DIFFUSION_STARTUP_BEGIN", str(_MODULE_IMPORT_BEGIN_S)
        )
    # If help is requested, show generate subcommand help without requiring --model-path
    if any(h in extra_argv for h in ("-h", "--help")):
        from sglang.multimodal_gen.runtime.entrypoints.cli.generate import (
            add_multimodal_gen_generate_args,
        )

        parser = argparse.ArgumentParser(description="SGLang Multimodal Generation")
        add_multimodal_gen_generate_args(parser)
        parser.parse_args(extra_argv)
        return

    model_path = get_model_path(extra_argv)
    is_diffusion_model = get_is_diffusion_model(model_path)
    if is_diffusion_model:
        from sglang.multimodal_gen.runtime.entrypoints.cli.generate import (
            add_multimodal_gen_generate_args,
            generate_cmd,
        )

        parser = argparse.ArgumentParser(description="SGLang Multimodal Generation")
        add_multimodal_gen_generate_args(parser)
        parsed_args, unknown_args = parser.parse_known_args(extra_argv)
        generate_cmd(parsed_args, unknown_args)
    else:
        raise Exception(
            f"Generate subcommand is not yet supported for model: {model_path}"
        )
