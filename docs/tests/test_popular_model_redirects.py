import importlib.util
import json
import unittest
from pathlib import Path


DOCS_ROOT = Path(__file__).resolve().parents[1]
REDIRECT_GENERATOR = DOCS_ROOT / "scripts" / "gen_redirects.py"

spec = importlib.util.spec_from_file_location("gen_redirects", REDIRECT_GENERATOR)
gen_redirects = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(gen_redirects)


POPULAR_MODEL_REDIRECTS = {
    "/basic_usage/deepseek_ocr.html": "/cookbook/autoregressive/DeepSeek/DeepSeek-OCR",
    "/basic_usage/deepseek_v3.html": "/cookbook/autoregressive/DeepSeek/DeepSeek-V3",
    "/basic_usage/deepseek_v32.html": "/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2",
    "/basic_usage/glm45.html": "/cookbook/autoregressive/GLM/GLM-4.5",
    "/basic_usage/glmv.html": "/cookbook/autoregressive/GLM/GLM-4.6V",
    "/basic_usage/gpt_oss.html": "/cookbook/autoregressive/OpenAI/GPT-OSS",
    "/basic_usage/kimi_k2_5.html": "/cookbook/autoregressive/Moonshotai/Kimi-K2.5",
    "/basic_usage/llama4.html": "/cookbook/autoregressive/Meta/Llama4",
    "/basic_usage/minimax_m2.html": "/cookbook/autoregressive/MiniMax/MiniMax-M2",
    "/basic_usage/popular_model_usage.html": "/cookbook/autoregressive/intro",
    "/basic_usage/qwen3.html": "/cookbook/autoregressive/Qwen/Qwen3-Next",
    "/basic_usage/qwen3_5.html": "/cookbook/autoregressive/Qwen/Qwen3.5",
    "/basic_usage/qwen3_vl.html": "/cookbook/autoregressive/Qwen/Qwen3-VL",
}


def route_to_file(route: str) -> Path:
    return DOCS_ROOT / f"{route.removeprefix('/')}.mdx"


class PopularModelRedirectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        docs_config = json.loads((DOCS_ROOT / "docs.json").read_text())
        cls.configured_redirects = docs_config["redirects"]
        cls.configured_by_source = {
            redirect["source"]: redirect["destination"]
            for redirect in cls.configured_redirects
        }

    def test_redirect_sources_are_unique(self):
        sources = [redirect["source"] for redirect in self.configured_redirects]
        self.assertEqual(len(sources), len(set(sources)))

    def test_migrated_routes_point_to_their_cookbook_pages(self):
        for source, destination in POPULAR_MODEL_REDIRECTS.items():
            with self.subTest(source=source):
                self.assertEqual(self.configured_by_source.get(source), destination)

    def test_generator_and_checked_in_config_agree(self):
        for source, destination in POPULAR_MODEL_REDIRECTS.items():
            generator_source = source.removesuffix(".html")
            with self.subTest(source=source):
                self.assertEqual(gen_redirects.EXPLICIT.get(generator_source), destination)

    def test_redirect_destinations_exist(self):
        for source, destination in POPULAR_MODEL_REDIRECTS.items():
            with self.subTest(source=source):
                self.assertTrue(
                    route_to_file(destination).is_file(),
                    f"{source} targets missing page {destination}",
                )

    def test_legacy_pages_are_not_still_published(self):
        for source in POPULAR_MODEL_REDIRECTS:
            legacy_stem = source.removeprefix("/basic_usage/").removesuffix(".html")
            for suffix in (".md", ".mdx", ".rst"):
                with self.subTest(source=source, suffix=suffix):
                    self.assertFalse(
                        (DOCS_ROOT / "docs" / "basic_usage" / f"{legacy_stem}{suffix}").exists()
                    )


if __name__ == "__main__":
    unittest.main()
