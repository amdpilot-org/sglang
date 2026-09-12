import json
import unittest
from itertools import combinations
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


DASHBOARD_PATH = (
    Path(__file__).resolve().parents[4]
    / "examples/monitoring/grafana/dashboards/json/sglang-dashboard.json"
)


class TestGrafanaDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dashboard = json.loads(DASHBOARD_PATH.read_text())
        cls.panels_by_title = {
            panel["title"]: panel for panel in cls.dashboard["panels"]
        }

    def test_cache_hit_rate_panels_have_distinct_semantics(self):
        existing_panel = self.panels_by_title["Cache Hit Rate"]
        self.assertEqual(
            existing_panel["targets"][0]["expr"], "sglang_cache_hit_rate"
        )

        token_weighted_panel = self.panels_by_title[
            "Token-weighted Prefix Cache Hit Rate"
        ]
        self.assertEqual(
            token_weighted_panel["targets"][0]["expr"],
            'sum(rate(sglang_realtime_tokens_total{mode="prefill_cache"}'
            '[$__rate_interval])) / sum(rate(sglang_realtime_tokens_total'
            '{mode=~"prefill_cache|prefill_compute"}[$__rate_interval]))',
        )
        self.assertEqual(
            token_weighted_panel["fieldConfig"]["defaults"]["unit"],
            "percentunit",
        )
        self.assertEqual(token_weighted_panel["fieldConfig"]["defaults"]["min"], 0)
        self.assertEqual(token_weighted_panel["fieldConfig"]["defaults"]["max"], 1)

    def test_token_weighted_query_excludes_decode_and_batch_averaging(self):
        expression = self.panels_by_title[
            "Token-weighted Prefix Cache Hit Rate"
        ]["targets"][0]["expr"]

        self.assertNotIn('mode="decode"', expression)
        self.assertNotIn('mode=~"prefill_cache|prefill_compute|decode"', expression)
        self.assertNotIn("avg(", expression)
        self.assertEqual(expression.count("sum(rate("), 2)
        self.assertEqual(expression.count("$__rate_interval"), 2)

        # A short cold request and a long warm request must be weighted by tokens,
        # not treated as equally important batches.
        batches = [(0, 10), (900, 100)]
        token_weighted = sum(hit for hit, _ in batches) / sum(
            hit + computed for hit, computed in batches
        )
        batch_average = sum(
            hit / (hit + computed) for hit, computed in batches
        ) / len(batches)
        self.assertAlmostEqual(token_weighted, 900 / 1010)
        self.assertAlmostEqual(batch_average, 0.45)
        self.assertGreater(token_weighted, batch_average)

    def test_panel_ids_and_grid_positions_are_unique(self):
        panels = self.dashboard["panels"]
        self.assertEqual(len({panel["id"] for panel in panels}), len(panels))
        for left, right in combinations(panels, 2):
            left_grid = left["gridPos"]
            right_grid = right["gridPos"]
            overlaps = (
                left_grid["x"] < right_grid["x"] + right_grid["w"]
                and right_grid["x"] < left_grid["x"] + left_grid["w"]
                and left_grid["y"] < right_grid["y"] + right_grid["h"]
                and right_grid["y"] < left_grid["y"] + left_grid["h"]
            )
            self.assertFalse(
                overlaps,
                f'{left["title"]!r} overlaps {right["title"]!r}',
            )


if __name__ == "__main__":
    unittest.main()
