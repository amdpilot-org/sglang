"""Regression tests for lossy xgrammar string-constraint combinations."""

import json
import unittest
from unittest.mock import MagicMock

from sglang.srt.constrained.base_grammar_backend import InvalidGrammarObject
from sglang.srt.constrained.xgrammar_backend import (
    XGrammarGrammarBackend,
    has_xgrammar_unsupported_pattern_length_combination,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestXGrammarPatternLengthCombination(unittest.TestCase):
    def test_detects_min_and_max_length_combinations(self):
        for bound in ("minLength", "maxLength"):
            with self.subTest(bound=bound):
                self.assertTrue(
                    has_xgrammar_unsupported_pattern_length_combination(
                        {"type": "string", "pattern": "^[a-z]+$", bound: 5}
                    )
                )

    def test_detects_nested_subschemas(self):
        schemas = (
            {
                "properties": {
                    "value": {
                        "type": "string",
                        "pattern": "^[a-z]+$",
                        "minLength": 5,
                    }
                }
            },
            {
                "$defs": {
                    "value": {
                        "type": "string",
                        "pattern": "^[a-z]+$",
                        "maxLength": 5,
                    }
                }
            },
            {
                "allOf": [
                    {
                        "type": "string",
                        "pattern": "^[a-z]+$",
                        "minLength": 5,
                    }
                ]
            },
        )
        for schema in schemas:
            with self.subTest(schema=schema):
                self.assertTrue(
                    has_xgrammar_unsupported_pattern_length_combination(schema)
                )

    def test_detects_constraints_split_across_all_of(self):
        schemas = (
            {
                "type": "object",
                "properties": {
                    "v": {
                        "allOf": [
                            {"type": "string", "pattern": "^[a-z]+$"},
                            {"type": "string", "minLength": 5},
                        ]
                    }
                },
            },
            {
                "$defs": {
                    "letters": {"type": "string", "pattern": "^[a-z]+$"},
                    "long": {"type": "string", "minLength": 5},
                },
                "type": "object",
                "properties": {
                    "v": {
                        "allOf": [
                            {"$ref": "#/$defs/letters"},
                            {"$ref": "#/$defs/long"},
                        ]
                    }
                },
            },
        )
        for schema in schemas:
            with self.subTest(schema=schema):
                self.assertTrue(
                    has_xgrammar_unsupported_pattern_length_combination(schema)
                )

    def test_detects_property_constraints_split_across_object_all_of(self):
        schemas = (
            {
                "type": "object",
                "allOf": [
                    {"properties": {"v": {"pattern": "^[a-z]+$"}}},
                    {"properties": {"v": {"minLength": 5}}},
                ],
            },
            {
                "$defs": {
                    "patterned": {"properties": {"v": {"pattern": "^[a-z]+$"}}},
                    "long": {"properties": {"v": {"minLength": 5}}},
                },
                "type": "object",
                "allOf": [
                    {"$ref": "#/$defs/patterned"},
                    {"$ref": "#/$defs/long"},
                ],
            },
        )
        for schema in schemas:
            with self.subTest(schema=schema):
                self.assertTrue(
                    has_xgrammar_unsupported_pattern_length_combination(schema)
                )

    def test_does_not_merge_constraints_at_different_instance_locations(self):
        schemas = (
            {
                "anyOf": [
                    {"type": "string", "pattern": "^[a-z]+$"},
                    {"type": "string", "minLength": 5},
                ]
            },
            {
                "properties": {
                    "patterned": {"type": "string", "pattern": "^[a-z]+$"},
                    "long": {"type": "string", "minLength": 5},
                }
            },
            {
                "allOf": [
                    {"properties": {"patterned": {"pattern": "^[a-z]+$"}}},
                    {"properties": {"long": {"minLength": 5}}},
                ]
            },
        )
        for schema in schemas:
            with self.subTest(schema=schema):
                self.assertFalse(
                    has_xgrammar_unsupported_pattern_length_combination(schema)
                )

    def test_allows_individual_constraints(self):
        for schema in (
            {"type": "string", "pattern": "^[a-z]+$"},
            {"type": "string", "minLength": 5},
            {"type": "string", "maxLength": 5},
        ):
            with self.subTest(schema=schema):
                self.assertFalse(
                    has_xgrammar_unsupported_pattern_length_combination(schema)
                )

    def test_ignores_property_names_and_instance_data(self):
        schema = {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "minLength": 5},
                "minLength": {"type": "string", "pattern": "^[a-z]+$"},
            },
            "const": {"pattern": "^[a-z]+$", "minLength": 5},
            "default": {"pattern": "^[a-z]+$", "maxLength": 5},
        }
        self.assertFalse(has_xgrammar_unsupported_pattern_length_combination(schema))

    def test_dispatch_rejects_before_xgrammar_compilation(self):
        backend = object.__new__(XGrammarGrammarBackend)
        backend.grammar_compiler = MagicMock()
        schemas = (
            {
                "type": "object",
                "properties": {
                    "v": {
                        "type": "string",
                        "pattern": "^[a-z]+$",
                        "minLength": 5,
                    }
                },
            },
            {
                "type": "object",
                "properties": {
                    "v": {
                        "allOf": [
                            {"type": "string", "pattern": "^[a-z]+$"},
                            {"type": "string", "minLength": 5},
                        ]
                    }
                },
            },
        )

        for schema in schemas:
            with self.subTest(schema=schema):
                result = backend.dispatch_json(json.dumps(schema))
                self.assertIsInstance(result, InvalidGrammarObject)
                self.assertIn("cannot enforce together", result.error_message)
        backend.grammar_compiler.compile_json_schema.assert_not_called()

    def test_dispatch_preserves_supported_schema_compilation(self):
        backend = object.__new__(XGrammarGrammarBackend)
        backend.grammar_compiler = MagicMock()
        backend.any_whitespace = True
        backend.grammar_compiler.compile_json_schema.side_effect = RuntimeError(
            "compiler reached"
        )

        result = backend.dispatch_json('{"type":"string","minLength":5}')

        self.assertIsInstance(result, InvalidGrammarObject)
        self.assertEqual(result.error_message, "compiler reached")
        backend.grammar_compiler.compile_json_schema.assert_called_once()


if __name__ == "__main__":
    unittest.main()
