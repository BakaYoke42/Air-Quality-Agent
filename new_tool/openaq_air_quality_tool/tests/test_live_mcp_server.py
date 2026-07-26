from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import mcp_server  # noqa: E402


class LiveMcpWrapperTests(unittest.TestCase):
    def test_docstring_contains_agent_selection_contract(self):
        docstring = mcp_server.get_current_air_quality.__doc__ or ""

        for section in ("Use when:", "Do NOT use:", "Returns:", "Prefer:", "Example:"):
            self.assertIn(section, docstring)
        self.assertIn("not an annual mean", docstring)
        self.assertIn("different stations", docstring)

    def test_valid_data_status_uses_standard_success_envelope(self):
        core_result = {
            "status": "partial",
            "requested_location": "Berlin",
            "pollutants": {"pm25": {"value": 8.4}, "no2": None},
        }

        with patch.object(
            mcp_server,
            "get_air_quality",
            return_value=core_result,
        ):
            payload = json.loads(mcp_server.get_current_air_quality("Berlin"))

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["data"], core_result)

    def test_upstream_failure_uses_standard_controlled_error_envelope(self):
        with patch.object(
            mcp_server,
            "get_air_quality",
            return_value={
                "status": "upstream_error",
                "error": "OpenAQ returned HTTP 429.",
            },
        ):
            payload = json.loads(mcp_server.get_current_air_quality("Berlin"))

        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["tool"], "get_current_air_quality")
        self.assertEqual(payload["error_type"], "upstream_error")
        self.assertNotIn("API", json.dumps(payload).upper())

    def test_server_configuration_matches_streamable_http_project(self):
        self.assertTrue(mcp_server.mcp.settings.stateless_http)
        self.assertTrue(mcp_server.mcp.settings.json_response)
        self.assertEqual(mcp_server.MCP_HOST, "127.0.0.1")

    def test_mcp_discovery_exposes_one_typed_location_tool(self):
        tools = asyncio.run(mcp_server.mcp.list_tools())

        self.assertEqual([tool.name for tool in tools], ["get_current_air_quality"])
        schema = tools[0].inputSchema
        self.assertEqual(schema["required"], ["location"])
        self.assertEqual(schema["properties"]["location"]["type"], "string")


if __name__ == "__main__":
    unittest.main()
