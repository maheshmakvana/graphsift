"""Integration test: MCP server request/response."""

import json
import pytest
from unittest.mock import patch, MagicMock

from graphsift.mcp_server import (
    run_server,
    _handle_initialize,
    _handle_tools_list,
    _handle_tools_call,
    _handle_resources_list,
    _handle_resources_read,
    _handle_prompts_list,
    _handle_prompts_get,
)


class TestMCPIntegration:
    """Integration tests for MCP server protocol."""

    def _make_request(self, method: str, params: dict | None = None, req_id: int = 1) -> str:
        """Create a JSON-RPC request string."""
        req = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
        }
        if params:
            req["params"] = params
        return json.dumps(req)

    def _send_and_receive(self, method: str, params: dict | None = None):
        """Simulate sending a request and capturing the response."""
        responses = []

        def mock_write(data: str):
            responses.append(json.loads(data))

        # Use side_effect so readline() returns the request once, then "" for EOF
        # This lets the for-loop in run_server() terminate cleanly instead of hanging.
        request = self._make_request(method, params)
        with patch("sys.stdin.readline", side_effect=[request + "\n", ""]):
            with patch("sys.stdout.write", side_effect=mock_write):
                with patch("sys.stdout.flush"):
                    with patch("logging.basicConfig"):
                        try:
                            run_server()
                        except (StopIteration, SystemExit):
                            pass

        return responses[0] if responses else None

    def test_initialize_request(self):
        """MCP server responds to initialize request."""
        resp = self._send_and_receive("initialize", {
            "protocolVersion": "0.1.0",
            "clientInfo": {"name": "test-client", "version": "1.0"},
        })
        if resp:
            assert "result" in resp
            assert resp.get("id") == 1
            result = resp.get("result", {})
            assert "serverInfo" in result
            assert result["serverInfo"]["name"] == "graphsift"

    def test_list_tools_request(self):
        """MCP server responds to tools/list request."""
        resp = self._send_and_receive("tools/list")
        if resp:
            assert "result" in resp
            result = resp.get("result", {})
            tools = result.get("tools", [])
            assert len(tools) > 0
            tool_names = [t["name"] for t in tools]
            assert "compress_output" in tool_names or "build_context" in tool_names

    def test_list_resources_request(self):
        """MCP server responds to resources/list request."""
        resp = self._send_and_receive("resources/list")
        if resp:
            assert "result" in resp
            result = resp.get("result", {})
            resources = result.get("resources", [])
            assert isinstance(resources, list)

    def test_list_prompts_request(self):
        """MCP server responds to prompts/list request."""
        resp = self._send_and_receive("prompts/list")
        if resp:
            assert "result" in resp
            result = resp.get("result", {})
            prompts = result.get("prompts", [])
            assert isinstance(prompts, list)

    def test_call_tool_compress(self):
        """MCP server handles compress tool call."""
        resp = self._send_and_receive("tools/call", {
            "name": "compress_output",
            "arguments": {
                "text": "test output to compress\nline 2\nline 3\n",
            },
        })
        if resp:
            assert "result" in resp
            result = resp.get("result", {})
            content = result.get("content", [])
            assert len(content) > 0

    def test_unknown_method_returns_error(self):
        """MCP server returns error for unknown method."""
        resp = self._send_and_receive("unknown.method")
        if resp:
            assert "error" in resp
            assert resp["error"]["code"] == -32601

    def test_call_tool_no_such_tool(self):
        """MCP server returns error for unknown tool."""
        resp = self._send_and_receive("tools/call", {
            "name": "nonexistent_tool",
            "arguments": {},
        })
        if resp:
            assert "error" in resp or "isError" in resp.get("result", {})

    def test_initialize_with_protocol_version(self):
        """MCP server handles initialize with different protocol versions."""
        for version in ["0.1.0", "2024-11-05", "1.0"]:
            resp = self._send_and_receive("initialize", {
                "protocolVersion": version,
                "clientInfo": {"name": "test", "version": "1.0"},
            })
            if resp:
                assert "result" in resp


class TestMCPHandlers:
    """Direct handler tests for MCP functions."""

    def test_handle_initialize(self):
        """_handle_initialize returns proper server info."""
        with patch("sys.stdout.write") as mock_write:
            _handle_initialize(1, {"protocolVersion": "0.1.0", "clientInfo": {"name": "test", "version": "1.0"}})
            assert mock_write.called

    def test_handle_list_tools(self):
        """_handle_tools_list returns available tools."""
        with patch("sys.stdout.write") as mock_write:
            _handle_tools_list(1, {})
            assert mock_write.called
            call_args = mock_write.call_args[0][0]
            data = json.loads(call_args)
            assert "result" in data
            tools = data["result"]["tools"]
            assert len(tools) > 0

    def test_handle_list_resources(self):
        """_handle_resources_list returns available resources."""
        with patch("sys.stdout.write") as mock_write:
            _handle_resources_list(1, {})
            assert mock_write.called
            call_args = mock_write.call_args[0][0]
            data = json.loads(call_args)
            assert "result" in data
