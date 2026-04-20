"""Tests for Response / Usage dataclasses."""
from __future__ import annotations

from cckit.types.responses import AgentResult, Response, Usage


class TestUsage:
    def test_from_dict_all_fields(self) -> None:
        usage = Usage.from_dict(
            {
                "input_tokens": 10,
                "output_tokens": 20,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 4,
            }
        )
        assert usage.input_tokens == 10
        assert usage.output_tokens == 20
        assert usage.cache_read_tokens == 3
        assert usage.cache_write_tokens == 4

    def test_from_dict_defaults_zero(self) -> None:
        usage = Usage.from_dict({})
        assert usage.input_tokens == 0
        assert usage.cache_read_tokens == 0


class TestResponse:
    def test_from_json_full(self) -> None:
        data = {
            "result": "ok",
            "session_id": "sid",
            "duration_ms": 1234,
            "usage": {"input_tokens": 5, "output_tokens": 6},
            "stop_reason": "stop",
            "is_error": False,
        }
        r = Response.from_json(data)
        assert r.result == "ok"
        assert r.session_id == "sid"
        assert r.usage.input_tokens == 5
        assert r.stop_reason == "stop"

    def test_from_json_minimal_has_empty_usage(self) -> None:
        r = Response.from_json({})
        assert r.result == ""
        assert r.usage.input_tokens == 0

    def test_error_helper(self) -> None:
        r = Response.error("boom")
        assert r.is_error is True
        assert r.result == "boom"


class TestAgentResult:
    def test_result_and_session_id_proxies(self) -> None:
        inner = Response(result="hello", session_id="sid-9")
        ar = AgentResult(response=inner, summary="s")
        assert ar.result == "hello"
        assert ar.session_id == "sid-9"
