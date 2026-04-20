"""Tests for streaming parser and handler."""
from __future__ import annotations

import json

import pytest

from cckit.streaming.events import (
    BaseEvent,
    MessageCompleteEvent,
    MessageStartEvent,
    ResultEvent,
    SystemEvent,
    TextChunkEvent,
    ToolResultEvent,
    ToolUseEvent,
)
from cckit.streaming.handler import StreamHandler
from cckit.streaming.parser import parse_line
from cckit.utils.errors import ParseError


class TestParseLine:
    def test_empty_line_returns_none(self) -> None:
        assert parse_line("") is None
        assert parse_line("   ") is None

    def test_result_event(self) -> None:
        data = {"type": "result", "result": "hello", "session_id": "s1", "duration_ms": 100}
        event = parse_line(json.dumps(data))
        assert isinstance(event, ResultEvent)
        assert event.result == "hello"
        assert event.session_id == "s1"
        assert event.duration_ms == 100

    def test_system_event(self) -> None:
        data = {"type": "system", "subtype": "init"}
        event = parse_line(json.dumps(data))
        assert isinstance(event, SystemEvent)
        assert event.subtype == "init"

    def test_assistant_text_event(self) -> None:
        data = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello world"}],
            },
        }
        event = parse_line(json.dumps(data))
        assert isinstance(event, TextChunkEvent)
        assert event.text == "Hello world"

    def test_assistant_tool_use_event(self) -> None:
        data = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "id": "tu_1",
                        "input": {"command": "ls"},
                    }
                ],
            },
        }
        event = parse_line(json.dumps(data))
        assert isinstance(event, ToolUseEvent)
        assert event.tool_name == "Bash"
        assert event.tool_input == {"command": "ls"}
        assert event.tool_use_id == "tu_1"

    def test_message_delta_stop(self) -> None:
        data = {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}
        event = parse_line(json.dumps(data))
        assert isinstance(event, MessageCompleteEvent)
        assert event.stop_reason == "end_turn"

    def test_content_block_delta_text(self) -> None:
        data = {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "chunk"},
        }
        event = parse_line(json.dumps(data))
        assert isinstance(event, TextChunkEvent)
        assert event.text == "chunk"

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ParseError):
            parse_line("{not valid json}")

    def test_message_start(self) -> None:
        data = {"type": "message_start", "message": {"role": "assistant"}}
        event = parse_line(json.dumps(data))
        assert isinstance(event, MessageStartEvent)
        assert event.role == "assistant"

    def test_message_stop(self) -> None:
        data = {"type": "message_stop"}
        event = parse_line(json.dumps(data))
        assert isinstance(event, MessageCompleteEvent)

    def test_content_block_start_tool_use(self) -> None:
        data = {
            "type": "content_block_start",
            "content_block": {
                "type": "tool_use",
                "name": "Bash",
                "id": "tu_9",
            },
        }
        event = parse_line(json.dumps(data))
        assert isinstance(event, ToolUseEvent)
        assert event.tool_name == "Bash"
        assert event.tool_use_id == "tu_9"

    def test_tool_result_string_content(self) -> None:
        data = {
            "type": "tool_result",
            "tool_use_id": "tu_1",
            "content": "stdout line",
            "is_error": False,
        }
        event = parse_line(json.dumps(data))
        assert isinstance(event, ToolResultEvent)
        assert event.content == "stdout line"

    def test_tool_result_list_content_flattens(self) -> None:
        data = {
            "type": "tool_result",
            "tool_use_id": "tu_2",
            "content": [
                {"type": "text", "text": "line1"},
                {"type": "text", "text": "line2"},
                {"type": "other"},
            ],
        }
        event = parse_line(json.dumps(data))
        assert isinstance(event, ToolResultEvent)
        assert "line1" in event.content
        assert "line2" in event.content

    def test_assistant_fallback_on_unknown_content(self) -> None:
        data = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "unknown"}],
            },
        }
        event = parse_line(json.dumps(data))
        assert isinstance(event, TextChunkEvent)

    def test_unknown_event_returns_base(self) -> None:
        data = {"type": "totally_unknown"}
        event = parse_line(json.dumps(data))
        assert isinstance(event, BaseEvent)

    def test_assistant_skips_non_dict_blocks(self) -> None:
        data = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": ["not-a-dict", {"type": "text", "text": "ok"}],
            },
        }
        event = parse_line(json.dumps(data))
        assert isinstance(event, TextChunkEvent)
        assert event.text == "ok"


class TestStreamHandler:
    @pytest.mark.asyncio
    async def test_collect_from_result_event(self) -> None:
        lines = [
            json.dumps({"type": "system", "subtype": "init"}),
            json.dumps(
                {
                    "type": "result",
                    "result": "final answer",
                    "session_id": "abc",
                    "duration_ms": 500,
                }
            ),
        ]

        async def gen():
            for line in lines:
                yield line

        handler = StreamHandler()
        response = await handler.collect_result(handler.process_stream(gen()))
        assert response.result == "final answer"
        assert response.session_id == "abc"
        assert response.duration_ms == 500

    @pytest.mark.asyncio
    async def test_collect_from_text_chunks(self) -> None:
        lines = [
            json.dumps({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hello "},
            }),
            json.dumps({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "World"},
            }),
        ]

        async def gen():
            for line in lines:
                yield line

        handler = StreamHandler()
        response = await handler.collect_result(handler.process_stream(gen()))
        assert response.result == "Hello World"

    @pytest.mark.asyncio
    async def test_process_stream_yields_events(self) -> None:
        lines = [
            json.dumps({"type": "result", "result": "done", "session_id": "", "duration_ms": 0}),
        ]

        async def gen():
            for line in lines:
                yield line

        handler = StreamHandler()
        events = []
        async for event in handler.process_stream(gen()):
            events.append(event)

        assert len(events) == 1
        assert isinstance(events[0], ResultEvent)

    @pytest.mark.asyncio
    async def test_process_stream_skips_blank_lines(self) -> None:
        lines = ["", "   ", json.dumps({"type": "system", "subtype": "init"})]

        async def gen():
            for line in lines:
                yield line

        handler = StreamHandler()
        events = [e async for e in handler.process_stream(gen())]
        assert len(events) == 1
        assert isinstance(events[0], SystemEvent)

    @pytest.mark.asyncio
    async def test_process_stream_swallows_parse_errors(self) -> None:
        lines = [
            "{not valid json}",
            json.dumps({"type": "result", "result": "ok", "session_id": "", "duration_ms": 0}),
        ]

        async def gen():
            for line in lines:
                yield line

        handler = StreamHandler()
        events = [e async for e in handler.process_stream(gen())]
        # Bad line is dropped, good line yields ResultEvent
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_collect_result_accumulates_usage(self) -> None:
        from cckit.streaming.events import UsageEvent

        async def gen():
            yield UsageEvent(
                input_tokens=10,
                output_tokens=20,
                cache_read_tokens=1,
                cache_write_tokens=2,
            )
            yield TextChunkEvent(text="Hi")
            # No result event — fall through to aggregated branch

        handler = StreamHandler()
        response = await handler.collect_result(gen())
        assert response.result == "Hi"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20
