"""Tests for agent classes."""
from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from cckit import CLI, CodeAgent, ConversationAgent, CustomAgent, ResearchAgent
from cckit.agents.base import BaseAgent
from cckit.core.config import SessionConfig
from cckit.streaming.events import Event, ResultEvent, TextChunkEvent
from cckit.types.responses import Response


class TestAgentConfig:
    def _make_cli(self) -> CLI:
        return CLI(binary_path="/fake/claude")

    def test_code_agent_default_tools(self) -> None:
        agent = CodeAgent(cli=self._make_cli())
        tools = agent.get_default_tools()
        assert "Read" in tools
        assert "Edit" in tools
        assert "Bash" in tools

    def test_research_agent_default_tools(self) -> None:
        agent = ResearchAgent(cli=self._make_cli())
        tools = agent.get_default_tools()
        assert "WebSearch" in tools
        assert "WebFetch" in tools

    def test_research_agent_system_prompt(self) -> None:
        agent = ResearchAgent(cli=self._make_cli())
        assert "research" in agent.get_system_prompt().lower()

    def test_custom_agent_system_prompt(self) -> None:
        agent = CustomAgent(
            name="Poet",
            cli=self._make_cli(),
            system_prompt="Write only haiku.",
        )
        assert agent.get_system_prompt() == "Write only haiku."

    def test_custom_agent_tools(self) -> None:
        agent = CustomAgent(
            cli=self._make_cli(),
            tools=["Read", "Grep"],
        )
        cfg = agent._make_config()
        assert cfg.tools == ["Read", "Grep"]

    def test_with_config_overrides(self) -> None:
        agent = CodeAgent(cli=self._make_cli())
        agent.with_config(model="opus", tools=["Read"])
        cfg = agent._make_config()
        assert cfg.model == "opus"
        assert cfg.tools == ["Read"]

    def test_bare_default_true(self) -> None:
        agent = CodeAgent(cli=self._make_cli())
        cfg = agent._make_config()
        assert cfg.bare is True

    def test_code_agent_system_prompt(self) -> None:
        agent = CodeAgent(cli=self._make_cli())
        assert "engineer" in agent.get_system_prompt().lower()

    def test_conversation_agent_no_tools(self) -> None:
        agent = ConversationAgent(cli=self._make_cli())
        assert agent.get_default_tools() == []


class _FakeCLI:
    def __init__(self, response: Response | None = None) -> None:
        self.response = response or Response(
            result="agent reply", session_id="sid-1"
        )
        self.events: list[Event] = [
            TextChunkEvent(text="chunk1"),
            ResultEvent(result="final", session_id="sid-2"),
        ]

    async def execute(self, prompt, *, session_config=None, resume=None, **_):
        return self.response

    async def execute_streaming(
        self, prompt, *, session_config=None, resume=None, **_
    ) -> AsyncIterator[Event]:
        for ev in self.events:
            yield ev


class TestBaseAgentExecute:
    @pytest.mark.asyncio
    async def test_execute_returns_agent_result(self) -> None:
        agent = CodeAgent(cli=_FakeCLI())  # type: ignore[arg-type]
        result = await agent.execute("task")
        assert result.result == "agent reply"
        assert result.session_id == "sid-1"
        assert result.summary == "agent reply"

    @pytest.mark.asyncio
    async def test_stream_execute_yields_events(self) -> None:
        agent = CodeAgent(cli=_FakeCLI())  # type: ignore[arg-type]
        events = []
        async for ev in agent.stream_execute("task"):
            events.append(ev)
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_chat_returns_session(self) -> None:
        agent = CodeAgent(cli=_FakeCLI())  # type: ignore[arg-type]
        session = await agent.chat()
        assert session is not None
        assert session.config.tools  # Code agent has default tools

    @pytest.mark.asyncio
    async def test_chat_preserves_given_session(self) -> None:
        agent = CodeAgent(cli=_FakeCLI())  # type: ignore[arg-type]
        existing = await agent.chat()
        again = await agent.chat(session=existing)
        assert again is existing

    def test_with_config_bare_override(self) -> None:
        agent = CodeAgent(cli=_FakeCLI())  # type: ignore[arg-type]
        agent.with_config(bare=False, system_prompt="hi")
        cfg = agent._make_config()
        assert cfg.bare is False
        assert cfg.system_prompt == "hi"

    def test_custom_agent_default_prompts(self) -> None:
        agent = CustomAgent(name="X", cli=_FakeCLI())  # type: ignore[arg-type]
        # No explicit prompt — defaults to empty string
        assert agent.get_system_prompt() == ""
        assert agent.get_default_tools() == []


class TestConversationAgent:
    @pytest.mark.asyncio
    async def test_chat_starts_session_lazily(self) -> None:
        agent = ConversationAgent(cli=_FakeCLI())  # type: ignore[arg-type]
        assert agent.get_session() is None

        result = await agent.chat("hello")
        assert result.result == "agent reply"
        assert agent.get_session() is not None

    @pytest.mark.asyncio
    async def test_reset_clears_session(self) -> None:
        agent = ConversationAgent(cli=_FakeCLI())  # type: ignore[arg-type]
        await agent.chat("hi")
        session = agent.get_session()
        assert session is not None
        session.session_id = "marker"
        session.history.add_user("x")

        agent.reset()
        assert session.session_id == ""
        assert session.get_history() == []

    def test_reset_without_session_is_safe(self) -> None:
        agent = ConversationAgent(cli=_FakeCLI())  # type: ignore[arg-type]
        agent.reset()  # should not raise
