"""Tests for Session and MessageHistory."""
from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime

import pytest

from cckit.core.config import SessionConfig
from cckit.session.history import MessageHistory
from cckit.session.manager import ConversationManager
from cckit.session.session import Session
from cckit.streaming.events import Event, ResultEvent, TextChunkEvent
from cckit.types.messages import Message, ToolUse
from cckit.types.responses import Response


class _FakeCLI:
    """Minimal CLI stand-in for Session tests."""

    def __init__(
        self,
        response: Response | None = None,
        events: list[Event] | None = None,
    ) -> None:
        self.response = response or Response(result="ok", session_id="sid")
        self.events = events or []
        self.executed: list[tuple[str, str | None]] = []
        self.streamed: list[tuple[str, str | None]] = []

    async def execute(
        self, prompt: str, *, session_config=None, resume=None, **_
    ) -> Response:
        self.executed.append((prompt, resume))
        return self.response

    async def execute_streaming(
        self, prompt: str, *, session_config=None, resume=None, **_
    ) -> AsyncIterator[Event]:
        self.streamed.append((prompt, resume))
        for ev in self.events:
            yield ev


class TestMessageHistory:
    def test_add_and_get(self) -> None:
        h = MessageHistory()
        h.add_user("hello")
        h.add_assistant("hi there")
        msgs = h.get_all()
        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[1].role == "assistant"

    def test_max_messages(self) -> None:
        h = MessageHistory(max_messages=3)
        for i in range(5):
            h.add_user(f"msg {i}")
        assert len(h) == 3
        # Should keep the last 3
        assert h.get_all()[0].content == "msg 2"

    def test_last_assistant(self) -> None:
        h = MessageHistory()
        h.add_user("q1")
        h.add_assistant("a1")
        h.add_user("q2")
        last = h.last_assistant()
        assert last is not None
        assert last.content == "a1"

    def test_last_assistant_none(self) -> None:
        h = MessageHistory()
        h.add_user("only user")
        assert h.last_assistant() is None

    def test_clear(self) -> None:
        h = MessageHistory()
        h.add_user("hello")
        h.clear()
        assert len(h) == 0

    def test_export_import(self, tmp_path) -> None:
        h = MessageHistory()
        h.add_user("question")
        h.add_assistant("answer", tool_uses=[ToolUse("Bash", {"command": "ls"}, "file.py")])

        path = tmp_path / "history.json"
        h.save(path)

        loaded = MessageHistory.load(path)
        msgs = loaded.get_all()
        assert len(msgs) == 2
        assert msgs[1].tool_uses[0].tool_name == "Bash"

    def test_to_dict_round_trip(self) -> None:
        h = MessageHistory()
        h.add_user("ping")
        exported = h.export()
        assert exported[0]["role"] == "user"
        assert exported[0]["content"] == "ping"
        assert "timestamp" in exported[0]


class TestSession:
    @pytest.mark.asyncio
    async def test_create_sets_config(self) -> None:
        cli = _FakeCLI()
        session = await Session.create(
            cli, tools=["Read"], model="sonnet", system_prompt="be kind"
        )
        assert session.config.tools == ["Read"]
        assert session.config.model == "sonnet"
        assert session.config.system_prompt == "be kind"
        assert session.config.bare is True

    @pytest.mark.asyncio
    async def test_send_captures_session_id_and_history(self) -> None:
        cli = _FakeCLI(Response(result="reply", session_id="new-sid"))
        session = Session(cli)  # type: ignore[arg-type]

        response = await session.send("hello")
        assert response.result == "reply"
        assert session.session_id == "new-sid"

        # Second call forwards the captured session_id via resume=
        await session.send("follow-up")
        assert cli.executed[1][1] == "new-sid"

        # History has user+assistant pairs for both turns
        msgs = session.get_history()
        assert [m.role for m in msgs] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]

    @pytest.mark.asyncio
    async def test_stream_aggregates_text_chunks(self) -> None:
        events = [
            TextChunkEvent(text="Hel"),
            TextChunkEvent(text="lo"),
        ]
        cli = _FakeCLI(events=events)
        session = Session(cli)  # type: ignore[arg-type]

        yielded = []
        async for ev in session.stream("hi"):
            yielded.append(ev)
        assert len(yielded) == 2

        last = session.history.last_assistant()
        assert last is not None
        assert last.content == "Hello"

    @pytest.mark.asyncio
    async def test_stream_result_event_wins(self) -> None:
        events = [
            TextChunkEvent(text="partial"),
            ResultEvent(result="final", session_id="s99"),
        ]
        cli = _FakeCLI(events=events)
        session = Session(cli)  # type: ignore[arg-type]

        async for _ in session.stream("hi"):
            pass
        assert session.session_id == "s99"
        last = session.history.last_assistant()
        assert last is not None
        assert last.content == "final"

    def test_resume_classmethod_sets_session_id(self) -> None:
        cli = _FakeCLI()
        session = Session.resume(cli, "abc-sid")  # type: ignore[arg-type]
        assert session.session_id == "abc-sid"

    def test_fork_copies_history_and_session_id(self) -> None:
        cli = _FakeCLI()
        session = Session(cli, session_id="shared")  # type: ignore[arg-type]
        session.history.add_user("q")
        session.history.add_assistant("a")

        forked = session.fork()
        assert forked.session_id == "shared"
        assert [m.content for m in forked.get_history()] == ["q", "a"]

        # Independent histories
        forked.history.add_user("more")
        assert len(session.get_history()) == 2
        assert len(forked.get_history()) == 3

    def test_clear_history_resets_session_id(self) -> None:
        cli = _FakeCLI()
        session = Session(cli, session_id="sid")  # type: ignore[arg-type]
        session.history.add_user("q")
        session.clear_history()
        assert session.get_history() == []
        assert session.session_id == ""


class TestConversationManager:
    @pytest.mark.asyncio
    async def test_new_session_and_list(self) -> None:
        cli = _FakeCLI()
        mgr = ConversationManager(cli)  # type: ignore[arg-type]

        session = await mgr.new_session("main")
        assert mgr.get("main") is session
        assert "main" in mgr.list_sessions()

    @pytest.mark.asyncio
    async def test_new_session_anonymous_uses_local_id(self) -> None:
        cli = _FakeCLI()
        mgr = ConversationManager(cli)  # type: ignore[arg-type]

        session = await mgr.new_session()
        assert session._local_id in mgr.list_sessions()

    @pytest.mark.asyncio
    async def test_new_session_overrides_config(self) -> None:
        cli = _FakeCLI()
        mgr = ConversationManager(cli)  # type: ignore[arg-type]
        cfg = SessionConfig(model="opus")

        session = await mgr.new_session("main", config=cfg)
        assert session.config is cfg

    def test_resume_by_session_id(self) -> None:
        cli = _FakeCLI()
        mgr = ConversationManager(cli)  # type: ignore[arg-type]

        session = mgr.resume("sid-123")
        assert session.session_id == "sid-123"
        assert "sid-123" in mgr.list_sessions()

    def test_resume_with_explicit_name(self) -> None:
        cli = _FakeCLI()
        mgr = ConversationManager(cli)  # type: ignore[arg-type]

        mgr.resume("sid-1", name="alias")
        assert mgr.get("alias") is not None

    def test_remove_and_clear(self) -> None:
        cli = _FakeCLI()
        mgr = ConversationManager(cli)  # type: ignore[arg-type]

        mgr.resume("a")
        mgr.resume("b")
        mgr.remove("a")
        assert "a" not in mgr.list_sessions()
        mgr.remove("missing")  # noop
        mgr.clear()
        assert mgr.list_sessions() == []

    def test_get_missing_returns_none(self) -> None:
        cli = _FakeCLI()
        mgr = ConversationManager(cli)  # type: ignore[arg-type]
        assert mgr.get("nope") is None
