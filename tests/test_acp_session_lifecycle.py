"""Lifecycle tests for ACPSession against the echo_rpc fixture.

We monkeypatch ``RpcTransport`` to substitute the real claude binary
with our fixture, so the whole ACPSession stack (transport + client +
handlers) is exercised without a real Claude install.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cckit.rpc.transport import RpcTransport
from cckit.session.acp_session import ACPSession
from cckit.streaming.events import UsageEvent
from cckit.types.responses import Usage

ECHO_SERVER = str(Path(__file__).parent / "fixtures" / "echo_rpc.py")


@pytest.fixture(autouse=True)
def _patch_transport_cmd(monkeypatch):
    """Rewrite whatever ACPSession passes as the binary into [python, echo_rpc]."""
    original_init = RpcTransport.__init__

    def patched_init(self, cmd):  # type: ignore[no-untyped-def]
        original_init(self, [sys.executable, ECHO_SERVER])

    monkeypatch.setattr(RpcTransport, "__init__", patched_init)


class TestACPSessionLifecycle:
    @pytest.mark.asyncio
    async def test_create_and_close(self) -> None:
        session = await ACPSession.create(binary_path="/fake/claude")
        try:
            assert session.session_id == "sid-echo"
        finally:
            await session.close()

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        async with await ACPSession.create(binary_path="/fake/claude") as s:
            assert s.session_id == "sid-echo"

    @pytest.mark.asyncio
    async def test_create_with_config(self) -> None:
        from cckit.core.config import ACPConfig

        cfg = ACPConfig(
            binary_path="/fake/claude",
            model="opus",
            permission_policy="auto_approve",
            client_name="test",
            client_version="0.0.1",
        )
        async with await ACPSession.create(config=cfg) as s:
            assert s.session_id == "sid-echo"

    @pytest.mark.asyncio
    async def test_create_with_kwargs(self) -> None:
        async with await ACPSession.create(
            binary_path="/fake/claude",
            model="sonnet",
            system_prompt="be kind",
            cwd="/tmp",
        ) as s:
            assert s.session_id == "sid-echo"

    @pytest.mark.asyncio
    async def test_connect_loads_session(self) -> None:
        async with await ACPSession.connect(
            "sid-prior", binary_path="/fake/claude"
        ) as s:
            assert s.session_id == "sid-prior"

    @pytest.mark.asyncio
    async def test_cancel_without_error(self) -> None:
        async with await ACPSession.create(binary_path="/fake/claude") as s:
            await s.cancel()  # fire-and-forget


class TestACPSessionFailurePaths:
    @pytest.mark.asyncio
    async def test_create_unwinds_on_initialize_failure(
        self, monkeypatch
    ) -> None:
        """If initialize() raises, the transport must be stopped."""
        from cckit.rpc.client import ACPClient
        from cckit.rpc.transport import RpcTransport

        stopped = {"flag": False}

        original_init = ACPClient.initialize

        async def failing_initialize(self, *a, **kw):
            raise RuntimeError("nope")

        original_stop = RpcTransport.stop

        async def tracked_stop(self, *a, **kw):
            stopped["flag"] = True
            await original_stop(self, *a, **kw)

        monkeypatch.setattr(ACPClient, "initialize", failing_initialize)
        monkeypatch.setattr(RpcTransport, "stop", tracked_stop)

        with pytest.raises(RuntimeError, match="nope"):
            await ACPSession.create(binary_path="/fake/claude")
        assert stopped["flag"] is True

    @pytest.mark.asyncio
    async def test_connect_unwinds_on_failure(self, monkeypatch) -> None:
        from cckit.rpc.client import ACPClient
        from cckit.rpc.transport import RpcTransport

        stopped = {"flag": False}

        async def failing_load(self, session_id):
            raise RuntimeError("load-fail")

        original_stop = RpcTransport.stop

        async def tracked_stop(self, *a, **kw):
            stopped["flag"] = True
            await original_stop(self, *a, **kw)

        monkeypatch.setattr(ACPClient, "load_session", failing_load)
        monkeypatch.setattr(RpcTransport, "stop", tracked_stop)

        with pytest.raises(RuntimeError, match="load-fail"):
            await ACPSession.connect("sid", binary_path="/fake/claude")
        assert stopped["flag"] is True

    @pytest.mark.asyncio
    async def test_close_swallows_close_session_error(
        self, monkeypatch
    ) -> None:
        """close() should not raise even if close_session() fails."""
        from cckit.rpc.client import ACPClient

        session = await ACPSession.create(binary_path="/fake/claude")

        async def failing_close(self):
            raise RuntimeError("cannot close")

        monkeypatch.setattr(ACPClient, "close_session", failing_close)

        # Should not raise
        await session.close()


class TestACPClientContextManagerException:
    @pytest.mark.asyncio
    async def test_aexit_swallows_close_session_exception(
        self, monkeypatch
    ) -> None:
        """__aexit__ should log and continue if close_session raises."""
        import sys
        from pathlib import Path

        from cckit.rpc.client import ACPClient
        from cckit.rpc.handlers import DefaultHandlers
        from cckit.rpc.transport import RpcTransport

        # Undo the autouse fixture patching for a fresh direct transport
        monkeypatch.undo()

        echo = str(Path(__file__).parent / "fixtures" / "echo_rpc.py")
        transport = RpcTransport([sys.executable, echo])
        client = ACPClient(transport, handlers=DefaultHandlers())

        async def failing_close(self):
            raise RuntimeError("close-fail")

        monkeypatch.setattr(ACPClient, "close_session", failing_close)

        # Enter/exit — exception must be swallowed
        async with client:
            pass


class TestEventsToResponseUsageBranch:
    def test_usage_event_populates_response(self) -> None:
        events = [
            UsageEvent(
                input_tokens=11,
                output_tokens=22,
                cache_read_tokens=3,
                cache_write_tokens=4,
            ),
        ]
        resp = ACPSession._events_to_response(events)
        assert resp.usage == Usage(
            input_tokens=11,
            output_tokens=22,
            cache_read_tokens=3,
            cache_write_tokens=4,
        )
