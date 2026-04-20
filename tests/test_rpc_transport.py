"""Tests for RpcTransport using the echo_rpc.py fixture."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cckit.rpc.transport import RpcTransport
from cckit.utils.errors import CLIError, RpcError, TransportError

ECHO_SERVER = str(Path(__file__).parent / "fixtures" / "echo_rpc.py")


def _make_transport() -> RpcTransport:
    return RpcTransport([sys.executable, ECHO_SERVER])


class TestTransportLifecycle:
    @pytest.mark.asyncio
    async def test_start_and_stop(self) -> None:
        transport = _make_transport()
        await transport.start()
        assert transport._proc is not None
        assert transport._proc.returncode is None
        await transport.stop()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self) -> None:
        transport = _make_transport()
        await transport.start()
        await transport.stop()
        await transport.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_binary_not_found(self) -> None:
        transport = RpcTransport(["/nonexistent/binary"])
        with pytest.raises(CLIError, match="Binary not found"):
            await transport.start()


class TestTransportRequests:
    @pytest.mark.asyncio
    async def test_echo_request(self) -> None:
        transport = _make_transport()
        await transport.start()
        try:
            result = await transport.request("echo", {"hello": "world"})
            assert result == {"hello": "world"}
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        transport = _make_transport()
        await transport.start()
        try:
            result = await transport.request("initialize", {})
            assert "capabilities" in result
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_error_response_raises_rpc_error(self) -> None:
        transport = _make_transport()
        await transport.start()
        try:
            with pytest.raises(RpcError, match="Test error"):
                await transport.request("error", {})
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        transport = _make_transport()
        await transport.start()
        try:
            with pytest.raises(TransportError, match="timed out"):
                await transport.request("slow", {"delay": 10}, timeout=0.1)
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_request_after_stop_raises(self) -> None:
        transport = _make_transport()
        await transport.start()
        await transport.stop()
        with pytest.raises(TransportError, match="not connected"):
            await transport.request("echo", {})


class TestTransportNotifications:
    @pytest.mark.asyncio
    async def test_send_notification(self) -> None:
        transport = _make_transport()
        await transport.start()
        try:
            # Should not raise (fire-and-forget)
            await transport.notify("some/event", {"data": 1})
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_receive_notification(self) -> None:
        transport = _make_transport()
        received: list[dict] = []
        transport.on_notification(
            "test/notification", lambda params: received.append(params)
        )

        await transport.start()
        try:
            result = await transport.request("notify_back", {"msg": "hello"})
            assert result["notified"] is True

            # Give the reader loop a moment to process the notification
            import asyncio

            await asyncio.sleep(0.1)
            assert len(received) == 1
            assert received[0]["msg"] == "hello"
        finally:
            await transport.stop()


class TestTransportIncomingRequests:
    @pytest.mark.asyncio
    async def test_handle_incoming_request(self) -> None:
        transport = _make_transport()

        async def handle_callback(params):
            return {"handled": True, "echo": params.get("msg", "")}

        transport.on_request("test/callback", handle_callback)

        await transport.start()
        try:
            result = await transport.request("callback", {"msg": "ping"})
            assert result["callback_result"]["handled"] is True
            assert result["callback_result"]["echo"] == "ping"
        finally:
            await transport.stop()


class TestMultipleRequests:
    @pytest.mark.asyncio
    async def test_sequential_requests(self) -> None:
        transport = _make_transport()
        await transport.start()
        try:
            r1 = await transport.request("echo", {"n": 1})
            r2 = await transport.request("echo", {"n": 2})
            r3 = await transport.request("echo", {"n": 3})
            assert r1 == {"n": 1}
            assert r2 == {"n": 2}
            assert r3 == {"n": 3}
        finally:
            await transport.stop()


class TestIncomingRequestErrorPaths:
    @pytest.mark.asyncio
    async def test_unknown_method_returns_method_not_found(self) -> None:
        """Server asks for a method we did not register — we answer error."""
        transport = _make_transport()
        await transport.start()
        try:
            with pytest.raises(RpcError) as excinfo:
                await transport.request(
                    "reverse_request",
                    {"client_method": "does/not/exist", "client_params": {}},
                )
            # METHOD_NOT_FOUND = -32601
            assert excinfo.value.code == -32601
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_handler_exception_returns_internal_error(self) -> None:
        """A registered handler that raises → generic INTERNAL_ERROR to remote."""
        transport = _make_transport()

        def broken(_params):
            raise RuntimeError("handler blew up")

        transport.on_request("client/broken", broken)

        await transport.start()
        try:
            with pytest.raises(RpcError) as excinfo:
                await transport.request(
                    "reverse_request",
                    {"client_method": "client/broken", "client_params": {}},
                )
            # INTERNAL_ERROR = -32603
            assert excinfo.value.code == -32603
            # H4: don't leak exception details
            assert "handler blew up" not in excinfo.value.args[0]
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_async_handler_round_trip(self) -> None:
        transport = _make_transport()

        async def ahandler(params):
            return {"got": params.get("v")}

        transport.on_request("client/a", ahandler)
        await transport.start()
        try:
            result = await transport.request(
                "reverse_request",
                {"client_method": "client/a", "client_params": {"v": 42}},
            )
            assert result["client_result"] == {"got": 42}
        finally:
            await transport.stop()


class TestNotificationHandlerErrors:
    @pytest.mark.asyncio
    async def test_notification_handler_exception_is_logged(self) -> None:
        """Notification handlers that raise should not crash the reader loop."""
        transport = _make_transport()

        def boom(_params):
            raise RuntimeError("notification handler raised")

        transport.on_notification("client/boom", boom)

        await transport.start()
        try:
            # Ask the fixture to send us a notification with method=client/boom
            await transport.request(
                "notify_to_client",
                {"method": "client/boom", "params": {}},
            )
            # Give the reader loop a moment
            import asyncio

            await asyncio.sleep(0.1)

            # Transport still works after the raise
            result = await transport.request("echo", {"ok": 1})
            assert result == {"ok": 1}
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_unregistered_notification_is_dropped(self) -> None:
        transport = _make_transport()
        await transport.start()
        try:
            await transport.request(
                "notify_to_client",
                {"method": "client/nothing", "params": {}},
            )
            # Still operable
            assert await transport.request("echo", {"ok": 1}) == {"ok": 1}
        finally:
            await transport.stop()


class TestStartTwice:
    @pytest.mark.asyncio
    async def test_start_twice_raises(self) -> None:
        transport = _make_transport()
        await transport.start()
        try:
            with pytest.raises(TransportError, match="already started"):
                await transport.start()
        finally:
            await transport.stop()


class TestTransportRobustness:
    @pytest.mark.asyncio
    async def test_notify_after_stop_raises(self) -> None:
        transport = _make_transport()
        await transport.start()
        await transport.stop()
        with pytest.raises(TransportError, match="not connected"):
            await transport.notify("x", {})

    @pytest.mark.asyncio
    async def test_non_json_line_is_skipped(self) -> None:
        """Reader logs+drops non-JSON lines, then continues normally."""
        transport = _make_transport()
        await transport.start()
        try:
            result = await transport.request("send_garbage", {})
            assert result == {"ok": True}
            # Transport still functional
            assert await transport.request("echo", {"x": 1}) == {"x": 1}
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_unrecognized_message_ignored(self) -> None:
        """A message with neither id nor method is logged and dropped."""
        transport = _make_transport()
        await transport.start()
        try:
            result = await transport.request("send_unrecognized", {})
            assert result == {"ok": True}
            assert await transport.request("echo", {"z": 3}) == {"z": 3}
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_response_for_unknown_id_ignored(self) -> None:
        """Reader drops responses with unknown ids without crashing."""
        transport = _make_transport()
        await transport.start()
        try:
            result = await transport.request("respond_to_unknown", {})
            assert result == {"ok": True}
            # Give the reader a moment
            import asyncio
            await asyncio.sleep(0.05)
            # Transport still functional
            assert await transport.request("echo", {"y": 2}) == {"y": 2}
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_stop_rejects_pending_requests(self) -> None:
        import asyncio

        transport = _make_transport()
        await transport.start()

        # Fire a slow request but don't await — stop() should reject it
        task = asyncio.create_task(transport.request("slow", {"delay": 10}))
        await asyncio.sleep(0.05)  # let it queue

        await transport.stop()
        with pytest.raises(TransportError):
            await task

    @pytest.mark.asyncio
    async def test_reader_loop_crash_terminates_subprocess(
        self, monkeypatch
    ) -> None:
        """If _on_message raises, the reader catches it and terminates the proc."""
        import asyncio

        transport = _make_transport()

        async def boom(_data):
            raise RuntimeError("reader crash")

        await transport.start()
        try:
            transport._on_message = boom  # type: ignore[assignment]

            # Trigger any server response to hit _on_message
            task = asyncio.create_task(
                transport.request("echo", {"x": 1}, timeout=5.0)
            )
            # Wait for reader to crash + terminate + futures reject
            with pytest.raises(TransportError):
                await task
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_request_broken_pipe_wrapped(self, monkeypatch) -> None:
        """Writing to a closed stdin should surface as TransportError."""
        transport = _make_transport()
        await transport.start()
        try:
            assert transport._proc and transport._proc.stdin

            def broken_write(self_, _data):
                raise BrokenPipeError()

            monkeypatch.setattr(
                type(transport._proc.stdin), "write", broken_write
            )
            with pytest.raises(TransportError, match="Broken pipe"):
                await transport.request("echo", {"x": 1})
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_stop_falls_through_to_kill_on_timeout(self) -> None:
        """stop() must hard-kill a process that ignores SIGTERM."""
        import asyncio

        sigterm_ignore = str(
            Path(__file__).parent / "fixtures" / "sigterm_ignore.py"
        )
        transport = RpcTransport([sys.executable, sigterm_ignore])
        await transport.start()

        assert transport._proc is not None
        # Wait for child to install the SIG_IGN handler (signals "ready\n")
        # via the reader loop. Give it a moment.
        await asyncio.sleep(0.3)

        # Short timeout forces wait_for() to time out → falls to kill()
        await transport.stop(timeout=0.2)
        assert transport._proc.returncode is not None
        # -9 = SIGKILL → kill branch taken (if process ignored SIGTERM)
        assert transport._proc.returncode == -9

    @pytest.mark.asyncio
    async def test_stderr_line_drained(self) -> None:
        """Background stderr-drain reads lines without blocking the RPC flow."""
        import asyncio

        transport = _make_transport()
        await transport.start()
        try:
            result = await transport.request("stderr_line", {})
            assert result == {"ok": True}
            # give drain loop a tick
            await asyncio.sleep(0.05)
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_pending_request_rejected_on_unexpected_exit(self) -> None:
        """Reader's finally block rejects futures when subprocess dies mid-request."""
        transport = _make_transport()
        await transport.start()
        try:
            # Server will exit without responding — our request is pending
            # when reader loop sees EOF → futures rejected via
            # TransportError("Subprocess exited").
            with pytest.raises(TransportError, match="Subprocess exited"):
                await transport.request(
                    "crash_without_response", {}, timeout=5.0
                )
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_subprocess_exit_rejects_pending(self) -> None:
        """If the subprocess exits, pending requests are rejected via TransportError."""
        import asyncio

        transport = _make_transport()
        await transport.start()
        try:
            # This call makes the server exit. Next request never gets a
            # response — reader sees EOF and rejects it.
            await transport.request("exit_mid_request", {})
            # Give reader loop a moment to notice EOF
            await asyncio.sleep(0.1)

            with pytest.raises(TransportError):
                await transport.request("echo", {"x": 1}, timeout=0.5)
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_send_response_after_stop_noop(self) -> None:
        """Incoming request whose handler finishes after stop() → silently drops response."""
        import asyncio

        transport = _make_transport()

        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_handler(_params):
            started.set()
            await release.wait()
            return {"ok": True}

        transport.on_request("client/slow", slow_handler)

        await transport.start()
        task = asyncio.create_task(
            transport.request(
                "reverse_request",
                {"client_method": "client/slow", "client_params": {}},
            )
        )
        await started.wait()
        # Stop transport before handler finishes — _send_response should noop
        await transport.stop()
        release.set()
        # Outer request was cancelled; drain the task
        with pytest.raises(TransportError):
            await task

    @pytest.mark.asyncio
    async def test_send_response_broken_pipe_swallowed(
        self, monkeypatch
    ) -> None:
        """If stdin breaks during _send_response, the error is logged, not raised."""
        import asyncio

        transport = _make_transport()
        handled = asyncio.Event()

        async def handler(_params):
            handled.set()
            # Give test time to break stdin before we return
            await asyncio.sleep(0.05)
            return {"ok": True}

        transport.on_request("client/h", handler)
        await transport.start()
        try:
            task = asyncio.create_task(
                transport.request(
                    "reverse_request",
                    {"client_method": "client/h", "client_params": {}},
                )
            )
            await handled.wait()

            # Break the subprocess stdin so _send_response hits BrokenPipeError
            assert transport._proc and transport._proc.stdin

            def broken_write(_self, _data):
                raise BrokenPipeError()

            monkeypatch.setattr(
                type(transport._proc.stdin), "write", broken_write
            )
            # Outer request never completes since server never gets our
            # broken-pipe response — we just need _send_response to have run
            # and hit the except. Cancel and move on.
            await asyncio.sleep(0.2)
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, BaseException):
                pass
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_stderr_drain_exception_is_logged(
        self, monkeypatch
    ) -> None:
        """Exceptions reading stderr are swallowed by the drain loop."""
        import asyncio

        transport = _make_transport()
        await transport.start()
        try:
            assert transport._proc and transport._proc.stderr

            async def broken_readline():
                raise RuntimeError("stderr read fail")

            monkeypatch.setattr(
                transport._proc.stderr, "readline", broken_readline
            )

            # Let the drain loop iterate at least once so the exception fires
            await asyncio.sleep(0.1)

            # Transport should still be functional
            assert await transport.request("echo", {"a": 1}) == {"a": 1}
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_notify_broken_pipe_wrapped(self, monkeypatch) -> None:
        transport = _make_transport()
        await transport.start()
        try:
            assert transport._proc and transport._proc.stdin

            def broken_write(self_, _data):
                raise BrokenPipeError()

            monkeypatch.setattr(
                type(transport._proc.stdin), "write", broken_write
            )
            with pytest.raises(TransportError, match="Broken pipe"):
                await transport.notify("x", {})
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_async_notification_handler_awaited(self) -> None:
        import asyncio

        transport = _make_transport()
        received: list[dict] = []

        async def handler(params):
            await asyncio.sleep(0)  # force await branch
            received.append(params)

        transport.on_notification("client/async", handler)
        await transport.start()
        try:
            await transport.request(
                "notify_to_client",
                {"method": "client/async", "params": {"v": 7}},
            )
            await asyncio.sleep(0.1)
            assert received == [{"v": 7}]
        finally:
            await transport.stop()
