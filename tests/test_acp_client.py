"""Tests for ACPClient using the echo_rpc.py fixture as a mock server."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cckit.rpc.client import ACPClient
from cckit.rpc.handlers import DefaultHandlers, PermissionPolicy
from cckit.rpc.transport import RpcTransport

ECHO_SERVER = str(Path(__file__).parent / "fixtures" / "echo_rpc.py")


def _make_client(
    permission_policy: PermissionPolicy = PermissionPolicy.AUTO_APPROVE,
) -> ACPClient:
    transport = RpcTransport([sys.executable, ECHO_SERVER])
    handlers = DefaultHandlers(permission_policy=permission_policy)
    return ACPClient(transport, handlers=handlers)


class TestACPClientLifecycle:
    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        client = _make_client()
        async with client:
            result = await client.initialize()
            assert "capabilities" in result

    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        client = _make_client()
        await client.transport.start()
        try:
            result = await client.initialize(
                client_info={"name": "test", "version": "0.0.1"}
            )
            assert isinstance(result, dict)
        finally:
            await client.transport.stop()


class TestACPClientHandlerRegistration:
    @pytest.mark.asyncio
    async def test_handlers_registered_on_transport(self) -> None:
        client = _make_client()
        transport = client.transport
        assert "session/request_permission" in transport._request_handlers
        assert "fs/read_text_file" in transport._request_handlers
        assert "fs/write_text_file" in transport._request_handlers
        assert "session/elicitation" in transport._request_handlers
        assert "session/update" in transport._notification_handlers


class TestACPClientSessionFlow:
    @pytest.mark.asyncio
    async def test_initialize_with_capabilities(self) -> None:
        client = _make_client()
        async with client:
            result = await client.initialize(
                client_info={"name": "t", "version": "0.1"},
                capabilities={"tools": True},
            )
            assert "capabilities" in result

    @pytest.mark.asyncio
    async def test_new_session_params_forwarded(self) -> None:
        client = _make_client()
        async with client:
            await client.initialize()
            sid = await client.new_session(
                model="opus", system_prompt="short", cwd="/tmp"
            )
            assert sid == "sid-echo"
            assert client.session_id == "sid-echo"

    @pytest.mark.asyncio
    async def test_new_session_handles_non_dict_result(
        self, monkeypatch
    ) -> None:
        client = _make_client()
        async with client:
            await client.initialize()

            async def fake_request(method, params=None, **kw):
                return "not-a-dict"

            monkeypatch.setattr(client._transport, "request", fake_request)
            sid = await client.new_session()
            assert sid == ""

    @pytest.mark.asyncio
    async def test_load_session_sets_id(self) -> None:
        client = _make_client()
        async with client:
            await client.initialize()
            result = await client.load_session("abc-sid")
            assert result["loaded"] == "abc-sid"
            assert client.session_id == "abc-sid"

    @pytest.mark.asyncio
    async def test_close_session_clears_id(self) -> None:
        client = _make_client()
        async with client:
            await client.initialize()
            await client.load_session("sid-x")
            assert client.session_id == "sid-x"
            await client.close_session()
            assert client.session_id is None

    @pytest.mark.asyncio
    async def test_close_session_without_id_is_noop(self) -> None:
        client = _make_client()
        async with client:
            await client.initialize()
            await client.close_session()  # no active session — noop

    @pytest.mark.asyncio
    async def test_prompt_without_session_raises(self) -> None:
        from cckit.utils.errors import SessionError

        client = _make_client()
        async with client:
            with pytest.raises(SessionError, match="No active session"):
                await client.prompt("hi")

    @pytest.mark.asyncio
    async def test_prompt_with_session(self) -> None:
        client = _make_client()
        async with client:
            await client.initialize()
            await client.load_session("sid-x")
            await client.prompt("hello")  # should not raise

    @pytest.mark.asyncio
    async def test_cancel_with_session_notifies(self) -> None:
        client = _make_client()
        async with client:
            await client.initialize()
            await client.load_session("sid-x")
            # Should not raise (fire-and-forget)
            await client.cancel()

    @pytest.mark.asyncio
    async def test_cancel_without_session_noop(self) -> None:
        client = _make_client()
        async with client:
            await client.cancel()


class TestACPClientSessionUpdateCallbacks:
    @pytest.mark.asyncio
    async def test_on_session_update_callback(self) -> None:
        client = _make_client()
        received = []
        client.on_session_update(lambda params: received.append(params))

        client._handle_session_update(
            {"type": "content_delta", "text": "hello"}
        )
        assert len(received) == 1
        assert received[0]["type"] == "content_delta"

    @pytest.mark.asyncio
    async def test_multiple_callbacks(self) -> None:
        client = _make_client()
        results_a = []
        results_b = []
        client.on_session_update(lambda p: results_a.append(p))
        client.on_session_update(lambda p: results_b.append(p))

        client._handle_session_update({"type": "test"})
        assert len(results_a) == 1
        assert len(results_b) == 1

    @pytest.mark.asyncio
    async def test_remove_session_update(self) -> None:
        client = _make_client()
        received = []
        cb = lambda p: received.append(p)  # noqa: E731
        client.on_session_update(cb)
        client._handle_session_update({"type": "a"})
        assert len(received) == 1

        client.remove_session_update(cb)
        client._handle_session_update({"type": "b"})
        assert len(received) == 1  # no new events

    def test_remove_unregistered_callback_is_safe(self) -> None:
        client = _make_client()
        client.remove_session_update(lambda p: None)  # no raise

    def test_callback_exception_is_swallowed(self) -> None:
        client = _make_client()
        good = []

        def bad(_):
            raise RuntimeError("callback fail")

        client.on_session_update(bad)
        client.on_session_update(lambda p: good.append(p))

        client._handle_session_update({"type": "x"})
        assert len(good) == 1  # good callback still fired


class TestDefaultHandlers:
    @pytest.mark.asyncio
    async def test_default_is_auto_deny(self) -> None:
        handlers = DefaultHandlers()
        result = await handlers.handle_permission({"tool_name": "Bash"})
        assert result["approved"] is False

    @pytest.mark.asyncio
    async def test_auto_approve_permission(self) -> None:
        handlers = DefaultHandlers(
            permission_policy=PermissionPolicy.AUTO_APPROVE
        )
        result = await handlers.handle_permission({"tool_name": "Bash"})
        assert result["approved"] is True

    @pytest.mark.asyncio
    async def test_auto_deny_permission(self) -> None:
        handlers = DefaultHandlers(permission_policy=PermissionPolicy.AUTO_DENY)
        result = await handlers.handle_permission({"tool_name": "Bash"})
        assert result["approved"] is False

    @pytest.mark.asyncio
    async def test_callback_permission(self) -> None:
        async def my_callback(params):
            return {"approved": params.get("tool_name") == "Read"}

        handlers = DefaultHandlers(
            permission_policy=PermissionPolicy.CALLBACK,
            permission_callback=my_callback,
        )
        result = await handlers.handle_permission({"tool_name": "Read"})
        assert result["approved"] is True

        result = await handlers.handle_permission({"tool_name": "Write"})
        assert result["approved"] is False

    def test_callback_without_callback_raises(self) -> None:
        with pytest.raises(ValueError, match="permission_callback is required"):
            DefaultHandlers(permission_policy=PermissionPolicy.CALLBACK)

    @pytest.mark.asyncio
    async def test_file_read_outside_workspace(self) -> None:
        handlers = DefaultHandlers(workspace_root="/tmp/safe")
        result = await handlers.handle_file_read({"path": "/etc/passwd"})
        assert "error" in result
        assert "outside workspace" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_file_read_nonexistent(self, tmp_path) -> None:
        handlers = DefaultHandlers(workspace_root=tmp_path)
        result = await handlers.handle_file_read(
            {"path": str(tmp_path / "nonexistent.txt")}
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_file_read_existing(self, tmp_path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        handlers = DefaultHandlers(workspace_root=tmp_path)
        result = await handlers.handle_file_read({"path": str(test_file)})
        assert result["content"] == "hello world"

    @pytest.mark.asyncio
    async def test_file_read_symlink_rejected(self, tmp_path) -> None:
        real_file = tmp_path / "real.txt"
        real_file.write_text("secret")
        link = tmp_path / "link.txt"
        link.symlink_to(real_file)

        handlers = DefaultHandlers(workspace_root=tmp_path)
        result = await handlers.handle_file_read({"path": str(link)})
        assert "error" in result
        assert "symlink" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_file_write(self, tmp_path) -> None:
        target = tmp_path / "output.txt"
        handlers = DefaultHandlers(workspace_root=tmp_path)
        result = await handlers.handle_file_write(
            {"path": str(target), "content": "written"}
        )
        assert result["success"] is True
        assert target.read_text() == "written"

    @pytest.mark.asyncio
    async def test_file_write_outside_workspace(self, tmp_path) -> None:
        handlers = DefaultHandlers(workspace_root=tmp_path)
        result = await handlers.handle_file_write(
            {"path": "/tmp/evil.txt", "content": "bad"}
        )
        assert "error" in result
        assert "outside workspace" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_file_read_path_traversal(self, tmp_path) -> None:
        handlers = DefaultHandlers(workspace_root=tmp_path)
        result = await handlers.handle_file_read(
            {"path": str(tmp_path / ".." / ".." / "etc" / "passwd")}
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_elicitation_declined(self) -> None:
        handlers = DefaultHandlers()
        result = await handlers.handle_elicitation(
            {"message": "Enter API key:"}
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_permission_unknown_policy_fails_closed(self) -> None:
        handlers = DefaultHandlers()
        handlers.permission_policy = object()  # type: ignore[assignment]
        result = await handlers.handle_permission({"tool_name": "Bash"})
        assert result["approved"] is False
        assert "unknown" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_permission_sync_callback(self) -> None:
        handlers = DefaultHandlers(
            permission_policy=PermissionPolicy.CALLBACK,
            permission_callback=lambda p: {"approved": True, "reason": "ok"},
        )
        result = await handlers.handle_permission({"tool_name": "Read"})
        assert result["approved"] is True

    @pytest.mark.asyncio
    async def test_file_read_too_large(self, tmp_path, monkeypatch) -> None:
        target = tmp_path / "big.txt"
        target.write_text("small")
        monkeypatch.setattr("cckit.rpc.handlers.MAX_READ_SIZE", 1)

        handlers = DefaultHandlers(workspace_root=tmp_path)
        result = await handlers.handle_file_read({"path": str(target)})
        assert "too large" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_file_read_io_error(self, tmp_path) -> None:
        target = tmp_path / "x.txt"
        target.write_bytes(b"\xff\xfe\xfd")  # non-utf8 content

        handlers = DefaultHandlers(workspace_root=tmp_path)
        # Force exception by making read_text fail — we'll simulate with
        # a directory marked as file-like through monkeypatching.
        # Simpler: write valid content, but monkeypatch read_text to raise.
        import cckit.rpc.handlers as mod

        original = mod.Path.read_text

        def boom(self, *a, **kw):  # type: ignore[no-untyped-def]
            raise OSError("permission denied")

        mod.Path.read_text = boom  # type: ignore[assignment]
        try:
            result = await handlers.handle_file_read({"path": str(target)})
            assert "failed to read" in result["error"].lower()
        finally:
            mod.Path.read_text = original  # type: ignore[assignment]

    @pytest.mark.asyncio
    async def test_file_write_io_error(self, tmp_path) -> None:
        import cckit.rpc.handlers as mod

        handlers = DefaultHandlers(workspace_root=tmp_path)
        target = tmp_path / "out.txt"

        original = mod.Path.write_text

        def boom(self, *a, **kw):  # type: ignore[no-untyped-def]
            raise OSError("disk full")

        mod.Path.write_text = boom  # type: ignore[assignment]
        try:
            result = await handlers.handle_file_write(
                {"path": str(target), "content": "x"}
            )
            assert "failed to write" in result["error"].lower()
        finally:
            mod.Path.write_text = original  # type: ignore[assignment]

    @pytest.mark.asyncio
    async def test_file_read_empty_path(self) -> None:
        handlers = DefaultHandlers()
        result = await handlers.handle_file_read({"path": ""})
        assert "empty path" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_resolve_os_error_is_wrapped(
        self, tmp_path, monkeypatch
    ) -> None:
        """Path.resolve(OSError) → returns {'error': ...} instead of crashing."""
        import cckit.rpc.handlers as mod

        # Build the handler first so __init__'s resolve() isn't intercepted
        handlers = DefaultHandlers(workspace_root=tmp_path)

        original_resolve = mod.Path.resolve

        def boom(self, *a, **kw):  # type: ignore[no-untyped-def]
            raise OSError("io fail")

        mod.Path.resolve = boom  # type: ignore[assignment]
        try:
            result = await handlers.handle_file_read({"path": "/tmp/x"})
            assert "cannot resolve" in result["error"].lower()
        finally:
            mod.Path.resolve = original_resolve  # type: ignore[assignment]
