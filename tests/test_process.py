"""Tests for ProcessManager (async subprocess wrapper)."""
from __future__ import annotations

import sys

import pytest

from cckit.core.process import ProcessManager
from cckit.utils.errors import CLIError, TimeoutError


class TestProcessRun:
    @pytest.mark.asyncio
    async def test_run_captures_stdout(self) -> None:
        pm = ProcessManager()
        stdout, stderr, code = await pm.run(
            [sys.executable, "-c", "print('hello')"]
        )
        assert "hello" in stdout
        assert code == 0

    @pytest.mark.asyncio
    async def test_run_captures_stderr(self) -> None:
        pm = ProcessManager()
        stdout, stderr, code = await pm.run(
            [sys.executable, "-c", "import sys; sys.stderr.write('oops')"]
        )
        assert "oops" in stderr
        assert code == 0

    @pytest.mark.asyncio
    async def test_run_nonzero_exit(self) -> None:
        pm = ProcessManager()
        _, _, code = await pm.run(
            [sys.executable, "-c", "import sys; sys.exit(3)"]
        )
        assert code == 3

    @pytest.mark.asyncio
    async def test_run_binary_not_found(self) -> None:
        pm = ProcessManager()
        with pytest.raises(CLIError, match="Binary not found"):
            await pm.run(["/nonexistent/binary/xyz"])

    @pytest.mark.asyncio
    async def test_run_timeout(self) -> None:
        pm = ProcessManager(timeout=0.2)
        with pytest.raises(TimeoutError, match="timed out"):
            await pm.run(
                [sys.executable, "-c", "import time; time.sleep(5)"]
            )


class TestProcessStreamLines:
    @pytest.mark.asyncio
    async def test_stream_lines_yields_lines(self) -> None:
        pm = ProcessManager()
        lines = []
        async for line in pm.stream_lines(
            [sys.executable, "-c", "print('a'); print('b'); print('c')"]
        ):
            lines.append(line)
        assert lines == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_stream_lines_binary_not_found(self) -> None:
        pm = ProcessManager()
        with pytest.raises(CLIError, match="Binary not found"):
            async for _ in pm.stream_lines(["/nonexistent/binary/xyz"]):
                pass

    @pytest.mark.asyncio
    async def test_stream_lines_drains_stderr(self) -> None:
        pm = ProcessManager()
        code = (
            "import sys; "
            "print('line1'); "
            "sys.stderr.write('err1\\n'); "
            "print('line2')"
        )
        lines = []
        async for line in pm.stream_lines([sys.executable, "-c", code]):
            lines.append(line)
        assert "line1" in lines
        assert "line2" in lines

    @pytest.mark.asyncio
    async def test_stream_lines_nonzero_exit(self) -> None:
        pm = ProcessManager()
        code = "import sys; print('out'); sys.exit(2)"
        lines = []
        async for line in pm.stream_lines([sys.executable, "-c", code]):
            lines.append(line)
        assert "out" in lines
