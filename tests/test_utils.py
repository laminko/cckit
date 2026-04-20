"""Tests for cckit.utils helpers and errors."""
from __future__ import annotations

import logging
import os

from cckit.utils.errors import (
    AuthError,
    CckitError,
    CLIError,
    ParseError,
    ProtocolError,
    RpcError,
    SessionError,
    TransportError,
)
from cckit.utils.helpers import expand_path, get_logger, safe_json_loads


class TestSafeJsonLoads:
    def test_valid_json(self) -> None:
        assert safe_json_loads('{"a": 1}') == {"a": 1}

    def test_invalid_json_returns_none(self) -> None:
        assert safe_json_loads("not json") is None


class TestExpandPath:
    def test_expands_tilde(self) -> None:
        result = expand_path("~/cckit-unit-test")
        assert "~" not in result
        assert result.endswith("cckit-unit-test")

    def test_expands_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("CCKIT_TEST_VAR", "/tmp/cckit-test")
        result = expand_path("$CCKIT_TEST_VAR/foo")
        assert result.endswith("/foo")


class TestGetLogger:
    def test_returns_logger_with_handler(self) -> None:
        logger = get_logger("cckit.test.utils")
        assert isinstance(logger, logging.Logger)
        assert logger.handlers

    def test_idempotent(self) -> None:
        logger1 = get_logger("cckit.test.utils.idem")
        count = len(logger1.handlers)
        logger2 = get_logger("cckit.test.utils.idem")
        # Handler count should not grow on repeated calls
        assert len(logger2.handlers) == count


class TestErrors:
    def test_cli_error_carries_metadata(self) -> None:
        err = CLIError("boom", exit_code=2, stderr="oops")
        assert err.exit_code == 2
        assert err.stderr == "oops"

    def test_auth_error_is_cli_error(self) -> None:
        assert issubclass(AuthError, CLIError)

    def test_parse_error_carries_raw(self) -> None:
        err = ParseError("bad", raw="<junk>")
        assert err.raw == "<junk>"

    def test_rpc_error_has_code_and_data(self) -> None:
        err = RpcError("oops", code=-32000, data={"x": 1})
        assert err.code == -32000
        assert err.data == {"x": 1}

    def test_all_subclass_cckit_error(self) -> None:
        for cls in (
            CLIError,
            AuthError,
            SessionError,
            ParseError,
            TransportError,
            RpcError,
            ProtocolError,
        ):
            assert issubclass(cls, CckitError)
