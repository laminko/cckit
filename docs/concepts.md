# Concepts

The mental model behind `cckit`. Read this once and the rest of the API follows.

## What cckit is

`cckit` is an async Python wrapper around the `claude` CLI binary. It does not call the Anthropic API directly — every call shells out to `~/.local/bin/claude`. That single fact explains most of the design:

- **Auth lives in the CLI**, not in `cckit` — see [the `bare` flag](#the-bare-flag).
- **Tool permissions, MCP servers, system prompts** are CLI flags, surfaced as Python kwargs.
- **Two transport modes** for the subprocess — one-shot per call vs. one long-lived process — drive the library's two session types.

## Execution paths

| | Subprocess model | Context across turns | When to use |
|---|---|---|---|
| `CLI` | New subprocess per call | No | Scripts, one-off prompts |
| `Session` | New subprocess per turn (uses `--resume`) | Yes | Multi-turn conversations |

**About `ACPSession`:** the library also ships an `ACPSession` class that speaks a bidirectional JSON-RPC 2.0 protocol over stdio (the "Agent Communication Protocol"). It was designed for agent→client callbacks — tool-use permission prompts, file read/write asks, cancellation mid-prompt. In practice, the current `claude` CLI (verified against v2.1.114) does **not** emit those callbacks; it speaks a one-way stream-json protocol. `ACPSession` therefore only behaves as advertised against an ACP-compatible mock or a third-party agent that implements the protocol. For everyday use, prefer `Session`.

## The `bare` flag

`bare=True` (the default on `CLI.execute`, `Session.create`, and every agent constructor) adds `--bare` to the spawned `claude` invocation. `--bare` tells `claude` to skip reading the macOS keychain / config files where OAuth credentials live.

The consequence:

| Auth method | `bare=True` (default) | `bare=False` |
|---|---|---|
| OAuth (`claude login`) | ❌ fails | ✅ works |
| `ANTHROPIC_API_KEY` env var | ✅ works | ✅ works |

So: **API-key users can leave the default**; **OAuth users must pass `bare=False`** on every entry point that accepts it.

`ACPSession` does not accept a `bare` parameter — it constructs its own invocation without `--bare`. Whether authentication succeeds depends on what the ACP server on the other end expects; see the caveat on `ACPSession` above.

## Streaming and events

Every streaming call yields the same `Event` union. Pattern-match with `isinstance`:

```python
from cckit import TextChunkEvent, ToolUseEvent, ResultEvent

async for event in cli.execute_streaming("Summarise auth.py"):
    if isinstance(event, TextChunkEvent):
        print(event.text, end="", flush=True)
    elif isinstance(event, ToolUseEvent):
        print(f"\n[using {event.tool_name}]")
    elif isinstance(event, ResultEvent):
        print(f"\n[done in {event.duration_ms}ms]")
```

Full event list:

- `TextChunkEvent(text)` — a delta of assistant text.
- `ToolUseEvent(tool_name, tool_input, tool_use_id)` — the assistant is invoking a tool.
- `ToolResultEvent(tool_use_id, content, is_error)` — result of a tool call.
- `MessageStartEvent(role)` / `MessageCompleteEvent(stop_reason, session_id)` — message boundaries.
- `UsageEvent(input_tokens, output_tokens, cache_read_tokens, cache_write_tokens)` — token counts.
- `ResultEvent(result, session_id, duration_ms, is_error)` — final turn result.
- `SystemEvent(subtype)` — misc.

Non-streaming calls (`execute`, `send`) collect the stream internally and return a `Response` with the accumulated text and metadata.

## Permission policies

`DefaultHandlers` (in `cckit.rpc`) implements three policies — `AUTO_DENY` (default), `AUTO_APPROVE`, `CALLBACK` — for responding to `session/request_permission` JSON-RPC requests from the agent. They fire only when something on the other end of the ACP transport actually sends those requests. The real `claude` binary does not, so these policies only matter against ACP-compatible mocks or third-party agents.

For real tool gating against `claude`, use CLI-level flags instead:

- `tools=["Read", "Grep", ...]` — explicit allowlist (maps to `--allowedTools`).
- `disallowed_tools=["Bash"]` — explicit denylist (maps to `--disallowedTools`).
- `permission_mode=PermissionMode.BYPASS_PERMISSIONS` — skip every gate (maps to `--permission-mode bypassPermissions`). Also available: `ACCEPT_EDITS`, `PLAN`, `DEFAULT`.

## Filesystem sandboxing

When `DefaultHandlers` *does* receive `fs/read_text_file` or `fs/write_text_file` requests, it confines paths to a `workspace_root` (default `Path.cwd()`), rejects symlinks, and caps reads at 10 MiB. Override via `DefaultHandlers(workspace_root="/some/dir")`.

As with permission policies, these handlers only fire against ACP-compatible peers — the real `claude` binary performs filesystem I/O inside its own process and does not ask the client for file contents.

## Tool names

Tool kwargs (`tools=[...]`) are CLI tool identifiers, passed through verbatim. Common ones:

- **Code**: `Read`, `Edit`, `Write`, `MultiEdit`, `Bash`, `Grep`, `Glob`
- **Research**: `WebSearch`, `WebFetch`
- **MCP**: `mcp__<server>__<tool>` (e.g. `mcp__filesystem__list_directory`). Write your own with [custom-tools.md](./custom-tools.md).

The built-in agents (`CodeAgent`, `ResearchAgent`) bundle sensible tool sets; `CustomAgent` lets you pick your own.

## Error hierarchy

Everything inherits from `CckitError`:

```
CckitError
├── CLIError            # non-zero CLI exit code
│   └── AuthError       # authentication failure
├── SessionError        # session lifecycle error
├── TimeoutError        # CLI timeout
├── ParseError          # stream-json could not be parsed
├── TransportError      # RPC transport / pipe error
├── RpcError            # JSON-RPC error response from agent
└── ProtocolError       # JSON-RPC protocol violation
```

Catch `CckitError` to handle every library error; catch the specific subclass to handle one case.
