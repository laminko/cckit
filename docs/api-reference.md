# API reference

Every public symbol exported from `cckit`, grouped by subpackage. Signatures are copied from source; defaults reflect the current release.

All asynchronous methods are marked `async`. Keyword-only parameters appear after `*`.

---

## `cckit.core`

### `class CLI`

```python
CLI(binary_path: str = "~/.local/bin/claude",
    timeout: float | None = None,
    config: CLIConfig | None = None)
```
Async wrapper around the `claude` CLI binary. Each call spawns a fresh subprocess.

```python
async CLI.execute(prompt: str, *,
                  session_config: SessionConfig | None = None,
                  resume: str | None = None,
                  **kwargs) -> Response
```
Run a one-shot prompt and return the collected `Response`. Extra kwargs are forwarded as `SessionConfig` fields (`tools`, `model`, `bare`, etc.).

```python
async CLI.execute_streaming(prompt, *, session_config=None, resume=None, **kwargs)
    -> AsyncIterator[Event]
```
Yield typed `Event`s as the CLI streams them.

```python
async CLI.execute_json(prompt, *, session_config=None, resume=None, **kwargs) -> dict
```
Return the raw JSON dict produced by `--output-format json`.

### `class CommandBuilder`

Internal builder that converts a `SessionConfig` into the argv list passed to `claude`. Exposed primarily for testing and extension.

### `dataclass CLIConfig`

| Field | Type | Default |
|---|---|---|
| `binary_path` | `str` | `"~/.local/bin/claude"` |
| `timeout` | `float \| None` | `None` |
| `default_model` | `str \| None` | `None` |
| `default_permission_mode` | `PermissionMode` | `PermissionMode.DEFAULT` |
| `extra_flags` | `list[str]` | `[]` |

### `dataclass SessionConfig`

| Field | Type | Default |
|---|---|---|
| `tools` | `list[str] \| None` | `None` |
| `disallowed_tools` | `list[str] \| None` | `None` |
| `model` | `str \| None` | `None` |
| `permission_mode` | `PermissionMode` | `PermissionMode.DEFAULT` |
| `system_prompt` | `str \| None` | `None` |
| `append_system_prompt` | `str \| None` | `None` |
| `mcp_config_path` | `str \| None` | `None` |
| `max_turns` | `int \| None` | `None` |
| `max_budget_usd` | `float \| None` | `None` |
| `cwd` | `str \| None` | `None` |
| `bare` | `bool` | `True` |
| `verbose` | `bool` | `False` |

### `dataclass ACPConfig`

| Field | Type | Default |
|---|---|---|
| `binary_path` | `str` | `"~/.local/bin/claude"` |
| `model` | `str \| None` | `None` |
| `permission_policy` | `str` | `"auto_approve"` |
| `client_name` | `str` | `"cckit"` |
| `client_version` | `str` | `"0.1.0"` |
| `request_timeout` | `float` | `30.0` |
| `shutdown_timeout` | `float` | `5.0` |

> Note: when `ACPConfig` is passed to `ACPSession.create(config=...)`, its `permission_policy` default is `"auto_approve"`. When it is *not* passed, the `permission_policy` kwarg on `create()` defaults to `AUTO_DENY`. These two code paths intentionally differ.

---

## `cckit.session`

### `class Session`

Multi-turn session backed by CLI `--resume`. One subprocess per `send()`.

```python
@classmethod
async Session.create(cli: CLI, *,
                     tools: list[str] | None = None,
                     model: str | None = None,
                     system_prompt: str | None = None,
                     bare: bool = True,
                     **kwargs) -> Session
```

```python
@classmethod
Session.resume(cli: CLI, session_id: str, config: SessionConfig | None = None) -> Session
```
Re-attach to an existing CLI session.

```python
async Session.send(message: str) -> Response
async Session.stream(message: str) -> AsyncIterator[Event]
Session.get_history() -> list[Message]
Session.fork() -> Session              # copy config + session_id + history
Session.clear_history() -> None
Session.session_id: str                # populated after first turn
```

### `class ACPSession`

Persistent session over JSON-RPC 2.0 stdio. One long-lived subprocess. Supports `async with`.

```python
@classmethod
async ACPSession.create(*,
    binary_path: str = "~/.local/bin/claude",
    model: str | None = None,
    system_prompt: str | None = None,
    cwd: str | None = None,
    permission_policy: PermissionPolicy = PermissionPolicy.AUTO_DENY,
    config: ACPConfig | None = None) -> ACPSession
```

```python
@classmethod
async ACPSession.connect(session_id: str, *,
    binary_path: str = "~/.local/bin/claude",
    permission_policy: PermissionPolicy = PermissionPolicy.AUTO_DENY) -> ACPSession
```

```python
async ACPSession.send(message: str) -> Response
async ACPSession.stream(message: str) -> AsyncIterator[Event]
async ACPSession.cancel() -> None       # cancel in-flight prompt
async ACPSession.close() -> None
ACPSession.session_id: str              # property
```

### `class MessageHistory`

```python
MessageHistory(max_messages: int | None = None)
MessageHistory.add(Message) | add_user(str) | add_assistant(str, tool_uses=None)
MessageHistory.get_all() -> list[Message]
MessageHistory.last_assistant() -> Message | None
MessageHistory.clear()
MessageHistory.export() -> list[dict]
MessageHistory.save(path) / MessageHistory.load(path) -> MessageHistory
```

### `class ConversationManager`

Named registry of `Session`s backed by a single `CLI`.

```python
ConversationManager(cli: CLI)
async ConversationManager.new_session(name=None, *, config=None, **kwargs) -> Session
ConversationManager.get(name) -> Session | None
ConversationManager.resume(session_id, name=None) -> Session
ConversationManager.list_sessions() -> list[str]
ConversationManager.remove(name)
ConversationManager.clear()
```

---

## `cckit.agents`

All agents share the `BaseAgent` interface.

### `class BaseAgent` (abstract)

```python
BaseAgent(cli: CLI | None = None, *,
         model: str | None = None,
         tools: list[str] | None = None,
         system_prompt: str | None = None,
         bare: bool = True,
         binary_path: str = "~/.local/bin/claude")

async BaseAgent.execute(task: str) -> AgentResult
async BaseAgent.stream_execute(task: str) -> AsyncIterator[Event]
async BaseAgent.chat(session: Session | None = None) -> Session     # returns a configured Session
BaseAgent.with_config(*, model=None, tools=None, system_prompt=None, bare=None) -> BaseAgent
```

Subclasses must implement `get_default_tools()` and `get_system_prompt()`.

### `class CodeAgent`
Default tools: `["Read", "Edit", "Write", "MultiEdit", "Bash", "Grep", "Glob"]`.

### `class ResearchAgent`
Default tools: `["Read", "Grep", "Glob", "WebSearch", "WebFetch"]`.

### `class ConversationAgent`
No tools by default. Manages its own `Session`. **Overrides `chat()`** with a message-sending signature:

```python
async ConversationAgent.start() -> ConversationAgent
async ConversationAgent.chat(message: str) -> AgentResult
ConversationAgent.get_session() -> Session | None
ConversationAgent.reset() -> None
```

### `class CustomAgent`
Fully user-defined.

```python
CustomAgent(name: str = "CustomAgent", *,
           cli: CLI | None = None,
           system_prompt: str = "",
           tools: list[str] | None = None,
           model: str | None = None,
           bare: bool = True,
           binary_path: str = "~/.local/bin/claude")
```

---

## `cckit.mcp`

### `class MCPManager`

```python
MCPManager()
MCPManager.add_server(name: str, command: str,
                      args: list[str] | None = None,
                      env: dict[str, str] | None = None) -> MCPManager    # chainable
MCPManager.add_python_server(name: str, *,
                             script: str | Path | None = None,
                             module: str | None = None,
                             python: str | None = None,         # defaults to sys.executable
                             args: list[str] | None = None,
                             env: dict[str, str] | None = None) -> MCPManager   # chainable
MCPManager.remove_server(name: str) -> None
MCPManager.to_config() -> dict                             # {"mcpServers": {...}}
MCPManager.write_config_file(path=None) -> str             # path (temp file if None)
MCPManager.cleanup() -> None                               # deletes temp file
```

`add_python_server` is a convenience for registering a Python-authored MCP server without hand-writing the `python -m module` / `python path/to/script.py` invocation. Pass exactly one of `script=` or `module=`.

### `class MCPServer`

Immutable holder for one server's config: `name`, `command`, `args`, `env`. Produced by `MCPManager.add_server`.

### `class FastMCP`

Re-exported from [`mcp.server.fastmcp`](https://pypi.org/project/mcp/). Use it to define tools in your own Python file; the file then becomes a subprocess Claude calls over stdio.

```python
FastMCP(name: str | None = None, *,
        instructions: str | None = None,
        ...)                                                      # many SDK-level kwargs; see the SDK for the full list

@server.tool(name: str | None = None,                             # defaults to the function name
             title: str | None = None,
             description: str | None = None,
             structured_output: bool | None = None) -> Callable   # decorator
server.add_tool(fn, name=..., description=...) -> None            # imperative equivalent
server.run(transport: "stdio" | "sse" | "streamable-http" = "stdio",
           mount_path: str | None = None) -> None                 # run loop; stdio is what cckit invokes
```

Usage shape:

```python
from cckit import FastMCP

server = FastMCP("my-tools")

@server.tool(description="...")
def my_tool(arg: str) -> dict:
    ...

if __name__ == "__main__":
    server.run()
```

- **Schema auto-generation** — parameter JSON schema is derived from type hints; Pydantic models become nested object schemas.
- **Async tools** — `async def` functions are awaited natively.
- **Errors** — raised exceptions are converted into MCP error responses; no manual error envelope needed.

See [`docs/custom-tools.md`](./custom-tools.md) for the full pattern and common pitfalls.

---

## `cckit.rpc`

### `class RpcTransport`

Bidirectional JSON-RPC 2.0 transport over an `asyncio.subprocess`. Used internally by `ACPSession`; exposed for advanced use.

```python
RpcTransport(cmd: list[str])
async RpcTransport.start() / stop()
async RpcTransport.request(method, params) -> Any
async RpcTransport.notify(method, params) -> None
RpcTransport.on_request(method, handler)
RpcTransport.on_notification(method, handler)
```

### `class ACPClient`

Typed ACP protocol wrapper around `RpcTransport`.

```python
ACPClient(transport: RpcTransport, handlers: DefaultHandlers | None = None)
async ACPClient.initialize(client_info=None, capabilities=None) -> dict
async ACPClient.new_session(*, model=None, system_prompt=None, cwd=None) -> str
async ACPClient.load_session(session_id) -> dict
async ACPClient.close_session() -> None
async ACPClient.prompt(message: str) -> None         # response streams via session/update
async ACPClient.cancel() -> None
ACPClient.on_session_update(cb) / remove_session_update(cb)
ACPClient.session_id: str | None
ACPClient.transport: RpcTransport
```

### `enum PermissionPolicy`

- `AUTO_APPROVE = "auto_approve"`
- `AUTO_DENY = "auto_deny"`
- `CALLBACK = "callback"` — requires `permission_callback` on `DefaultHandlers`.

### `class DefaultHandlers` (from `cckit.rpc`, not re-exported top-level)

```python
DefaultHandlers(
    permission_policy: PermissionPolicy = PermissionPolicy.AUTO_DENY,
    permission_callback: Callable | None = None,    # required if policy is CALLBACK
    workspace_root: Path | str | None = None,       # defaults to Path.cwd()
)
```

Handles agent→client requests: `session/request_permission`, `fs/read_text_file`, `fs/write_text_file`, `session/elicitation`. File I/O is confined to `workspace_root`, rejects symlinks, caps reads at 10 MiB.

> **Note:** the real `claude` binary does not emit these requests (verified against v2.1.114); these handlers only fire against ACP-compatible mocks or third-party agents. See [concepts.md](./concepts.md#permission-policies).

---

## `cckit.streaming`

### Events

All events inherit from `BaseEvent` (carries the raw dict on `.raw`).

| Event | Fields |
|---|---|
| `TextChunkEvent` | `text: str` |
| `ToolUseEvent` | `tool_name`, `tool_input: dict`, `tool_use_id` |
| `ToolResultEvent` | `tool_use_id`, `content: str`, `is_error: bool` |
| `MessageStartEvent` | `role: str` |
| `MessageCompleteEvent` | `stop_reason: str`, `session_id: str` |
| `UsageEvent` | `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens` |
| `ResultEvent` | `result: str`, `session_id`, `duration_ms`, `is_error` |
| `SystemEvent` | `subtype: str` |

`Event` is the union of all of the above.

### `class StreamHandler`

Parses NDJSON lines and emits typed events. Exposed for custom pipelines.

---

## `cckit.types`

### `dataclass Response`

| Field | Type | Default |
|---|---|---|
| `result` | `str` | — |
| `session_id` | `str` | `""` |
| `duration_ms` | `int` | `0` |
| `usage` | `Usage` | `Usage()` |
| `stop_reason` | `str` | `"end_turn"` |
| `model_usage` | `dict` | `{}` |
| `is_error` | `bool` | `False` |

Factory: `Response.from_json(data)`, `Response.error(message)`.

### `dataclass AgentResult`

- `response: Response`
- `summary: str = ""`
- `artifacts: list[str] = []`
- Properties: `.result`, `.session_id` (forwarded from `response`).

### `dataclass Message`

- `role: str` (`"user"` | `"assistant"`)
- `content: str`
- `timestamp: datetime`
- `tool_uses: list[ToolUse]`

### `dataclass Usage`

`input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens` (all `int`, default `0`). Factory: `Usage.from_dict(data)`.

### `enum OutputFormat`

- `TEXT = "text"`
- `JSON = "json"`
- `STREAM_JSON = "stream-json"`

### `enum PermissionMode`

- `DEFAULT`, `ACCEPT_EDITS`, `DONT_ASK`, `BYPASS`, `PLAN`

---

## `cckit.utils` — errors

```
CckitError
├── CLIError(message, exit_code=None, stderr="")
│   └── AuthError
├── SessionError
├── TimeoutError
├── ParseError(message, raw="")
├── TransportError
├── RpcError(message, code=0, data=None)
└── ProtocolError
```
