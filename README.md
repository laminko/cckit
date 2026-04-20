# cckit

An async Python framework that wraps the [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) for building local LLM agents.

> **Not affiliated with Anthropic.** "Claude" and "Claude Code" are trademarks of Anthropic, PBC. `cckit` is an independent third-party toolkit that calls the `claude` CLI binary via subprocess.

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for package management
- The `claude` CLI at `~/.local/bin/claude`

### Authentication

The wrapped `claude` CLI needs one of:

- **OAuth** — run `claude login` once. Pass `bare=False` to `CLI.execute`, `Session.create`, and any agent constructor (the library's `bare=True` default disables OAuth/keychain reads).
- **API key** — set `ANTHROPIC_API_KEY`. The default `bare=True` then works as-is.

`ACPSession` ignores this and works with either auth method out of the box.

## Getting started

```bash
# 1. Clone
git clone https://github.com/laminko/cckit.git
cd cckit

# 2. Authenticate — pick one
claude login                          # OAuth (recommended)
# export ANTHROPIC_API_KEY=sk-ant-... # or API key

# 3. Install
uv sync

# 4. Run an example
uv run python examples/basic_usage.py
```

## Quick start

```python
import asyncio
from cckit import CLI

async def main():
    cli = CLI()
    response = await cli.execute("What is 2 + 2?", bare=False)
    print(response.result)

asyncio.run(main())
```

## Core concepts

The library offers two execution paths plus one way to extend Claude with your own tools:

- **One-shot (`CLI` / `Session`)** — spawns a fresh `claude` subprocess per call. Simple, good for scripts.
- **Persistent (`ACPSession`)** — keeps a single `claude` subprocess alive and speaks JSON-RPC 2.0 over stdio. **Note:** the real `claude` binary does not emit the callback messages `ACPSession` was designed for (permissions, file I/O, elicitation); those fire only against ACP-compatible mocks or third-party agents. See [docs/concepts.md](./docs/concepts.md#execution-paths). Prefer `Session` for everyday use.
- **Custom tools (`FastMCP` + `MCPManager.add_python_server`)** — expose any Python function as a tool Claude can call. See [docs/custom-tools.md](./docs/custom-tools.md).

### One-shot execution

```python
response = await cli.execute("Summarise this file", tools=["Read"])
```

`cli.execute_json(...)` returns the raw JSON dict instead of a `Response`.

### Streaming

```python
from cckit import TextChunkEvent

async for event in cli.execute_streaming("Explain async/await"):
    if isinstance(event, TextChunkEvent):
        print(event.text, end="", flush=True)
```

Other event types: `ToolUseEvent`, `ToolResultEvent`, `MessageCompleteEvent`, `ResultEvent`.

### Multi-turn sessions

A `Session` preserves context across turns using `--resume` under the hood.

```python
from cckit import Session

session = await Session.create(cli, tools=["Read", "Edit"])
r1 = await session.send("Read auth.py")
r2 = await session.send("Find all callers of login()")  # context preserved
```

### Agents

Pre-configured `BaseAgent` subclasses bundle a tool set and system prompt.

```python
from cckit import CodeAgent, ResearchAgent, CustomAgent

# Coding (Read/Edit/Write/MultiEdit/Bash/Grep/Glob)
result = await CodeAgent().execute("Fix type errors in src/")

# Research (Read/Grep/Glob/WebSearch/WebFetch)
result = await ResearchAgent().execute("Summarise recent repo changes")

# Fully custom
agent = CustomAgent(
    name="SecurityAuditor",
    system_prompt="You are a security expert. Find vulnerabilities.",
    tools=["Read", "Grep"],
)
result = await agent.execute("Audit the auth module")
```

`ConversationAgent` is a general-purpose multi-turn chat wrapper that manages its own `Session`.

### MCP integration

```python
from pathlib import Path
from cckit import CLI, MCPManager, Session

workspace = str(Path.cwd())

mcp = MCPManager()
mcp.add_server("filesystem", "npx",
               ["-y", "@modelcontextprotocol/server-filesystem", workspace])
config_path = mcp.write_config_file()

try:
    session = await Session.create(
        CLI(),
        mcp_config_path=config_path,
        tools=["mcp__filesystem__list_directory"],
    )
    response = await session.send(
        f"Use mcp__filesystem__list_directory to list {workspace}."
    )
    print(response.result)
finally:
    mcp.cleanup()
```

#### Writing your own tools in Python

Expose any Python function as a tool Claude can call. See [`docs/custom-tools.md`](./docs/custom-tools.md) for the full guide.

```python
# my_tools.py
from cckit import FastMCP

server = FastMCP("my-tools")

@server.tool(description="Look up a user by ID.")
def lookup_user(user_id: str) -> dict:
    return {"id": user_id, "name": "Alice"}

if __name__ == "__main__":
    server.run()
```

Register it and call through cckit:

```python
mcp.add_python_server("my-tools", script="my_tools.py")
tools = ["mcp__my-tools__lookup_user"]
```

### ACP: persistent bidirectional sessions

`ACPSession` keeps one `claude` subprocess alive and speaks JSON-RPC 2.0 over stdio (Agent Communication Protocol). It was designed for agent→client callbacks — permission prompts, file reads/writes, mid-prompt cancellation. **The real `claude` CLI (verified against v2.1.114) does not emit those callbacks**; `ACPSession` behaves as designed only against ACP-compatible mocks or third-party agents. For everyday use with the real binary, prefer `Session`. See [docs/concepts.md](./docs/concepts.md#execution-paths) for the full story.

```python
from cckit import ACPSession, PermissionPolicy, TextChunkEvent

async with await ACPSession.create(
    permission_policy=PermissionPolicy.AUTO_APPROVE,
) as session:
    response = await session.send("What is 2+2?")
    print(response.result)

    async for event in session.stream("Write a haiku about async code."):
        if isinstance(event, TextChunkEvent):
            print(event.text, end="", flush=True)
```

Permission policies: `AUTO_APPROVE`, `AUTO_DENY` (default), `CALLBACK` (supply your own async/sync callback).

File read/write callbacks from the agent are confined to a workspace root (defaults to the current directory), reject symlinks, and enforce a 10 MiB read cap.

Lower-level building blocks are available: `RpcTransport`, `ACPClient`, and `PermissionPolicy` from `cckit`; `DefaultHandlers` from `cckit.rpc`.

## Project layout

```
cckit/
├── core/        # CLI, CommandBuilder, CLIConfig, SessionConfig, ACPConfig
├── streaming/   # StreamHandler, parsers, typed Events
├── session/     # Session, ACPSession, ConversationManager, MessageHistory
├── agents/      # BaseAgent, CodeAgent, ResearchAgent, ConversationAgent, CustomAgent
├── mcp/         # MCPServer, MCPManager, FastMCP (authoring helper)
├── rpc/         # RpcTransport, ACPClient, DefaultHandlers, PermissionPolicy
├── types/       # Response, AgentResult, Message, Usage, OutputFormat, PermissionMode
└── utils/       # errors (CckitError base + CLIError, AuthError, RpcError, TransportError, ...), helpers
```

## Tests

```bash
uv run pytest
```

## Examples

```bash
uv run python examples/basic_usage.py
uv run python examples/streaming_example.py
uv run python examples/multi_turn_conversation.py
uv run python examples/custom_agent.py
uv run python examples/mcp_integration.py
uv run python examples/mcp_custom_tool.py
```

## Credits

Inspired by [t3.code](https://github.com/pingdotgg/t3code).
