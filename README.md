# claude-agent

An async Python framework that wraps the [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) for building local LLM agents.

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for package management
- The `claude` CLI at `~/.local/bin/claude`

### Authentication

The wrapped `claude` CLI needs one of:

- **OAuth** — run `claude login` once. Pass `bare=False` to `ClaudeCLI.execute`, `Session.create`, and any agent constructor (the library's `bare=True` default disables OAuth/keychain reads).
- **API key** — set `ANTHROPIC_API_KEY`. The default `bare=True` then works as-is.

`ACPSession` ignores this and works with either auth method out of the box.

## Install

```bash
uv sync
```

## Quick start

```python
import asyncio
from claude_agent import ClaudeCLI

async def main():
    cli = ClaudeCLI()
    response = await cli.execute("What is 2 + 2?", bare=False)
    print(response.result)

asyncio.run(main())
```

## Core concepts

The library offers two execution paths:

- **One-shot (`ClaudeCLI` / `Session`)** — spawns a fresh `claude` subprocess per call. Simple, good for scripts.
- **Persistent (`ACPSession`)** — keeps a single `claude` subprocess alive and talks to it over JSON-RPC 2.0. Good for long conversations, permission prompts, and file callbacks.

### One-shot execution

```python
response = await cli.execute("Summarise this file", tools=["Read"])
```

`cli.execute_json(...)` returns the raw JSON dict instead of a `Response`.

### Streaming

```python
from claude_agent import TextChunkEvent

async for event in cli.execute_streaming("Explain async/await"):
    if isinstance(event, TextChunkEvent):
        print(event.text, end="", flush=True)
```

Other event types: `ToolUseEvent`, `ToolResultEvent`, `MessageCompleteEvent`, `ResultEvent`.

### Multi-turn sessions

A `Session` preserves context across turns using `--resume` under the hood.

```python
from claude_agent import Session

session = await Session.create(cli, tools=["Read", "Edit"])
r1 = await session.send("Read auth.py")
r2 = await session.send("Find all callers of login()")  # context preserved
```

### Agents

Pre-configured `BaseAgent` subclasses bundle a tool set and system prompt.

```python
from claude_agent import CodeAgent, ResearchAgent, CustomAgent

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
from claude_agent import ClaudeCLI, MCPManager, Session

workspace = str(Path.cwd())

mcp = MCPManager()
mcp.add_server("filesystem", "npx",
               ["-y", "@modelcontextprotocol/server-filesystem", workspace])
config_path = mcp.write_config_file()

try:
    session = await Session.create(
        ClaudeCLI(),
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

### ACP: persistent bidirectional sessions

`ACPSession` keeps one `claude` subprocess alive and speaks JSON-RPC 2.0 over stdio (Agent Communication Protocol). This lets the agent call back into your code for permission checks, file reads, and file writes, and lets you cancel an in-flight prompt.

```python
from claude_agent import ACPSession, PermissionPolicy, TextChunkEvent

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

Lower-level building blocks (`RpcTransport`, `ACPClient`, `DefaultHandlers`) are also exported if you need them.

## Project layout

```
claude_agent/
├── core/        # ClaudeCLI, CommandBuilder, CLIConfig, SessionConfig, ACPConfig
├── streaming/   # StreamHandler, parsers, typed Events
├── session/     # Session, ACPSession, ConversationManager, MessageHistory
├── agents/      # BaseAgent, CodeAgent, ResearchAgent, ConversationAgent, CustomAgent
├── mcp/         # MCPServer, MCPManager
├── rpc/         # RpcTransport, ACPClient, DefaultHandlers, PermissionPolicy
├── types/       # Response, AgentResult, Message, Usage, OutputFormat, PermissionMode
└── utils/       # errors (CLIError, RpcError, TransportError, ...), helpers
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
```

## Credits

Inspired by [t3.code](https://github.com/pingdotgg/t3code).
