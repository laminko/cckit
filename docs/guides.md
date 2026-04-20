# Guides

Task-oriented recipes. Each snippet is runnable; longer variants live in [`examples/`](../examples/).

All snippets assume you've authenticated per [getting-started.md](./getting-started.md). OAuth users: add `bare=False` to the relevant calls.

## Multi-turn conversation

`Session` preserves context between `send()` calls using the CLI's `--resume` flag.

```python
import asyncio
from cckit import CLI, Session

async def main():
    cli = CLI()
    session = await Session.create(cli, tools=["Read"], bare=False)
    r1 = await session.send("Read auth.py")
    r2 = await session.send("List all callers of login()")  # context preserved
    print(r2.result)

asyncio.run(main())
```

Full example: [`examples/multi_turn_conversation.py`](../examples/multi_turn_conversation.py)

## Streaming output

Yield events as they arrive. Use `isinstance` to discriminate.

```python
from cckit import CLI, TextChunkEvent, ToolUseEvent

cli = CLI()
async for event in cli.execute_streaming("Explain async/await", bare=False):
    if isinstance(event, TextChunkEvent):
        print(event.text, end="", flush=True)
    elif isinstance(event, ToolUseEvent):
        print(f"\n[{event.tool_name}]", flush=True)
```

Full example: [`examples/streaming_example.py`](../examples/streaming_example.py)

## Pre-configured agents

`CodeAgent`, `ResearchAgent`, and `ConversationAgent` bundle a tool set and a system prompt.

```python
from cckit import CodeAgent, ResearchAgent

# Tools: Read, Edit, Write, MultiEdit, Bash, Grep, Glob
result = await CodeAgent(bare=False).execute("Fix type errors in src/")

# Tools: Read, Grep, Glob, WebSearch, WebFetch
result = await ResearchAgent(bare=False).execute("Summarise recent repo changes")
```

`ConversationAgent` manages its own `Session` — use `chat()` repeatedly:

```python
from cckit import ConversationAgent

agent = ConversationAgent(bare=False)
r1 = await agent.chat("Remember the number 42")
r2 = await agent.chat("What number did I ask you to remember?")
```

## Custom agent

```python
from cckit import CustomAgent

auditor = CustomAgent(
    name="SecurityAuditor",
    system_prompt="You are a security expert. Find vulnerabilities.",
    tools=["Read", "Grep"],
    bare=False,
)
result = await auditor.execute("Audit the auth module")
```

Full example: [`examples/custom_agent.py`](../examples/custom_agent.py)

## MCP integration

Register MCP servers with `MCPManager`, write the config file, pass its path to `Session.create`.

```python
import asyncio
from pathlib import Path
from cckit import CLI, MCPManager, Session

async def main():
    workspace = str(Path.cwd())

    mcp = MCPManager()
    mcp.add_server(
        "filesystem",
        "npx",
        ["-y", "@modelcontextprotocol/server-filesystem", workspace],
    )
    config_path = mcp.write_config_file()

    try:
        session = await Session.create(
            CLI(),
            bare=False,
            mcp_config_path=config_path,
            tools=["mcp__filesystem__list_directory"],
        )
        response = await session.send(
            f"Use mcp__filesystem__list_directory to list {workspace}."
        )
        print(response.result)
    finally:
        mcp.cleanup()  # removes the temp config file

asyncio.run(main())
```

MCP tool names follow the pattern `mcp__<server>__<tool>`.

Full example: [`examples/mcp_integration.py`](../examples/mcp_integration.py)

### Writing your own tools in Python

Expose any Python function as a tool Claude can call. See [custom-tools.md](./custom-tools.md) for the full pattern and runnable example.

```python
from cckit import FastMCP

server = FastMCP("my-tools")

@server.tool(description="Look up a user by ID.")
def lookup_user(user_id: str) -> dict:
    return {"id": user_id, "name": "Alice"}

if __name__ == "__main__":
    server.run()
```

Register it with `MCPManager.add_python_server("my-tools", script="my_tools.py")`.

## Persistent ACP session

`ACPSession` keeps one `claude` subprocess alive over JSON-RPC 2.0. It was designed for permission prompts, file callbacks, and cancellation. **The real `claude` binary (v2.1.114) does not emit those callbacks** — `ACPSession` works as advertised only against an ACP-compatible mock or a third-party agent. See [concepts.md](./concepts.md#execution-paths).

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

`ACPSession` ignores the `bare` flag — it works with OAuth or API key as-is.

### ACP with permission callbacks

> **Note:** the code below only has effect against an ACP-compatible peer (the in-tree mock at `tests/fixtures/mock_acp_server.py`, or a third-party agent). The real `claude` binary does not emit `session/request_permission`, so the `gate` callback will never fire when this runs against `claude` itself. For real tool gating, use CLI-level `tools=`, `disallowed_tools=`, or `permission_mode=` instead.

To decide per-tool whether to approve (against an ACP-compatible peer), construct `DefaultHandlers` yourself and wire up a lower-level `ACPClient`.

```python
from cckit import ACPClient, PermissionPolicy, RpcTransport
from cckit.rpc import DefaultHandlers

def gate(params: dict) -> dict:
    tool = params.get("tool_name", "")
    return {"approved": tool in {"Read", "Grep"}}

handlers = DefaultHandlers(
    permission_policy=PermissionPolicy.CALLBACK,
    permission_callback=gate,
    workspace_root="/safe/workspace",  # sandbox for fs/read_text_file & fs/write_text_file
)

transport = RpcTransport([
    "/path/to/claude",
    "--input-format", "stream-json",
    "--output-format", "stream-json",
    "--verbose",
])
await transport.start()
client = ACPClient(transport, handlers=handlers)
await client.initialize(client_info={"name": "myapp", "version": "1.0"})
session_id = await client.new_session()
await client.prompt("…")
```

The callback can be sync or async. Returning `{"approved": True/False, "reason": "..."}` is the contract.

### Cancelling an in-flight prompt

```python
import asyncio

async with await ACPSession.create() as session:
    task = asyncio.create_task(session.send("Run a long task"))
    await asyncio.sleep(2)
    await session.cancel()
    try:
        await task
    except Exception:
        pass
```

## Saving and resuming sessions

`Session` exposes its `session_id` once the first turn returns. Save it, then re-attach later:

```python
session = await Session.create(cli, bare=False)
await session.send("First turn")
saved_id = session.session_id

# ... later, possibly in another process ...

resumed = Session.resume(cli, session_id=saved_id)
await resumed.send("Continuing where we left off")
```

`ACPSession.connect(session_id)` does the equivalent for ACP sessions.

## Collecting token usage

Every `Response` carries a `Usage` dataclass:

```python
response = await cli.execute("Hello", bare=False)
print(response.usage.input_tokens, response.usage.output_tokens)
print(response.usage.cache_read_tokens, response.usage.cache_write_tokens)
```

During streaming, watch for `UsageEvent`.
