# Getting started

A ten-minute tour: install, authenticate, run your first prompt.

## 1. Install

```bash
git clone https://github.com/laminko/cckit.git
cd cckit
uv sync
```

Requirements:

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- The `claude` CLI at `~/.local/bin/claude` (the default; override via `CLI(binary_path=...)`)

## 2. Authenticate

`cckit` does not handle auth itself — it shells out to the `claude` binary, which needs credentials. Pick one:

**OAuth** (recommended for personal use):

```bash
claude login
```

Because OAuth tokens live in the system keychain, you must pass `bare=False` so the spawned `claude` process can read them:

```python
await cli.execute("hi", bare=False)
await Session.create(cli, bare=False)
CodeAgent(bare=False)
```

**API key** (recommended for scripts / CI):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

With an API key set, the `bare=True` default works unchanged.

> `ACPSession` ignores the `bare` flag entirely — it works with either auth method out of the box.

For the full reasoning behind `bare`, see [concepts.md → the `bare` flag](./concepts.md#the-bare-flag).

## 3. Your first prompt

```python
import asyncio
from cckit import CLI

async def main():
    cli = CLI()
    response = await cli.execute("What is 2 + 2?", bare=False)
    print(response.result)

asyncio.run(main())
```

Run it:

```bash
uv run python your_script.py
```

If you see a `CLIError` about authentication, revisit step 2 — most first-run failures are `bare=True` + OAuth, or a missing `ANTHROPIC_API_KEY`.

## 4. What to read next

| If you want to… | Read |
|---|---|
| Understand when to use `CLI` vs `Session` vs `ACPSession` | [concepts.md](./concepts.md) |
| See working code for streaming, multi-turn, MCP, ACP | [guides.md](./guides.md) |
| Write a custom tool Claude can call | [custom-tools.md](./custom-tools.md) |
| Look up a specific class or method | [api-reference.md](./api-reference.md) |
| Browse runnable examples | [`examples/`](../examples/) |
