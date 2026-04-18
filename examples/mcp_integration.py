"""MCP server integration example.

Spins up the filesystem MCP server rooted at the current working directory
and asks Claude to list its contents using that server.
"""
import asyncio
from pathlib import Path

from claude_agent import ClaudeCLI, MCPManager, Session


async def main() -> None:
    workspace = str(Path.cwd())

    mcp = MCPManager()
    mcp.add_server(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", workspace],
    )

    config_path = mcp.write_config_file()
    print(f"MCP config written to: {config_path}")

    try:
        cli = ClaudeCLI()
        session = await Session.create(
            cli,
            bare=False,
            mcp_config_path=config_path,
            tools=["mcp__filesystem__list_directory"],
        )

        response = await session.send(
            f"Use the mcp__filesystem__list_directory tool to list the files in {workspace}. "
            "Reply with a short summary of what you see."
        )
        print(response.result)
    finally:
        mcp.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
