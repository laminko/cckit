"""Multi-turn conversation with context preserved between turns."""
import asyncio

from claude_agent import ClaudeCLI, Session


async def main() -> None:
    cli = ClaudeCLI()
    session = await Session.create(cli, bare=False)

    print("=== Multi-turn conversation ===\n")

    r1 = await session.send("My favourite colour is blue. Remember this.")
    print(f"Turn 1: {r1.result}\n")

    r2 = await session.send("What is my favourite colour?")
    print(f"Turn 2: {r2.result}\n")

    print(f"Session ID: {session.session_id}")
    print(f"History length: {len(session.get_history())} messages")


if __name__ == "__main__":
    asyncio.run(main())
