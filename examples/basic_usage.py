"""Basic one-shot usage of ClaudeCLI."""
import asyncio

from claude_agent import ClaudeCLI


async def main() -> None:
    cli = ClaudeCLI()

    response = await cli.execute(
        "What is 2 + 2? Answer in one line.",
        bare=False,
    )

    print("Result:", response.result)
    print("Session ID:", response.session_id)
    print("Duration:", response.duration_ms, "ms")


if __name__ == "__main__":
    asyncio.run(main())
