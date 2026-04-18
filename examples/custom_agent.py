"""Custom agent with a user-defined system prompt."""
import asyncio

from claude_agent import CustomAgent


async def main() -> None:
    agent = CustomAgent(
        name="HaikuPoet",
        system_prompt=(
            "You are a haiku poet. Every response must be a haiku "
            "(5-7-5 syllable structure). Nothing else."
        ),
        bare=False,
    )

    result = await agent.execute("Write a haiku about async Python programming.")
    print(result.result)


if __name__ == "__main__":
    asyncio.run(main())
