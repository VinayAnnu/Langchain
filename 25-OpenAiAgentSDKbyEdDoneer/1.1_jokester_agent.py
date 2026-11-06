
import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent
from agents.extensions.models.litellm_model import LitellmModel
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


from agents import (
    Agent,
    Runner,
    trace,
    function_tool,
    SQLiteSession
)


# --------------------------------------------------
# 1. Load environment variables and create groq model
# --------------------------------------------------

load_dotenv(override=True)

#  Define the Groq model
groq_model = LitellmModel(
    model="groq/openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)

# --------------------------------------------------
# 2. Create a custom function tool
# --------------------------------------------------

@function_tool
def get_current_year(timezone: str) -> str:
    """
    Return the current calendar year for a timezone.

    Args:
        timezone: Valid timezone, such as Asia/Kolkata or UTC.
    """

    try:
        current_time = datetime.now(ZoneInfo(timezone))
        return str(current_time.year)

    except ZoneInfoNotFoundError:
        return f"Invalid timezone: {timezone}"



# --------------------------------------------------
# 3. Define the agent
# --------------------------------------------------

agent = Agent(
    name="Jokester",

    instructions=(
        "You are a funny and creative joke teller. "
        "Tell concise, original jokes. "
        "When the user asks for a joke involving the current year, "
        "call get_current_year using Asia/Kolkata timezone."
    ),

    model=groq_model,

    tools=[get_current_year]
)


# --------------------------------------------------
# 4. Create a persistent conversation session
# --------------------------------------------------

session = SQLiteSession(
    "jokester_user_001",
    "jokester.db"
)


# --------------------------------------------------
# 5. Normal execution with tracing
# --------------------------------------------------

async def run_agent():

    with trace(
        workflow_name="Jokester Agent Workflow"
    ):

        result = await Runner.run(
            agent,
            "Tell me a joke about Autonomous AI Agents.",
            session=session
        )

        print("\nAgent Response:")
        print(result.final_output)

        print("\nLast Agent:")
        print(result.last_agent.name)


# --------------------------------------------------
# 6. Streaming execution
# --------------------------------------------------

async def stream_agent():

    result = Runner.run_streamed(
        agent,
        input="Tell me five jokes about AI Agents." ##You can also pass session=session to Runner.run_streamed() when you want streamed conversational turns to use persistent history.
    )

    print("\nStreaming Response:\n")

    async for event in result.stream_events():

        if (
            event.type == "raw_response_event"
            and isinstance(
                event.data,
                ResponseTextDeltaEvent
            )
        ):

            print(
                event.data.delta,
                end="",
                flush=True
            )

    print("\n")


# --------------------------------------------------
# 7. Application entry point
# --------------------------------------------------

async def main():

    print("1. Normal Agent Execution")

    await run_agent()

    print("\n2. Streaming Agent Execution")

    await stream_agent()


if __name__ == "__main__":

    asyncio.run(main())
