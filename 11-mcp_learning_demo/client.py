"""LLM-powered MCP client that connects to two local MCP servers."""
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR.parent / ".env")


async def main() -> None:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is missing. Copy .env.example to .env and add your key."
        )

    # The client launches each server as a separate process over stdio.this configuration tells the MCP adapter how to launch each server. It does not itself define addition, 
    # multiplication, or weather functionality. Those tools are defined inside the server files.
    mcp_client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio",
                "command": sys.executable, ## sys.executable returns the path to the Python interpreter running your current client.For example, it might point to:/Users/your-user/Documents/Langchain/cenv/bin/python
                ##This is useful because it helps launch the MCP servers using the same Python environment as the client, where you installed the required packages.Instead of hardcoding a path such as python3, the program uses the current interpreter.
                "args": [str(PROJECT_DIR / "servers" / "math_server.py")], ##The args list contains the arguments passed to the command.
                ##Conceptually, the process launched is equivalent to:/path/to/current/python /path/to/project/servers/math_server.py
            },
            "weather": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(PROJECT_DIR / "servers" / "weather_server.py")],
            },
        }
    )

    # Ask both MCP servers which tools they expose.
    tools = await mcp_client.get_tools()
    print("Discovered MCP tools:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    model = ChatGroq(model="openai/gpt-oss-20b")

    ##create_agent() creates a LangChain agent that can use the specified model and tools. In current LangChain, the agent is built on LangGraph machinery, which manages the tool-calling workflow.
    agent = create_agent( 
        model=model,
        tools=tools,
        system_prompt=(
            "You are a helpful assistant. Use the available MCP tools whenever "
            "the user's request requires addition, multiplication, or current weather. "
            "For arithmetic, pass the numbers to the appropriate tool rather than "
            "calculating mentally. For weather, call the weather tool and report the "
            "location and observation time when available. If the request is ambiguous "
            "or required inputs are missing, ask a clarifying question. Do not claim "
            "to have current weather unless the weather tool returned data."
        ),
    )

    print("\nMCP demo ready. Ask a question, or type 'quit' to exit.")
    print("Examples: 'Add 19 and 23', 'Multiply 12 by 8', 'What's the weather in Bengaluru?'\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user_input}]}
            )
            print("\n" + "=" * 70)
            print(f"\n Response received from agent: \n")
            print("=" * 70)
            print(f"{result}\n")

            print(f"\nAssistant: {result['messages'][-1].content}\n")
        except Exception as exc:
            print(f"\nRequest failed: {exc}\n")


if __name__ == "__main__":
    asyncio.run(main())

## MCP standardizes how tools are exposed and called; LangChain connects those tools to the LLM through an agent; 
# and the LLM decides which available capability is appropriate for the user's request.