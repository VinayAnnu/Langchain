# The expected flow is:
# 1. Supervisor receives the request.
# 2. Supervisor selects research-agent.
# 3. Supervisor invokes the task tool.
# 4. Research agent uses internet_search.
# 5. Tavily returns search results.
# 6. Research agent synthesizes its findings.
# 7. Researcher's response returns to the supervisor.
# 8. Supervisor prepares the final response.

import os

from dotenv import load_dotenv
from pathlib import Path
from deepagents import create_deep_agent

from tools import internet_search

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Basic subagent demo"

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR.parent / ".env")
# --------------------------------------------------
# 1. Initialize main agent model
# --------------------------------------------------

from langchain_ollama import ChatOllama

main_model = ChatOllama(
    model="qwen3:8b",
    temperature=0,
    # model_kwargs={
    #     "parallel_tool_calls": True
    # }
)

# --------------------------------------------------
# 2. Initialize subagent model
# --------------------------------------------------

research_model = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)

# --------------------------------------------------
# 4. Define research subagent
# --------------------------------------------------

research_subagent = {
    "name": "research-agent",

    "description": (
        "Researches technical topics in depth using "
        "internet search and returns reliable findings "
        "with source URLs."
    ),

    "system_prompt": """
    You are an expert technical research assistant.

    Your responsibilities:

    1. Understand the research topic.
    2. Use internet_search to find relevant information.
    3. Search multiple queries when necessary.
    4. Prefer reliable and authoritative sources.
    5. Do not invent facts or source URLs.
    6. Summarize the findings clearly.
    7. Include source URLs in your final response.

    Return a well-organized research report.
    """,

    "tools": [internet_search],

    # Optional model override.
    # If omitted, the subagent inherits the main model.
    "model": research_model,
}

# --------------------------------------------------
# 5. Register subagents
# --------------------------------------------------

subagents = [
    research_subagent
]

# --------------------------------------------------
# 6. Define supervisor instructions
# --------------------------------------------------

supervisor_prompt = """
You are a research supervisor.

You have access to a specialized subagent called
research-agent.

Your responsibilities:

1. Understand the user's research question.
2. Delegate detailed internet research to research-agent.
3. Wait for the subagent's response.
4. Review and synthesize its findings.
5. Present a comprehensive, well-structured answer.
6. Preserve source URLs in your final response.

Do not pretend that research was performed
if you did not actually invoke the subagent.
"""

# --------------------------------------------------
# 7. Create Deep Agent
# --------------------------------------------------

agent = create_deep_agent(
    model=main_model,
    subagents=subagents,
    system_prompt=supervisor_prompt,
)

# --------------------------------------------------
# 8. Invoke the main agent
# --------------------------------------------------

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Research LLM Gateways in depth. "
                ),
            }
        ]
    }
)

# --------------------------------------------------
# 9. Print conversation history
# --------------------------------------------------

print("\n========== CONVERSATION HISTORY ==========\n")

for message in result["messages"]:
    message.pretty_print()
