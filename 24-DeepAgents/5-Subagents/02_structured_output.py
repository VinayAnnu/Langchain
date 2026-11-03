
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pathlib import Path
from langchain.agents.structured_output import ToolStrategy

from deepagents import create_deep_agent

from tools import internet_search


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Basic subagent structured output demo"

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR.parent / ".env")


# --------------------------------------------------
# 2. Define structured output schema
# --------------------------------------------------

class ResearchFindings(BaseModel):
    """
    Structured research findings returned by the
    research subagent.
    """

    summary: str = Field(
        description="A concise summary of the research."
    )

    key_findings: list[str] = Field(
        description="Important findings from the research."
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1."
    )

    sources: list[str] = Field(
        description="Source URLs supporting the findings."
    )


# --------------------------------------------------
# 3. Initialize main agent model
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
#  Initialize subagent model
# --------------------------------------------------

research_model = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)


# --------------------------------------------------
# 4. Define structured research subagent
# --------------------------------------------------

research_subagent = {

    "name": "structured-researcher",

    "description": (
        "Researches technical topics using internet search "
        "and returns validated structured findings."
    ),

    "system_prompt": """
    You are an expert technical researcher.

    Your responsibilities:

    1. Understand the research question.
    2. Use internet_search to collect information.
    3. Search multiple queries if necessary.
    4. Prefer authoritative and reliable sources.
    5. Identify the most important findings.
    6. Provide a concise summary.
    7. Include URLs of the sources you actually used.
    8. Assign a confidence score between 0 and 1.

    Never invent findings or source URLs.

    Return the result using the configured structured
    output schema.
    """,

    "tools": [internet_search],

    "model": research_model,

    # Explicit tool-based structured output strategy.
    "response_format": ToolStrategy(ResearchFindings),
}


# --------------------------------------------------
# 5. Register subagent
# --------------------------------------------------

subagents = [
    research_subagent
]


# --------------------------------------------------
# 6. Define supervisor prompt
# --------------------------------------------------

supervisor_prompt = """
You are a research supervisor.

You have access to a specialist called
structured-researcher.

Your responsibilities:

1. Understand the user's research request.
2. Delegate research to structured-researcher.
3. Receive its structured JSON findings.
4. Review the summary, key findings, confidence
   and source URLs.
5. Prepare a detailed, well-organized final report.

Do not invent findings or citations.
Clearly distinguish research findings from your own
additional explanations.
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
# 8. Invoke agent
# --------------------------------------------------

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Research recent advances in quantum "
                    "computing. Explain major breakthroughs, "
                    "current challenges, applications and "
                    "future research directions."
                ),
            }
        ]
    }
)


# --------------------------------------------------
# 9. Inspect conversation history
# --------------------------------------------------

print("\n========== CONVERSATION HISTORY ==========\n")

for message in result["messages"]:
    message.pretty_print()
