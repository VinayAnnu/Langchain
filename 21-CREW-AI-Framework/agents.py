
from crewai import Agent, LLM

from tools import youtube_search_tool
import os
# Load environment variables from .env.
from dotenv import load_dotenv
load_dotenv()

# Configure the language model explicitly.
llm = LLM(
    model="ollama/qwen2.5:3b",
    base_url="http://localhost:11434",
    temperature=0.2,
)


blog_researcher = Agent(
    role="YouTube Technical Researcher",
    goal=(
        "Find accurate, relevant information about {topic} "
        "from the specified YouTube channel."
    ),
    backstory=( ## Gives the LLM behavioral context for how to approach the job.
        "You are an experienced AI and data science researcher. "
        "You identify relevant videos, extract useful concepts, "
        "and organize findings into a factual research report. "
        "You distinguish retrieved information from assumptions."
    ),
    llm=llm,
    tools=[youtube_search_tool],
    verbose=True,
    allow_delegation=False,
    max_iter=5,
)


blog_writer = Agent(
    role="Technical Blog Writer",
    goal=(
        "Create a clear, engaging, technically accurate blog "
        "about {topic} using the research provided."
    ),
    backstory=(
        "You are an experienced technical writer who explains "
        "complex AI, machine learning, and data science concepts "
        "in simple language. You structure articles with headings, "
        "examples, comparisons, and practical takeaways."
    ),
    llm=llm,
    tools=[],
    verbose=True,
    allow_delegation=False,
    max_iter=3,
)
