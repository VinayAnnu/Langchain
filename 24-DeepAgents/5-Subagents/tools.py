import truststore
truststore.inject_into_ssl()

import os
from typing import Literal

from dotenv import load_dotenv
from tavily import TavilyClient
from langchain.tools import tool
from pathlib import Path

# Resolve the project directory
PROJECT_DIR = Path(__file__).resolve().parent

# Load .env from the parent directory
load_dotenv(PROJECT_DIR.parent.parent / ".env")

# Validate Tavily API key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError(
        "TAVILY_API_KEY is missing. "
        "Please configure it in your .env file."
    )

# --------------------------------------------------
# 2. Initialize Tavily client
# --------------------------------------------------

tavily_client = TavilyClient(
    api_key=TAVILY_API_KEY
)

# --------------------------------------------------
# 3. Define Tavily search tool
# --------------------------------------------------

@tool ##This converts your ordinary Python function into a LangChain tool.With it, LangChain can expose the function's name, description, and parameter schema to the LLM.
def internet_search(
    query: str,
    max_results: int = 1,
    topic: Literal["general", "news", "finance"] = "general", ##This restricts the accepted values to three choices.
    include_raw_content: bool = False,
) -> str:
    """
    Search the internet using Tavily.

    Args:
        query: The search query.
        max_results: Maximum number of results to return.
        topic: Search category: general, news, or finance.
        include_raw_content: Whether to include extracted page content.

    Returns:
        Search results containing titles, URLs and content.
    """

    if not query.strip():
        return "Search query cannot be empty."

    response = tavily_client.search(
        query=query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content,
    )

    search_results = response.get("results", [])

    if not search_results:
        return f"No results found for query: {query}"

    formatted_results = []

    for result in search_results:

        formatted_results.append(
            f"""
                Title: {result.get('title', 'N/A')}

                URL: {result.get('url', 'N/A')}

            Content: {result.get('content', 'N/A')[:1200]}
            """
        )

    return "\n".join(formatted_results)
