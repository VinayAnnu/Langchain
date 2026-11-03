
import truststore
truststore.inject_into_ssl()
import os

from langchain_tavily import TavilySearch
from langchain.tools import tool
from dotenv import load_dotenv
from pathlib import Path

# Resolve the project directory
PROJECT_DIR = Path(__file__).resolve().parent

# Load .env from the parent directory
load_dotenv(PROJECT_DIR.parent.parent / ".env")

# Validate Tavily API key
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY is missing from .env")

# --------------------------------------------------
# Initialize Tavily Search tool
# --------------------------------------------------

tavily_search = TavilySearch(
    max_results=1,
    topic="general",
    tavily_api_key=tavily_api_key
)



# --------------------------------------------------
# Create custom LangChain tool
# --------------------------------------------------

@tool("tavily_web_search")
def tavily_web_search(query: str) -> str:
    """
    Search the internet for reliable information
    about a given topic using Tavily.
    """

    # Execute Tavily search
    response = tavily_search.invoke(
        {"query": query}
    )

    # Handle dictionary response
    if isinstance(response, dict):
        search_results = response.get("results", [])

    # Handle list response
    elif isinstance(response, list):
        search_results = response

    # Handle string or other response formats
    else:
        return str(response)[:1500]

    if not search_results:
        return f"No search results found for: {query}"

    results = []

    for item in search_results:

        results.append(
            f"""
            Title: {item.get('title', 'N/A')}
            URL: {item.get('url', 'N/A')}
            Content: {str(item.get('content', 'N/A'))[:700]}
            """
        )

    # Limit total output to 1500 characters
    return "\n".join(results)[:1500]
