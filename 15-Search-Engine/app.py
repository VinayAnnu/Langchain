# Keep this before importing libraries that make HTTPS requests.
import truststore
truststore.inject_into_ssl()

import os
import re
import uuid

import streamlit as st
from dotenv import load_dotenv
from ddgs import DDGS
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import (
    WikipediaQueryRun,
    DuckDuckGoSearchRun,
)

import arxiv
from langgraph.checkpoint.memory import InMemorySaver

# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="LangChain Search Assistant",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 LangChain Search Assistant")
st.caption(
    "Ask questions and let the AI choose between web search, "
    "academic papers, and Wikipedia."
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL ="openai/gpt-oss-20b"

if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY is missing. Add it to your .env file "
        "and restart the Streamlit application."
    )
    st.stop()

# --------------------------------------------------
# 2. Configure the arXiv client
# --------------------------------------------------

def create_arxiv_client() -> arxiv.Client:
    """Create an arXiv client using the HTTPS endpoint."""

    client = arxiv.Client()

    # Workaround verified in this environment:
    # the installed arxiv package defaults to HTTP.
    client.query_url_format = (
        "https://export.arxiv.org/api/query?{}"
    )

    return client


arxiv_client = create_arxiv_client()

# --------------------------------------------------
# 3. Define the arXiv tool
# --------------------------------------------------

@tool
def search_arxiv(query: str) -> str:
    """
    Search arXiv for academic papers by title, research topic,
    keywords, or arXiv paper ID. Return paper titles, authors,
    abstracts, and URLs.
    """

    query = query.strip()

    if not query:
        return "Please provide a paper title, topic, keywords, or arXiv ID."

    try:
        # Detect an arXiv identifier dynamically.
        # No specific paper ID is hardcoded.
        is_paper_id = bool(
            re.fullmatch(
                r"(?:\d{4}\.\d{4,5})(?:v\d+)?",
                query,
            )
        )

        if is_paper_id:
            search = arxiv.Search( ## Prepare the search configuration
                id_list=[query],
                max_results=1,
            )
        else:
            # Search across paper metadata and abstracts.
            safe_query = query.replace('"', "").strip()

            search = arxiv.Search(
                query=f'all:"{safe_query}"',
                max_results=3,
                sort_by=arxiv.SortCriterion.Relevance,
            )

        papers = list(arxiv_client.results(search)) ##. Execute the search and provide an iterator of results. Consume the iterator and collect its results into a Python list

        if not papers:
            return f"No arXiv papers found for: {query}"

        results = []

        for paper in papers:
            results.append(
                f"Title: {paper.title}\n"
                f"Authors: {', '.join(a.name for a in paper.authors)}\n"
                f"Published: {paper.published.date()}\n"
                f"Abstract: {paper.summary[:2500]}\n"
                f"URL: https://arxiv.org/abs/{paper.get_short_id()}"
            )

        return "\n\n---\n\n".join(results)

    except Exception:
        # Do not expose internal stack traces to the model or user.
        return (
            "The arXiv search failed. Please try again or use "
            "the web search tool to locate the paper."
        )

# --------------------------------------------------
# 4. Define the Wikipedia tool
# --------------------------------------------------
import wikipedia

# Set a descriptive User-Agent before the first Wikipedia request.
wikipedia.set_user_agent(
    "LangChainResearchAgent/1.0 (research project)"
)

wikipedia_wrapper = WikipediaAPIWrapper(
    top_k_results=2,
    doc_content_chars_max=3000,
)

wikipedia_tool = WikipediaQueryRun(
    name="wikipedia_search",
    description=(
        "Search Wikipedia for established background knowledge, "
        "definitions, people, places, scientific concepts, and "
        "general encyclopedia information."
    ),
    api_wrapper=wikipedia_wrapper,
)

# --------------------------------------------------
# 5. Define the web search tool
# --------------------------------------------------

@tool
def web_search_tool(query: str) -> str:
    """Search the public web using Bing for websites, current information,
    news, documentation, and topics requiring broader web research.
    
    Use this tool to find current information, recent documentation,
    news, and official technical references.

    Make queries specific. Prefer official documentation sources.
    One well-targeted search is usually sufficient. Do not repeat
    an equivalent query if useful results have already been returned."""

    try:
        with DDGS(timeout=20) as ddgs:
            results = ddgs.text(
                query,
                max_results=5,
                backend="bing",
            )

        if not results:
            return "No web search results were found."

        formatted_results = []

        for index, result in enumerate(results, start=1):
            formatted_results.append(
                f"{index}. {result.get('title', 'No title')}\n"
                f"URL: {result.get('href', '')}\n"
                f"Summary: {result.get('body', '')}"
            )

        return "\n\n".join(formatted_results)

    except Exception as exc:
        return f"Web search failed: {type(exc).__name__}: {exc}"

# --------------------------------------------------
# 6. Configure the LLM and modern agent
# --------------------------------------------------

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0,
)

tools = [
    search_arxiv,
    wikipedia_tool,
    web_search_tool,
]

SYSTEM_PROMPT = """
You are a helpful research assistant.

Choose tools according to the user's question:

- Use search_arxiv for academic papers, research topics,
  paper titles, and arXiv identifiers.
- Use wikipedia_search for established background knowledge
  and encyclopedia-style explanations.
- Use web_search for broader web research, documentation,
  recent information, and current events.

You may use more than one tool when the question requires it.
Do not claim that you searched a source unless you actually
used the corresponding tool.

Base your answer on the retrieved information when tools are used.
Explain the answer clearly, include relevant source URLs, and
be transparent when the available results are insufficient.

Use previous conversation messages to understand follow-up questions.
Do not reveal private internal reasoning.

Never call a tool repeatedly just to verify the same information.
"""

# Keep the checkpointer in Streamlit session state so it survives
# Streamlit script reruns for this session.
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = InMemorySaver()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I can search academic papers, Wikipedia, "
                "and the web. What would you like to explore?"
            ),
        }
    ]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=st.session_state.checkpointer,
)

# --------------------------------------------------
# 7. Display existing conversation
# --------------------------------------------------

with st.sidebar:
    st.header("About")
    st.write(f"**Model:** `{GROQ_MODEL}`")
    st.write("**Tools:**")
    st.write("- Academic paper search (arXiv)")
    st.write("- Wikipedia")
    st.write("- DuckDuckGo web search")

    if st.button("Clear conversation", use_container_width=True):
        # A new thread starts a fresh conversation.
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Conversation cleared. What would you like to search for?",
            }
        ]
        st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# 8. Handle a new question and stream agent activity
# --------------------------------------------------

if prompt := st.chat_input("Ask a question..."):

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        final_answer = None

        try:
            with st.status(
                "Working on your question...",
                expanded=True,
            ) as status:

                # The checkpointer restores earlier conversation
                # messages for this thread automatically.
                config = {
                    "configurable": {
                        "thread_id": st.session_state.thread_id
                    }
                }

                config["recursion_limit"] = 10

                for update in agent.stream(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config=config,
                    stream_mode="updates",
                ):
                    for node_name, node_update in update.items():
                        new_messages = node_update.get("messages", [])

                        for message in new_messages:
                            if isinstance(message, AIMessage):
                                # Show selected tools, not private reasoning.
                                for tool_call in message.tool_calls:
                                    tool_name = tool_call.get(
                                        "name", "unknown_tool"
                                    )
                                    st.write(
                                        f"🔧 Calling tool: `{tool_name}`"
                                    )

                                # The final assistant message has no
                                # outstanding tool calls.
                                if not message.tool_calls:
                                    content = message.content

                                    if isinstance(content, str):
                                        final_answer = content
                                    elif isinstance(content, list):
                                        text_parts = [
                                            item.get("text", "")
                                            for item in content
                                            if isinstance(item, dict)
                                            and item.get("type") == "text"
                                        ]
                                        final_answer = "\n".join(text_parts)

                            elif isinstance(message, ToolMessage):
                                st.write(
                                    f"✅ Tool completed: `{message.name}`"
                                )

                status.update(
                    label="Answer ready",
                    state="complete",
                    expanded=False,
                )

            if not final_answer:
                final_answer = (
                    "I couldn't generate a final answer. "
                    "Please try rephrasing your question."
                )

        except Exception as exc:
            # Display a useful error without exposing the API key.
            st.error(
                "The request failed. Check your Groq model configuration, "
                "network connection, and tool dependencies."
            )
            st.caption(f"Error type: {type(exc).__name__}")

            final_answer = (
                "Sorry, I couldn't complete that request. "
                "Please try again."
            )

        st.markdown(final_answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final_answer,
        }
    )

#TEST CASES
# 1. Find the paper Attention Is All You Need and explain its main contribution.
# Tests arXiv paper search and summarization.

# 2. What is machine learning?
# Tests general knowledge and Wikipedia tool selection.

# 3. Find the latest LangGraph documentation about create_agent.
# Tests web search and documentation retrieval.

# 4. Compare the Transformer paper with retrieval-augmented generation.
# Tests whether the agent can use multiple tools.

# 5. Who wrote that paper? Give me its arXiv link.
# Tests follow-up questions using conversation history.

# 6.what is the current weather in bengaluru india?
# Tests web search 