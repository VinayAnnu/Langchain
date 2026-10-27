# --------------------------------------------------
# 0. Configure SSL for the company Zscaler environment
# --------------------------------------------------
# Keep this before importing libraries that make HTTPS requests.
import truststore

truststore.inject_into_ssl()

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.tools import (
    WikipediaQueryRun,
)
from langchain_community.utilities import (
    WikipediaAPIWrapper,
)
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS

from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_classic.agents import (
    AgentExecutor,
    create_openai_tools_agent,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
import arxiv
from langchain_core.tools import tool

# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from your .env file.")


# Avoid a missing User-Agent warning from web loaders.
os.environ.setdefault(
    "USER_AGENT",
    "LangChainResearchAgent/1.0"
)

# --------------------------------------------------
# 2. Configure Wikipedia tool
# --------------------------------------------------
import wikipedia

# Set a descriptive User-Agent before the first Wikipedia request.
wikipedia.set_user_agent(
    "LangChainResearchAgent/1.0 (research project)"
)

wiki_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=1000,
)

wiki_tool = WikipediaQueryRun(
    api_wrapper=wiki_wrapper,
)

# Give the tool a clear purpose for the LLM.
wiki_tool.description = (
    "Search Wikipedia for general factual information, "
    "definitions, people, places, and scientific concepts."
)

# --------------------------------------------------
# 3. Configure arXiv research tool
# --------------------------------------------------

arxiv_client = arxiv.Client()
arxiv_client.query_url_format = (
    "https://export.arxiv.org/api/query?{}"
)

@tool
def arxiv_tool(query: str) -> str:
    """Search arXiv for academic papers by title, topic, or paper ID."""

    query = query.strip()

    if not query:
        return "Please provide a paper title, topic, or arXiv ID."

    # Let the arXiv API search dynamically.
    search = arxiv.Search(
        query=f'all:"{query}"',
        max_results=3,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = list(arxiv_client.results(search))

    if not papers:
        return "No matching papers found."

    return "\n\n".join(
        f"Title: {paper.title}\n"
        f"Authors: {', '.join(a.name for a in paper.authors)}\n"
        f"Abstract: {paper.summary}\n"
        f"URL: https://arxiv.org/abs/{paper.get_short_id()}"
        for paper in papers
    )
# --------------------------------------------------
# 4. Configure Ollama embeddings
# --------------------------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# --------------------------------------------------
# 5. Build or load the LangSmith FAISS index
# --------------------------------------------------

LANGSMITH_URL = "https://docs.smith.langchain.com/"

FAISS_INDEX_DIR = Path("faiss_langsmith_index")

# Set True to reload the website and rebuild the index.
REBUILD_INDEX = False


def create_or_load_vectorstore():
    """
    Load an existing local FAISS index when available.
    Otherwise, load LangSmith documentation, split it,
    create embeddings, and save a new index.
    """

    index_file = FAISS_INDEX_DIR / "index.faiss"
    metadata_file = FAISS_INDEX_DIR / "index.pkl"

    index_exists = (
        index_file.exists()
        and metadata_file.exists()
    )

    if index_exists and not REBUILD_INDEX:
        print("Loading existing FAISS index...")

        # Only load an index that you created and trust.
        vectorstore = FAISS.load_local(
            str(FAISS_INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

        print("FAISS index loaded successfully.")
        return vectorstore

    print("Loading LangSmith documentation...")

    loader = WebBaseLoader(LANGSMITH_URL)
    docs = loader.load()

    if not docs:
        raise RuntimeError(
            "No documents were loaded from LangSmith."
        )

    print(f"Loaded {len(docs)} document(s).")

    # Split the documents into overlapping chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    documents = text_splitter.split_documents(docs)

    if not documents:
        raise RuntimeError(
            "Document splitting produced no chunks."
        )

    print(f"Created {len(documents)} chunks.")

    # Generate embeddings and build the vector store.
    vectorstore = FAISS.from_documents(
        documents,
        embeddings,
    )

    # Persist the index for future executions.
    vectorstore.save_local(
        str(FAISS_INDEX_DIR)
    )

    print("FAISS index created and saved.")

    return vectorstore


vectorstore = create_or_load_vectorstore()

# --------------------------------------------------
# 6. Create the LangSmith retriever tool
# --------------------------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3},
)

langsmith_tool = create_retriever_tool(
    retriever,
    "langsmith-search",
    (
        "Search the indexed LangSmith documentation for "
        "information about LangSmith setup, tracing, debugging, "
        "evaluation, monitoring, and LLM application development. "
        "Use this tool for questions specifically about LangSmith."
    ),
)

# --------------------------------------------------
# 7. Register all tools
# --------------------------------------------------

tools = [
    wiki_tool,
    arxiv_tool,
    langsmith_tool,
]

print("\nAvailable tools:")

for tool in tools:
    print(f"- {tool.name}: {tool.description}")

# --------------------------------------------------
# 8. Configure the Groq LLM
# --------------------------------------------------

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="openai/gpt-oss-20b",
    temperature=0,
)

# --------------------------------------------------
# 9. Create the agent prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful research assistant.

            You have access to three tools:

            1. Wikipedia:
               Use for general factual information.

            2. arXiv:
               Use for academic papers and scientific research.

            3. langsmith-search:
               Use for questions about LangSmith and its
               documentation.

            Instructions:
            - Select the most appropriate tool for the question.
            - Use retrieved tool results to support your answer.
            - Do not claim that you searched a source if you did not.
            - If the available information is insufficient, say so.
            - Provide a clear, concise, and accurate final answer.
            """
        ),
        ("human", "{input}"),
        MessagesPlaceholder(
            variable_name="agent_scratchpad" ## agent_scratchpad : Holds intermediate agent messages, such as tool calls and tool results.
        ),
    ]
)

# --------------------------------------------------
# 10. Create the tool-calling agent
# --------------------------------------------------

agent = create_openai_tools_agent(
    llm=llm,
    tools=tools, ## The agent needs the tool definitions so it can tell the LLM which tools are available, what each tool does, and what arguments each tool accepts.agent uses tool definitions to decide which tool to call and with what input.
    prompt=prompt,
)

# --------------------------------------------------
# 11. Create the AgentExecutor
# --------------------------------------------------

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools, ## AgentExecutor uses the tools to actually execute the selected tool and return its result to the agent.
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,
)

# --------------------------------------------------
# 12. Ask questions
# --------------------------------------------------

def ask_research_agent(question: str) -> str:
    """
    Send a question to the agent and return its final answer.
    """

    question = question.strip()

    if not question:
        raise ValueError("Please enter a question.")

    response = agent_executor.invoke(
        {"input": question}
    )

    return response["output"]


if __name__ == "__main__":
    print("\nResearch Agent is ready!")
    print("Type 'exit' to stop.\n")

    while True:
        user_question = input("You: ").strip()

        if user_question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            answer = ask_research_agent(user_question)
            print("\nAssistant:", answer, "\n")

        except Exception as exc:
            print(
                f"\nError while processing your question: {exc}\n"
            )

