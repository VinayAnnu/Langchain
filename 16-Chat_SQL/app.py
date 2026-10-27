# Keep this before importing libraries that make HTTPS requests.
import truststore
truststore.inject_into_ssl()

import logging
import os
import re
import sqlite3
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq


# ---------------------------------------------------------
# 1. Application configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="SQL Chatbot",
    page_icon="🦜",
    layout="centered",
)

st.title("🦜 Chat with your SQL Database")
st.caption("Ask questions about your student records in plain English.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Keep the model configurable instead of hardcoding it
# throughout the application.
MODEL_NAME = "openai/gpt-oss-20b"

MAX_ROWS = 100
MAX_ITERATIONS = 6

# Resolve student.db relative to this Python file.
DB_PATH = (Path(__file__).parent / "student.db").resolve()

# ---------------------------------------------------------
# 2. Read the API key securely
# ---------------------------------------------------------

# Prefer an environment variable. The sidebar input is
# a convenient alternative for local learning.
load_dotenv()
api_key = os.getenv("GROQ_API_KEY", "").strip()

if not api_key:
    api_key = st.sidebar.text_input(
        "Groq API key",
        type="password",
    ).strip()

if not api_key:
    st.info(
        "Set GROQ_API_KEY or enter your Groq API key "
        "in the sidebar."
    )
    st.stop()


@st.cache_resource
def get_checkpointer():
    """Create one reusable checkpointer for this app process."""
    return InMemorySaver()

# ---------------------------------------------------------
# 3. Create a read-only SQLite engine
# ---------------------------------------------------------

@st.cache_resource
def get_database_engine(db_path: str) -> Engine:
    """Create a reusable, read-only SQLite engine."""

    # SQLite URI mode=ro requires the database to exist
    # and prevents this connection from writing to it.
    def open_read_only_connection():
        return sqlite3.connect(
            f"file:{db_path}?mode=ro",
            uri=True,
            timeout=10,
            check_same_thread=False,
        )

    # SQLAlchemy obtains connections through our factory.
    return create_engine(
        "sqlite://",
        creator=open_read_only_connection,
        pool_pre_ping=True,
    )


# ---------------------------------------------------------
# 4. Build restricted tools for the agent
# ---------------------------------------------------------

def build_agent(engine: Engine, groq_api_key: str, checkpointer : BaseCheckpointSaver):
    """Build an LLM agent with limited database tools."""

    # Discover the real table names in the database.
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    # Only expose explicitly approved tables to the agent.
    allowed_tables = {"STUDENT"} & existing_tables

    if not allowed_tables:
        raise ValueError(
            "The STUDENT table was not found in student.db."
        )

    # Capture only approved table names in the tools.
    @tool
    def get_database_schema() -> str:
        """Return the schema of the approved database tables."""

        schema_parts = []

        for table_name in sorted(allowed_tables):
            columns = inspector.get_columns(table_name)

            column_descriptions = [
                f"{column['name']} ({column['type']})"
                for column in columns
            ]

            schema_parts.append(
                f"Table: {table_name}\n"
                f"Columns: {', '.join(column_descriptions)}"
            )

        return "\n\n".join(schema_parts)

    @tool
    def run_readonly_sql(query: str) -> str:
        """Execute one read-only SELECT query on approved tables.

        Use this tool to retrieve data. Only SELECT queries
        are permitted. Return at most 100 rows.
        """

        # Remove whitespace and trailing semicolons.
        query = query.strip().rstrip(";").strip()

        if not query:
            return "Error: SQL query cannot be empty."

        # Accept SELECT or WITH queries only.
        # This rejects INSERT, UPDATE, DELETE, DROP, etc.
        if not re.match(
            r"^(SELECT|WITH)\b",
            query,
            flags=re.IGNORECASE,
        ):
            return "Error: Only SELECT queries are permitted."

        # Reject multiple SQL statements.
        if ";" in query:
            return "Error: Only one SQL statement is allowed."

        # Ensure the query references only approved tables.
        # This is a basic guardrail, not a complete SQL parser.
        referenced_tables = re.findall(
            r"\b(?:FROM|JOIN)\s+"
            r"(?:[\"`\[]?)([A-Za-z_][A-Za-z0-9_]*)",
            query,
            flags=re.IGNORECASE,
        )

        if not referenced_tables:
            return (
                "Error: Query must select from an approved table."
            )

        if any(
            table.upper() not in allowed_tables
            for table in referenced_tables
        ):
            return "Error: Query references a disallowed table."

        # Execute using the read-only database connection.
        try:
            with engine.connect() as connection:
                result = connection.execute(text(query))

                # Fetch at most MAX_ROWS into application memory.
                rows = result.fetchmany(MAX_ROWS + 1)
                truncated = len(rows) > MAX_ROWS
                rows = rows[:MAX_ROWS]

                # Convert results to dictionaries for readability.
                records = [
                    dict(row._mapping)
                    for row in rows
                ]

            if not records:
                return "The query returned no rows."

            output = repr(records)

            if truncated:
                output += (
                    f"\nResults truncated to {MAX_ROWS} rows."
                )

            return output

        except Exception:
            # Log technical details on the server, not in the UI.
            logger.exception("Read-only SQL query failed.")
            return (
                "The SQL query failed. Check the table schema "
                "and try a corrected SELECT query."
            )

    # Use a model that supports tool calling.
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model=MODEL_NAME,
        temperature=0,
    )

    # Provide explicit instructions to the LLM.
    system_prompt = f"""
You are a helpful SQL database assistant.

Rules:
1. Answer questions using the database tools.
2. Inspect the database schema before querying if necessary.
3. Only query the approved STUDENT table.
4. Use SELECT queries only.
5. Never invent query results.
6. Keep queries focused and avoid unnecessary columns.
7. Limit queries to {MAX_ROWS} rows when appropriate.
8. Explain the results clearly and concisely.
9. If a query fails, correct it when possible.
10. Do not claim a query succeeded unless the tool returned results.
"""

    # Create a modern tool-calling agent.
    return create_agent(
        model=llm,
        tools=[
            get_database_schema,
            run_readonly_sql,
        ],
        system_prompt=system_prompt,
        checkpointer=checkpointer,  # Enable agent state persistence
    )


# ---------------------------------------------------------
# 5. Initialize the database and agent
# ---------------------------------------------------------

try:
    if not DB_PATH.is_file():
        st.error(f"Database file not found: {DB_PATH.name}")
        st.stop()

    engine = get_database_engine(str(DB_PATH))
    checkpointer = get_checkpointer()
    # Check connectivity and confirm the database is readable.
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    agent = build_agent(engine, api_key, checkpointer)

except Exception:
    logger.exception("Failed to initialize SQL chatbot.")
    st.error(
        "The chatbot could not be initialized. "
        "Check the database file, API key, model name, "
        "and server logs."
    )
    st.stop()


# ---------------------------------------------------------
# 6. Initialize and display chat history
# ---------------------------------------------------------
# Each browser session gets its own conversation thread.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! Ask me anything about your student records."
            ),
        }
    ]

if st.sidebar.button("Clear chat history"):
    st.session_state.messages = []

    # A new thread starts with no previous conversation.
    st.session_state.thread_id = str(uuid.uuid4())
    st.rerun()

# Redisplay previous messages after Streamlit reruns.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------
# 7. Accept a question and invoke the agent
# ---------------------------------------------------------

user_query = st.chat_input(
    "Example: Which student scored the highest marks?"
)

if user_query and user_query.strip():

    # Save the user message.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):

        try:
            # Show progress without a callback handler.
            with st.spinner("Analyzing your question..."):

                # The create_agent API accepts a list of messages.
                result = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": user_query,
                            }
                        ]
                    },
                    config={
                        "configurable": {
                            "thread_id": st.session_state.thread_id,
                        },
                        "recursion_limit": MAX_ITERATIONS * 2,
                    },
                )

                # The final AI message contains the answer.
                final_message = result["messages"][-1]
                answer = final_message.content

                # Handle text-block responses as well as plain text.
                if isinstance(answer, list):
                    answer = "\n".join(
                        block["text"]
                        for block in answer
                        if isinstance(block, dict)
                        and block.get("type") == "text"
                    )

                if not isinstance(answer, str) or not answer.strip():
                    answer = "The agent returned no text answer."

        except Exception:
            logger.exception("Agent invocation failed.")
            answer = (
                "Sorry, I couldn't answer that question. "
                "Please try again or check the application logs."
            )

        # Display and save the final answer.
        st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )
