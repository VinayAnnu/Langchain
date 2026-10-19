"""
Conversational RAG with PDF uploads and chat history.

Stack:
- Streamlit
- LangChain
- Groq (GPT-OSS-20B)
- Hugging Face embeddings
- Chroma vector store
"""

# --------------------------------------------------
# 1. Configure SSL for the company Zscaler environment
# --------------------------------------------------
# Keep this before importing libraries that make HTTPS requests.
import truststore

truststore.inject_into_ssl()

# --------------------------------------------------
# 2. Imports
# --------------------------------------------------
import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

# --------------------------------------------------
# 3. Load environment variables
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

# Assumes .env is one directory above this Python file.
load_dotenv(PROJECT_DIR.parent / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# --------------------------------------------------
# 4. Initialize Streamlit session state
# --------------------------------------------------
def initialize_session_state():
    defaults = {
        "store": {},
        "vectorstore": None, ## After processing documents, you might assign a Chroma vector store:
        "vectorstore_signature": None, ## This variable is intended to identify which collection or version of documents was used to build the current vector store.
        "observed_upload_signature": None, ## This variable is intended to track the most recently observed state of the uploaded files.
    }

    for key, value in defaults.items():
        if key not in st.session_state: ## st.session_state stores data that should survive Streamlit reruns.
            st.session_state[key] = value


initialize_session_state()

# --------------------------------------------------
# 5. Initialize the embedding model
# --------------------------------------------------
@st.cache_resource
def get_embeddings():
    """Load the embedding model once per Streamlit server."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

# --------------------------------------------------
# 6. Create a signature for uploaded PDFs
# --------------------------------------------------
def calculate_upload_signature(uploaded_files):
    """
    Generate a hash based on file names and file contents.

    If the uploaded PDFs change, the signature changes.
    This helps detect when an existing vector store is stale.
    """
    if not uploaded_files:
        return None

    digest = hashlib.sha256() ##Create a SHA-256 hash object: converts input data into a fixed-length hash of 256 bits, or 64 hexadecimal characters.

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()

        digest.update(uploaded_file.name.encode("utf-8")) ## Add the filename to the hash
        digest.update(b"\0") ##b"\0" represents a single null byte. It acts as a separator between the filename and the file contents.
        digest.update(file_bytes) ## This adds the actual PDF bytes to the hash calculation.
        digest.update(b"\0") ## This separates one file's data from the next file's data when multiple PDFs are uploaded.

    return digest.hexdigest() ##hexdigest() returns the final hash as a hexadecimal string : 64 or 256 characters long. This string is a unique representation 
                                ##of the uploaded PDFs and their contents. it does not contain the actual PDF data, but it can be used to check if the uploaded
                                #  files have changed since the last time they were processed.


# --------------------------------------------------
# 7. Process PDFs and create the Chroma vector store
# --------------------------------------------------
def create_vectorstore(uploaded_files):
    """
    Load uploaded PDFs using unique temporary files,
    preserve source filenames, split the text, and
    create a Chroma vector store.
    """
    if not uploaded_files:
        raise ValueError("Please upload at least one PDF.")

    documents = []

    for uploaded_file in uploaded_files:
        temp_path = None

        try:
            file_bytes = uploaded_file.getvalue()

            # Use a unique temporary file for every PDF.
            with tempfile.NamedTemporaryFile(suffix=".pdf",delete=False,) as temp_file:
                temp_file.write(file_bytes)
                temp_path = Path(temp_file.name) ### ex: Path("/tmp/tmpa8x72k.pdf")  (Path object)

            # Extract PDF pages.
            loader = PyPDFLoader(str(temp_path))
            pdf_documents = loader.load()

            if not pdf_documents:
                raise ValueError(
                    f"No pages could be extracted from "
                    f"{uploaded_file.name}."
                )

            # Preserve the original uploaded filename.
            for document in pdf_documents:
                document.metadata["source"] = uploaded_file.name
                document.metadata["file_name"] = uploaded_file.name

            documents.extend(pdf_documents)

        except Exception as exc:
            raise RuntimeError(
                f"Could not process '{uploaded_file.name}': {exc}"
            ) from exc

        finally:
            # Clean up the temporary file.
            if temp_path is not None:
                temp_path.unlink(missing_ok=True) ##unlink() deletes the file at that path.missing_ok=True means Python does not raise a FileNotFoundError if the file has already been deleted.

    if not documents:
        raise ValueError(
            "No PDF pages were extracted from the uploaded files."
        )

    # Split pages into smaller chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    splits = text_splitter.split_documents(documents)

    if not splits:
        raise ValueError(
            "No text chunks were extracted. "
            "The PDFs may be scanned images requiring OCR."
        )

    # Embed the chunks and build the vector store.
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
    )

    return vectorstore, len(documents), len(splits)

# --------------------------------------------------
# 8. Get or create chat history for a session
# --------------------------------------------------
def get_session_history(session: str) -> BaseChatMessageHistory:
    """
    Return the chat history for the supplied session ID.
    Create a new history when the session does not exist.
    """
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()

    return st.session_state.store[session]


# --------------------------------------------------
# 9. Build the conversational RAG chain
# --------------------------------------------------
def build_conversational_rag_chain(llm, vectorstore):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    # Prompt used to rewrite follow-up questions.
    contextualize_q_system_prompt = """
    Given the conversation history and the latest user question,
    which might refer to something discussed earlier, formulate
    a standalone question that can be understood without the
    conversation history.

    Do not answer the question.
    If rewriting is unnecessary, return the question unchanged.
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    # Use the conversation history to improve retrieval.
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt,
    )

    # Prompt used to generate the final answer.
    system_prompt = """
    You are an assistant that answers questions about uploaded
    research papers and PDF documents.

    Use the retrieved context to answer the user's question.
    Use the conversation history to understand follow-up questions.

    Instructions:
    - Answer clearly and directly.
    - Use only information supported by the retrieved context.
    - Combine relevant information from multiple chunks when useful.
    - If the context is insufficient, explain what is missing.
    - Do not invent facts.

    Retrieved context:
    {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    # Combine retrieved documents and generate an answer.
    question_answer_chain = create_stuff_documents_chain( ##This chain takes the retrieved documents, combines their contents, and inserts them into the {context} variable in qa_prompt
        llm,
        qa_prompt,
    )

    # Connect retrieval with answer generation.
    rag_chain = create_retrieval_chain( ## The retrieval chain passes the retrieved documents to the answer chain under the context key.
        history_aware_retriever,
        question_answer_chain,
    )

    # Automatically load and update the conversation history.
    conversational_rag_chain = RunnableWithMessageHistory( ## Reads the session ID and loads existing history
        rag_chain,
        get_session_history, ## RunnableWithMessageHistory extracts the session ID from config and calls your history function.It's similar to giving LangChain a 
                                ##reference to a function and saying:When you need conversation history, call this function.
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain


# --------------------------------------------------
# 10. Streamlit user interface
# --------------------------------------------------
st.set_page_config(
    page_title="Conversational PDF RAG",
    page_icon="📚",
    layout="wide",
)

st.title("Conversational RAG with PDF Uploads")
st.write(
    "Upload research papers, ask questions, "
    "and ask follow-up questions using chat history."
)

# API key input.
api_key = st.sidebar.text_input(
    "Groq API key",
    value=GROQ_API_KEY,
    type="password",
    help="You can configure GROQ_API_KEY in your .env file.",
)

# Conversation identifier.
session_id = st.sidebar.text_input(
    "Session ID",
    value="default_session",
).strip()

if not session_id:
    session_id = "default_session"

st.sidebar.caption(
    "Each session ID maintains a separate conversation history "
    "within the current Streamlit session."
)


# --------------------------------------------------
# 11. Upload PDFs
# --------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_uploader",
)

current_signature = calculate_upload_signature(uploaded_files)

# --------------------------------------------------
# 12. Invalidate stale data when uploads change
# --------------------------------------------------
if (current_signature != st.session_state.observed_upload_signature):
    st.session_state.observed_upload_signature = current_signature

    # The old vector store no longer matches the uploaded files.
    st.session_state.vectorstore = None
    st.session_state.vectorstore_signature = None

    # Avoid carrying answers from old PDFs into a new document set.
    st.session_state.store = {}



# --------------------------------------------------
# 13. Process button
# --------------------------------------------------
if st.button("Process PDFs",type="primary",disabled=not uploaded_files,):
    try:
        with st.spinner(
            "Loading PDFs, creating embeddings, and building Chroma..."
        ):
            vectorstore, page_count, chunk_count = create_vectorstore(
                uploaded_files
            )

        # Store the index in session state to reuse across reruns.
        st.session_state.vectorstore = vectorstore
        st.session_state.vectorstore_signature = current_signature

        st.success(
            f"Processed {len(uploaded_files)} PDF(s), "
            f"{page_count} page(s), and "
            f"{chunk_count} chunk(s)."
        )

    except Exception as exc:
        st.session_state.vectorstore = None
        st.session_state.vectorstore_signature = None
        st.error(f"PDF processing failed: {exc}")


# Show indexing status.
if (
    st.session_state.vectorstore is not None
    and st.session_state.vectorstore_signature == current_signature
):
    st.success("Your uploaded PDFs are ready for questions.")
elif uploaded_files:
    st.info("Click 'Process PDFs' before asking questions.")
else:
    st.info("Upload one or more PDFs to get started.")


# --------------------------------------------------
# 14. Display the selected conversation history. display the previous conversation between the user and the AI assistant whenever the Streamlit application reruns.Think of this code as the chat history renderer for your application.
# --------------------------------------------------
session_history = get_session_history(session_id) ##get_session_history() retrieves the message history associated with that session.

for message in session_history.messages:
    if message.type == "human":  ## LangChain messages have types that identify their source.
        role = "user"
    elif message.type == "ai":
        role = "assistant" ## LangChain uses "human" and "ai" for these message types, while Streamlit's st.chat_message() uses "user" and "assistant" for the corresponding chat roles.
    else:
        # Skip unsupported message types in this simple UI.
        continue

    with st.chat_message(role):
        st.markdown(message.content) ## Using st.markdown() also allows Markdown formatting, such as headings, bold text, lists, and code blocks.


# --------------------------------------------------
# 15. Accept questions and run conversational RAG
# --------------------------------------------------
user_input = st.chat_input(
    "Ask a question about your PDFs..."
)

if user_input:
    # Display the new user message.
    with st.chat_message("user"):
        st.markdown(user_input)

    if not api_key:
        with st.chat_message("assistant"):
            st.warning(
                "Enter your Groq API key in the sidebar before chatting."
            )

    elif (
        st.session_state.vectorstore is None
        or st.session_state.vectorstore_signature != current_signature
    ):
        with st.chat_message("assistant"):
            st.warning(
                "Please upload your PDFs and click 'Process PDFs' "
                "before asking a question."
            )

    else:
        try:
            with st.chat_message("assistant"):
                with st.spinner(
                    "Searching documents and generating an answer..."
                ):
                    llm = ChatGroq(
                        model="openai/gpt-oss-20b",
                        api_key=api_key,
                    )

                    conversational_rag_chain = (
                        build_conversational_rag_chain(
                            llm,
                            st.session_state.vectorstore,
                        )
                    )

                    response = conversational_rag_chain.invoke(
                        {"input": user_input},
                        config={
                            "configurable": {
                                "session_id": session_id
                            }
                        },
                    )

                st.markdown(response["answer"])

                # Display sources used to formulate the answer.
                with st.expander("Retrieved document sources"):
                    for index, document in enumerate(response.get("context", []),start=1,):
                        source = document.metadata.get(
                            "source", "Unknown source"
                        )

                        page = document.metadata.get("page")

                        st.markdown(
                            f"**Source {index}: {source}**"
                        )

                        if page is not None:
                            # PyPDFLoader page numbers are typically
                            # zero-based, so add one for display.
                            st.caption(f"Page: {page + 1}")

                        st.write(document.page_content)
                        st.divider()

        except Exception as exc:
            st.error(f"Question answering failed: {exc}")
