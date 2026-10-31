
import truststore
truststore.inject_into_ssl()

import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import (
    ChatNVIDIA,
    NVIDIAEmbeddings,
)
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

CHAT_MODEL = os.getenv(
    "NVIDIA_CHAT_MODEL",
    "openai/gpt-oss-20b",
)

EMBEDDING_MODEL = os.getenv(
    "NVIDIA_EMBEDDING_MODEL",
    "nvidia/nemotron-3.5-embed-1b",
)

PDF_DIRECTORY = Path("./us_census")

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
RETRIEVAL_K = 3


# ============================================================
# 2. STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="NVIDIA RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 NVIDIA RAG Assistant")
st.caption(
    "Ask questions about your PDFs using NVIDIA-hosted models."
)

if not NVIDIA_API_KEY:
    st.error(
        "NVIDIA_API_KEY is missing. Add it to your .env file."
    )
    st.stop()


# ============================================================
# 3. INITIALIZE MODELS
# ============================================================

@st.cache_resource
def get_llm(api_key, model_name):
    return ChatNVIDIA(
        model=model_name,
        nvidia_api_key=api_key,
        temperature=0.2,
        max_completion_tokens=1024,
        timeout=120,
    )


@st.cache_resource
def get_embeddings(api_key, model_name):
    return NVIDIAEmbeddings(
        model=model_name,
        nvidia_api_key=api_key,
    )


llm = get_llm(NVIDIA_API_KEY, CHAT_MODEL)

embeddings = get_embeddings(
    NVIDIA_API_KEY,
    EMBEDDING_MODEL,
)


# ============================================================
# 4. RAG PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful document question-answering assistant.

Answer the question using only the supplied document context.
If the context does not contain enough information, clearly say
that the answer cannot be determined from the provided documents.

Do not invent facts or claim that unsupported information
appears in the documents.

Document context:
{context}

Question:
{question}

Answer:
"""
)

answer_chain = prompt | llm | StrOutputParser()


# ============================================================
# 5. DOCUMENT INGESTION AND VECTOR STORE CREATION
# ============================================================

def build_vector_store():
    if not PDF_DIRECTORY.exists():
        raise FileNotFoundError(
            f"PDF directory not found: {PDF_DIRECTORY.resolve()}"
        )

    pdf_files = list(PDF_DIRECTORY.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {PDF_DIRECTORY.resolve()}"
        )

    # Load PDF documents and their metadata.
    loader = PyPDFDirectoryLoader(str(PDF_DIRECTORY))
    documents = loader.load()

    if not documents:
        raise ValueError(
            "No readable PDF pages were loaded."
        )

    # Split documents into overlapping text chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "No text chunks were created from the PDFs."
        )

    # Generate embeddings and build the local FAISS index.
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vector_store, len(documents), len(chunks)


# Store the vector store for this Streamlit session.
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


# ============================================================
# 6. INGESTION BUTTON
# ============================================================

with st.sidebar:
    st.subheader("Document ingestion")

    st.write(f"PDF directory: `{PDF_DIRECTORY}`")
    st.write(f"Chat model: `{CHAT_MODEL}`")
    st.write(f"Embedding model: `{EMBEDDING_MODEL}`")

    if st.button("Build / Rebuild Vector Store", use_container_width=True):
        try:
            with st.spinner(
                "Loading PDFs, generating embeddings, and building FAISS..."
            ):
                start_time = time.perf_counter()

                (
                    vector_store,
                    document_count,
                    chunk_count,
                ) = build_vector_store()

                st.session_state.vector_store = vector_store

                elapsed = time.perf_counter() - start_time

            st.success("Vector store is ready!")
            st.write(f"PDF pages loaded: {document_count}")
            st.write(f"Text chunks created: {chunk_count}")
            st.write(f"Indexing time: {elapsed:.2f} seconds")

        except Exception as exc:
            st.error(f"Document ingestion failed: {exc}")


# ============================================================
# 7. QUESTION ANSWERING
# ============================================================

question = st.text_input(
    "Ask a question about your documents",
    placeholder="For example: What does the census say about population?",
)

if question:
    if st.session_state.vector_store is None:
        st.warning(
            "Build the vector store from the sidebar before asking questions."
        )
        st.stop()

    try:
        with st.spinner("Searching documents and generating an answer..."):
            start_time = time.perf_counter()

            # Retrieve the most relevant document chunks.
            retriever = st.session_state.vector_store.as_retriever(
                search_kwargs={"k": RETRIEVAL_K}
            )

            retrieved_docs = retriever.invoke(question)

            if not retrieved_docs:
                st.warning("No relevant document chunks were found.")
                st.stop()

            # Combine retrieved text into the LLM context.
            context = "\n\n".join(
                doc.page_content for doc in retrieved_docs
            )

            # Generate an answer using the NVIDIA-hosted LLM.
            answer = answer_chain.invoke(
                {
                    "context": context,
                    "question": question,
                }
            )

            elapsed = time.perf_counter() - start_time

        st.subheader("Answer")
        st.markdown(answer)

        st.caption(f"Retrieval + generation time: {elapsed:.2f} seconds")

        # Display source chunks and their metadata.
        with st.expander("Retrieved document chunks"):
            for index, doc in enumerate(retrieved_docs, start=1):
                st.markdown(f"**Source chunk {index}**")
                st.write(doc.page_content)
                st.caption(
                    f"Source: {doc.metadata.get('source', 'Unknown')} | "
                    f"Page: {doc.metadata.get('page', 'Unknown')}"
                )
                st.divider()

    except Exception as exc:
        st.error(f"Question answering failed: {exc}")
