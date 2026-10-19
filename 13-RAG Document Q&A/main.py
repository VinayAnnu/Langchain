import truststore ## added due to SSL error in Groq client. it is primarily due to Zscaler SSL interception installed by company. This is a workaround to fix the SSL error.
truststore.inject_into_ssl() ## added due to SSL error in Groq client. it is primarily due to Zscaler SSL interception. This is a workaround to fix the SSL error.

import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR.parent / ".env")

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY is missing. Please check your .env file.")
    st.stop()

# --------------------------------------------------
# 2. Initialize models
# --------------------------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=groq_api_key
)



# --------------------------------------------------
# 3. Define the prompt
# --------------------------------------------------


prompt = ChatPromptTemplate.from_template("""
You are an AI assistant that answers questions about research papers.

Use the retrieved context to answer the user's question.

Instructions:
- Identify relevant information across all context chunks.
- Combine related information to form a coherent answer.
- Answer the question directly and clearly.
- Do not require the context to contain the exact wording of the question.
- If the context is insufficient to answer the question, clearly state
  what information is missing.
- Do not invent facts unsupported by the context.

Context:
{context}

Question:
{input}

Answer:
""")



# --------------------------------------------------
# 4. Create vector embeddings
# --------------------------------------------------

def create_vector_embedding():
    pdf_directory =PROJECT_DIR / "research_papers"

    if not pdf_directory.exists():
        st.error(f"PDF directory not found: {pdf_directory}")
        return False

    try:
        with st.spinner("Loading and processing PDF documents..."):

            loader = PyPDFDirectoryLoader(str(pdf_directory))
            docs = loader.load()

            if not docs:
                st.error(
                    "No PDF documents found or no pages could be loaded."
                )
                return False

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            final_documents = text_splitter.split_documents(docs)

            if not final_documents:
                st.error("No text chunks were extracted from the PDFs.")
                return False

            vectors = FAISS.from_documents(
                final_documents,
                embeddings
            )

            st.session_state["vectors"] = vectors

        st.success(
            f"Vector database created successfully! "
            f"Processed {len(docs)} pages into "
            f"{len(final_documents)} chunks."
        )
        return True

    except Exception as e:
        st.error(f"Error creating vector database: {e}")
        return False



# --------------------------------------------------
# 5. Streamlit interface
# --------------------------------------------------

st.title("RAG Document Q&A with Groq and Ollama")

user_prompt = st.text_input(
    "Enter your query about the research papers"
)

if st.button("Document Embedding"):
    create_vector_embedding()

# --------------------------------------------------
# 6. Retrieve documents and generate an answer
# --------------------------------------------------

if user_prompt:

    if "vectors" not in st.session_state:
        st.warning(
            "Please click 'Document Embedding' first "
            "to process your PDF documents."
        )

    else:
        try:
            document_chain = create_stuff_documents_chain(
                llm,
                prompt
            )

            retriever = st.session_state["vectors"].as_retriever(
                search_kwargs={"k": 4}
            )

            retrieval_chain = create_retrieval_chain(
                retriever,
                document_chain
            )

            with st.spinner("Searching documents and generating answer..."):

                start = time.perf_counter()

                response = retrieval_chain.invoke(
                    {"input": user_prompt}
                )

                elapsed = time.perf_counter() - start

            st.write(f"Response time: {elapsed:.2f} seconds")

            st.subheader("Answer")
            st.write(response["answer"])

            # Display retrieved document chunks
            with st.expander("Document Similarity Search"):

                for i, doc in enumerate(response["context"], start=1):
                    st.markdown(f"**Document Chunk {i}**")
                    st.write(doc.page_content)

                    if doc.metadata:
                        st.caption(str(doc.metadata))

                    st.divider()

        except Exception as e:
            st.error(f"Error while generating the answer: {e}")
