import truststore
truststore.inject_into_ssl()

import os
import time
from urllib.parse import urlparse

import streamlit as st
from groq import AuthenticationError, RateLimitError
from langchain_community.document_loaders import WebBaseLoader, YoutubeLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
# --------------------------------------------------
# 1. Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Content Summarizer",
    page_icon="🦜",
    layout="centered",
)

st.title("🦜 AI Content Summarizer")
st.write("Summarize a YouTube video or a website using AI.")

# --------------------------------------------------
# 2. Application configuration
# --------------------------------------------------

MODEL_NAME = "openai/gpt-oss-20b"
CHUNK_SIZE = 2_000
CHUNK_OVERLAP = 300
MAX_CHUNKS = 30

# --------------------------------------------------
# 3. Read the Groq API key
# --------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    default_api_key = os.getenv("GROQ_API_KEY", "")

    api_key = st.text_input(
        "Groq API Key",
        value=default_api_key,
        type="password",
        help="Your key is used only to make Groq API requests.",
    )

    st.caption(f"Model: {MODEL_NAME}")

# --------------------------------------------------
# 4. Validate the submitted URL
# --------------------------------------------------

def validate_url(url: str) -> tuple[bool, str]:
    """Validate basic URL structure and supported protocols."""

    try:
        parsed = urlparse(url.strip())

        if parsed.scheme not in {"http", "https"}:
            return False, "The URL must start with http:// or https://."

        if not parsed.hostname:
            return False, "Please enter a valid website URL."

        host = parsed.hostname.lower()

        if host == "localhost" or host.endswith(".localhost"):
            return False, "Localhost URLs are not supported."

        return True, ""

    except ValueError:
        return False, "The URL format is invalid."


def is_youtube_url(url: str) -> bool:
    """Check whether the URL belongs to a recognized YouTube host."""

    host = (urlparse(url).hostname or "").lower()

    return host in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
        "www.youtube-nocookie.com",
    }

# --------------------------------------------------
# 5. Load content from YouTube or a website
# --------------------------------------------------

def load_documents(url: str):
    """Load source content and return LangChain Documents."""

    if is_youtube_url(url):
        loader = YoutubeLoader.from_youtube_url( ##YoutubeLoader depends on transcript availability.if transcript not available then it will not load the content
            url,
            language=["en"], ##This tells the loader to look for an English transcript.
            add_video_info=False, ##This tells the loader not to add additional video information, such as its title and publication date, to the document metadata.
        )
    else:
        loader = WebBaseLoader(
            web_paths=(url,),
            requests_kwargs={"timeout": 20},
            header_template={
                "User-Agent": "Mozilla/5.0 (compatible; ContentSummarizer/1.0)" ##A User-Agent header identifies the client making the request. Some websites use it when deciding how to respond to different clients.
            },
            raise_for_status=True, ##This tells the loader to raise an HTTP error when the response has an unsuccessful status code, such as 404 Not Found or 500 Internal Server Error.
        )

    documents = loader.load()

    if not documents:
        raise ValueError("No readable content was found at this URL.")

    # Remove empty documents before summarization.
    documents = [
        doc for doc in documents
        if doc.page_content and doc.page_content.strip()
    ]

    if not documents:
        raise ValueError(
            "The source returned no text to summarize. "
            "The page may be empty or its transcript unavailable."
        )

    return documents

# --------------------------------------------------
# 6. Split long content into manageable chunks
# --------------------------------------------------

def split_documents(documents):
    """Split documents to keep individual LLM requests manageable."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    if len(chunks) > MAX_CHUNKS:
        raise ValueError(
            f"The source produced {len(chunks)} chunks. "
            f"This app currently supports at most {MAX_CHUNKS}. "
            "Use a shorter source or reduce the amount of content."
        )

    return chunks

# --------------------------------------------------
# 7. Build the summarization chains
# --------------------------------------------------

def create_summarization_chains(api_key: str):
    """Create reusable map and final-summary LCEL chains."""

    llm = ChatGroq(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0,
        max_retries=2,
    )

    map_prompt = ChatPromptTemplate.from_template(
        """
        Summarize the important information in the content below.
        Preserve key facts, names, dates, arguments, and conclusions.
        Do not invent facts.
        Keep the summary concise, around 100 words.

        Content:
        {text}
        """
    )

    final_prompt = ChatPromptTemplate.from_template(
        """
        Create a clear, well-organized summary of the following
        partial summaries.

        Include the main ideas, important facts, and conclusions.
        Remove repetition. Do not add unsupported information.
        Aim for approximately 300 words, if the source contains
        enough information.

        Partial summaries:
        {text}
        """
    )

    # LCEL: prompt -> LLM -> plain text output.
    map_chain = map_prompt | llm | StrOutputParser()
    final_chain = final_prompt | llm | StrOutputParser()

    return map_chain, final_chain


# --------------------------------------------------
# 8. Summarize the content
# --------------------------------------------------

def summarize_documents(documents, map_chain, final_chain):
    """Summarize each chunk, then combine the partial summaries."""

    chunks = split_documents(documents)
    partial_summaries = []

    # Sequential processing helps avoid token-rate-limit spikes.
    progress = st.progress(0, text="Preparing content...") 

    for index, chunk in enumerate(chunks, start=1):
        summary = map_chain.invoke({"text": chunk.page_content})
        partial_summaries.append(summary)

        progress.progress(
            index / len(chunks),
            text=f"Summarizing chunk {index} of {len(chunks)}...",
        )

        # A small pause helps space out requests; it does not
        # guarantee that the account's token-per-minute limit is met.
        if index < len(chunks):
            time.sleep(1)

    progress.progress(1.0, text="Creating the final summary...") ##Streamlit progress bar to 100% and changes its message to “Creating the final summary

    combined_text = "\n\n".join(partial_summaries)

    # If the source was split, the final call combines its summaries.
    final_summary = final_chain.invoke({"text": combined_text})

    progress.empty()
    return final_summary, len(chunks)


# --------------------------------------------------
# 9. Streamlit user interface
# --------------------------------------------------

url = st.text_input(
    "YouTube video or website URL",
    placeholder="https://www.youtube.com/watch?v=...",
)

summarize_clicked = st.button(
    "Summarize content",
    type="primary", ##makes the button the main, visually emphasized button
    use_container_width=True, ##This makes the button expand to the available width of its container.
)


# --------------------------------------------------
# 10. Run the application with error handling
# --------------------------------------------------

if summarize_clicked:
    clean_url = url.strip()

    if not api_key.strip():
        st.error("Enter your Groq API key in the sidebar.")
        st.stop()

    if not clean_url:
        st.error("Enter a YouTube video or website URL.")
        st.stop()

    valid, validation_message = validate_url(clean_url)

    if not valid:
        st.error(validation_message)
        st.stop()

    try:
        with st.spinner("Loading source content..."):
            documents = load_documents(clean_url)

        map_chain, final_chain = create_summarization_chains(
            api_key.strip()
        )

        with st.spinner("Generating your summary..."):
            summary, chunk_count = summarize_documents(
                documents,
                map_chain,
                final_chain,
            )

        st.success("Summary generated successfully.")
        st.subheader("Summary")
        st.write(summary)

        with st.expander("Source details"):
            st.write(f"URL: {clean_url}")
            st.write(f"Documents loaded: {len(documents)}")
            st.write(f"Chunks summarized: {chunk_count}")

    except RateLimitError:
        st.error(
            "Groq's token or request rate limit was reached. "
            "Wait before retrying, reduce the content size, "
            "or check your account's model limits."
        )

    except AuthenticationError:
        st.error(
            "Groq rejected the API key. Check that it is valid "
            "and has not been revoked."
        )

    except ValueError as exc:
        st.error(str(exc))

    except Exception as exc:
        # Log technical details in the server terminal, not the UI.
        print(f"Summarization failed: {type(exc).__name__}: {exc}")
        st.error(
            "The content could not be summarized. Check that the "
            "source is accessible and that its transcript or text "
            "can be loaded, then try again."
        )