
from crewai_tools import YoutubeChannelSearchTool

# Use a local Ollama embedding model instead of OpenAI embeddings.This configuration is intended to make the tool use Ollama for generating document embeddings. It does not change the LLM that your agents use to research or write the blog.
rag_config = {
    "embedding_model": {
        "provider": "ollama",
        "config": {
            "model_name": "nomic-embed-text",
            "url": "http://localhost:11434/api/embeddings",
        },
    }
}

youtube_search_tool = YoutubeChannelSearchTool(
    youtube_channel_handle="https://www.youtube.com/user/krishnaik06",
    collection_name="krishnaik_ollama_rag", ## A collection is similar to a table or logical grouping in a database. In this case, it stores indexed video content and its associated vector representations.
    config=rag_config,
)
