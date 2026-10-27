Complete implementation using Streamlit and Ollama
We'll use a simple, maintainable approach:
- Keep all messages in Streamlit session state for the visible transcript.
- Keep a separate, smaller list of messages for model context.
- When the context becomes too long, summarize older turns and retain recent turns.
- Send only the summary, recent messages, system prompt, and current question to Ollama.
- Stream the generated answer into the UI.
- Add a New chat button to clear the current conversation.
This version uses character limits as a lightweight guardrail, not an exact token counter. Different languages and code snippets can use different numbers of tokens per character.

What memory strategy should we use?
start with rolling summary + recent turns. Later, when you want CodeGuru to understand a large repository, add code retrieval (RAG) as a separate capability.

-----------
## configures the model's context window to 8,192 tokens, which must accommodate both the input context and the generated output.
-----------

Important improvements before using this in production
The implementation above is a good learning version, but there are three limitations to address next.
- Context size: MAX_MESSAGE_CHARS limits each message, but eight messages of 6,000 characters plus a summary can still exceed an 8,192-token context. For a more conservative starting point, reduce MAX_MESSAGE_CHARS to around 2,000–3,000 and measure actual token usage where possible. Character counts are only a rough approximation.
- Summary accuracy: Summaries can lose exact code details. If a follow-up depends on an earlier implementation, preserve important code snippets separately or retrieve the relevant earlier turn instead of relying solely on a summary.
- Persistent storage: Session state is not long-term storage. If you want to reopen old conversations after a browser session ends or the application restarts, add SQLite or another database.
One further architectural point: do not confuse chat memory with repository knowledge. For CodeGuru to answer questions about a large codebase, use RAG to retrieve relevant files and functions. Keep conversation memory for the user's intent and follow-up context, and use code retrieval for technical facts from the repository.

This gives you a sensible progression: local Ollama → Streamlit chat → bounded conversation memory → persistent history → repository-aware RAG.