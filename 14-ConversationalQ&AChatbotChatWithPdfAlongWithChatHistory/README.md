1. PDF upload and processing
PyPDFLoader → Text splitter


2. Embeddings and vector storage
HuggingFace Embeddings → Chroma


3. Chat history
ChatMessageHistory → previous questions and answers


4. History-aware retrieval
Rewrites follow-up questions before searching Chroma


5. Answer generation
Groq-hosted LLM + retrieved context + chat history

-------
The biggest addition compared with your previous project is the history-aware retriever and chat-history management.
-------
Library or class	                Purpose
streamlit	                        Builds the web interface
PyPDFLoader	                        Loads text and metadata from uploaded PDFs
RecursiveCharacterTextSplitter	    Splits long documents into smaller chunks
HuggingFaceEmbeddings	            Converts text into numerical vectors
Chroma	                            Stores vectors and retrieves similar documents
ChatGroq	                        Provides access to a chat LLM hosted by Groq
ChatPromptTemplate	                Defines instructions and messages for the LLM
MessagesPlaceholder	                Inserts a list of chat messages into a prompt
create_history_aware_retriever	    Uses chat history to make a follow-up question suitable for retrieval
create_stuff_documents_chain	    Passes retrieved documents to the LLM
create_retrieval_chain	            Connects retrieval to answer generation
ChatMessageHistory	                Stores conversation messages
BaseChatMessageHistory	            Base interface for chat-history implementations
RunnableWithMessageHistory	        Wraps a chain to load and update chat history
--------------------

Full execution flow: an example
Suppose you upload a PDF about the Transformer architecture and ask these questions in the same session.
FIRST QUESTION
User: What is the Transformer architecture?

Retrieval
Chroma retrieves chunks describing the Transformer.
Generation
Groq answers using the retrieved context.
History
The question and answer are stored.


SECOND QUESTION
User: What are its advantages?

History-aware retrieval
The LLM uses the first exchange to rewrite the question as “What are the advantages of the Transformer architecture?”
Retrieval
Chroma searches for relevant chunks using the rewritten question.
Generation
Groq answers using the retrieved context, the latest question, and the conversation history.
History
The new question and answer are added to the session history.


This illustrates the key difference between your original RAG application and conversational RAG.
The original application searches using the latest question alone. The conversational version can use the preceding conversation to interpret follow-up questions.
-------------------
The most important takeaway: conversational RAG combines two types of context:
- Document context — information retrieved from your uploaded PDFs.
- Conversation context — previous questions and answers that help interpret the latest question.
The history-aware retriever uses conversation context to improve document retrieval, while the answer-generation chain uses retrieved documents and conversation history to produce a relevant response.
----------------
Two important things to know
1. Chat history is still in memory. It persists across Streamlit reruns within the same session, but not reliably across server restarts or browser sessions. A production app should use persistent history storage if conversations must survive those events.
2. Chroma is still in-memory in this version. The vector store is retained in Streamlit session state, so the PDFs do not need to be re-embedded on every question. If the app restarts or the session state is lost, you'll need to process the PDFs again. We can add persistent Chroma storage later if you want.