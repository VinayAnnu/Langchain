
import truststore
truststore.inject_into_ssl()
import os

from langchain.chat_models import init_chat_model
from langchain.agents.middleware import TodoListMiddleware

from deepagents import create_deep_agent

from tools import tavily_web_search


# --------------------------------------------------
# 1. Initialize the LLM
# --------------------------------------------------
from langchain_ollama import ChatOllama
## Ollama qwen2.5:3b
# Initialize the modern Ollama chat model
model = ChatOllama(
    model="qwen3:8b",
    temperature=0,
    model_kwargs={
        "parallel_tool_calls": True
    }
)

# model = init_chat_model(
#     "groq:openai/gpt-oss-20b",
#     temperature=0, ##makes responses more deterministic.
#     reasoning_format="hidden", ##Hidden does not mean the model stops reasoning. It means the reasoning content is not returned in the visible response.
#     reasoning_effort="low", ##reduces the model's reasoning effort and can reduce token consumption.
#     # model_kwargs={
#     #     "parallel_tool_calls": True
#     # }
# )


# --------------------------------------------------
# 2. Define supervisor instructions
# --------------------------------------------------

system_prompt = """
You are a research supervisor responsible for producing
a comprehensive research report.

Your task is to research:
1. Artificial Intelligence and Machine Learning.
2. Deep Learning and Generative AI.

MANDATORY WORKFLOW:

STEP 1: DELEGATION

You MUST delegate exactly two research tasks:

Task 1:
- Delegate to ai_researcher.
- Research Artificial Intelligence and Machine Learning.

Task 2:
- Delegate to dl_researcher.
- Research Deep Learning and Generative AI.

Do not delegate any additional tasks.
Do not use the general-purpose subagent.

STEP 2: COLLECT RESULTS

Wait until both subagents return their research reports.

You must use BOTH reports.

Do not ignore, replace, or discard either report.

STEP 3: SYNTHESIS

Combine both reports into one comprehensive research document.

The final document MUST contain:

1. Introduction to Artificial Intelligence
2. Machine Learning fundamentals
3. Types of Machine Learning
4. Important ML algorithms
5. Deep Learning fundamentals
6. Neural Network architectures
7. Transformers and Attention
8. Generative AI models
9. Real-world applications
10. Challenges and future directions
11. References and source URLs

Preserve important technical details from both research reports.

Do not generate a generic summary.
Do not invent research findings or source URLs.

STEP 4: SAVE THE REPORT

After completing the synthesis, you MUST call the
write_file tool exactly once.

Save the complete report to:

/reports/ai_research.md

Do not finish your task until the write_file tool
has been called.

STEP 5: FINAL RESPONSE

After saving the report, provide:
- A concise summary of the completed research.
- The report filename.
- Confirmation based on the actual file-writing tool result.

Never claim that the report was saved unless
write_file successfully executes.
"""


# --------------------------------------------------
# 3. Define specialized subagents
# --------------------------------------------------

subagents = [

    {
        "name": "ai_researcher",

        "description": (
            "Specialist researcher for Artificial Intelligence "
            "and Machine Learning. Researches AI fundamentals, "
            "ML algorithms, supervised and unsupervised learning."
        ),

        "system_prompt": """
        You are an expert AI and Machine Learning researcher.

        MANDATORY TOOL USAGE:

        1. You MUST use the registered tool named
           `tavily_web_search` to conduct internet research.
        2. Never refer to this tool as `web_search`.
        3. Perform at least two relevant searches.
        4. Base your research on the returned search results.
        5. Include source URLs in your report.
        6. Never fabricate sources or research findings.

        RESEARCH TOPICS:

        - Artificial Intelligence fundamentals.
        - Machine Learning fundamentals.
        - Supervised Learning.
        - Unsupervised Learning.
        - Reinforcement Learning.
        - Important ML algorithms.
        - Real-world applications.
        - Ethical considerations.

        Return a structured research report.
        """,

        "tools": [tavily_web_search]
    },

    {
        "name": "dl_researcher",

        "description": (
            "Specialist researcher for Deep Learning, "
            "Neural Networks, Transformers and Generative AI."
        ),

        "system_prompt": """
        You are an expert Deep Learning and Generative AI researcher.

        MANDATORY TOOL USAGE:

        1. You MUST use the registered tool named
           `tavily_web_search` to conduct internet research.
        2. Never refer to this tool as `web_search`.
        3. Perform at least two relevant searches.
        4. Base your research on the returned search results.
        5. Include source URLs in your report.
        6. Never fabricate sources or research findings.

        RESEARCH TOPICS:

        - Deep Learning fundamentals.
        - Artificial Neural Networks.
        - CNNs and RNNs.
        - Transformers and Attention.
        - Generative AI.
        - Large Language Models.
        - Diffusion Models.
        - Real-world applications.

        Return a structured research report.
        """,

        "tools": [tavily_web_search]
    }
]


# --------------------------------------------------
# 4. Create the Deep Agent
# --------------------------------------------------

agent = create_deep_agent(

    model=model,

    #tools=[tavily_web_search],

    system_prompt=system_prompt,

    subagents=subagents,

    #Enable explicit planning using write_todos
    middleware=[
        TodoListMiddleware()
    ],

    debug=True
)


# --------------------------------------------------
# 5. Execute the research task
# --------------------------------------------------

def main():

    try:

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": """
                            Research AI, ML, Deep Learning and Generative AI.
                            You MUST use my custom Tavily search tool named
                            `tavily_web_search` for all internet-based research.

                            Delegate exactly two tasks:
                            
                            Task 1:
                            Use ai_researcher to research AI and ML.
                            
                            Task 2:
                            Use dl_researcher to research Deep Learning and Generative AI.
                            
                            Wait for both research results.
                            
                            Combine their findings into one report and save it at:
                            
                            /reports/ai_research.md
                            """
                    }
                ]
            }
        )

        print("\n========== FINAL RESPONSE ==========\n")

        print(result["messages"][-1].content)

        print("\n========== MESSAGE HISTORY ==========\n")

        for index, message in enumerate(result["messages"]):

            print(f"\n--- MESSAGE {index} ---")

            print("Type:", type(message).__name__)

            print("Name:", getattr(message, "name", None))

            print("Tool Calls:", getattr(message, "tool_calls", None))

            print("Content:", str(message.content)[:500])

    except Exception as error:

        print("\n========== EXECUTION ERROR ==========\n")

        print(type(error).__name__)

        print(error)


if __name__ == "__main__":
    main()