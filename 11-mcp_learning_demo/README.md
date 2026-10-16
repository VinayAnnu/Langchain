# MCP Learning Demo: Two Servers + LLM Client

A beginner-friendly project with two independent MCP servers and one LLM-powered client.

- **Math MCP server:** `add(a, b)` and `multiply(a, b)`
- **Weather MCP server:** `get_current_weather(city)` using Open-Meteo
- **Client:** connects to both servers, discovers their tools, and gives the tools to a LangChain agent powered by Groq.

## Architecture

```text
User question
    |
    v
client.py (LangChain agent + Groq LLM)
    | discovers and calls tools using MCP
    +------------------------+
    |                        |
    v                        v
math_server.py          weather_server.py
add(a, b)               get_current_weather(city)
multiply(a, b)                 |
                               v
                       Open-Meteo API
```

The model chooses a tool based on the user's request. The Python tool implementation runs on the relevant MCP server. MCP standardizes how the client discovers and calls tools.

## Requirements

- Python 3.11+ recommended
- A Groq API key
- Internet access for Groq and live weather lookup

## Setup on macOS (VS Code terminal)

Run these commands from this project folder.

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure your Groq API key

```bash
cp .env.example .env
```

Open `.env` and replace `your_groq_api_key_here` with your real Groq API key. Do not commit `.env` or share the key.

### 3. Run the client

```bash
python client.py
```

The client starts both MCP server processes automatically over stdio. You do not need to start the servers manually in separate terminals.

## Try these questions

```text
Add 19 and 23.
What is 12 multiplied by 8?
What's the current weather in Bengaluru?
Tell me the weather in Paris.
What is 8 times 7, then add 6 to the result?
```

For the last example, the model may call more than one tool if it decides a sequence of tool calls is needed.

## How it works

1. `FastMCP` exposes Python functions as MCP tools on each server.
2. `MultiServerMCPClient` launches both server scripts as subprocesses.
3. `get_tools()` discovers tools from both servers and adapts them to LangChain tools.
4. `create_agent()` gives the model those tools and instructions.
5. The model chooses a tool and produces structured arguments.
6. The MCP adapter sends the tool call to the appropriate server.
7. The server runs the function and returns the result to the agent.
8. The model uses the result to formulate the final answer.

## Files

```text
mcp_learning_demo/
├── client.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── servers/
    ├── math_server.py
    └── weather_server.py
```

## Troubleshooting

- **`GROQ_API_KEY is missing`:** Ensure `.env` exists in the project root and contains the key.
- **Import errors:** Activate `.venv` and run `pip install -r requirements.txt`.
- **Weather lookup fails:** Check your internet connection; the weather server calls Open-Meteo.
- **The model doesn't choose a tool:** Try a direct, explicit request such as `Multiply 12 by 8`. Tool selection is decided by the model, so prompts can affect the decision.
- **Server output:** For stdio transport, avoid printing debug logs to standard output in server scripts. MCP uses stdout for protocol messages.

## Learning exercises

1. Add a `subtract(a, b)` tool to the math server.
2. Add a `divide(a, b)` tool and handle division by zero.
3. Add a third MCP server.
4. Log which tool the model called.
5. Add tests for the math functions.