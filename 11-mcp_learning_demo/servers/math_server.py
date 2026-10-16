"""MCP server that exposes two tools: addition and multiplication.
Its job is to make these Python functions available to an MCP client, such as the LangChain client."""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("math-server")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers and return their sum."""
    return a + b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return their product."""
    return a * b


if __name__ == "__main__":
    # stdio is intended for an MCP client to launch and communicate with this process.
    mcp.run(transport="stdio")
